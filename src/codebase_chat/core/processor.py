import asyncio
from pathlib import Path
from typing import List, Optional, Type, Any, Callable, Tuple
import git
import time
import numpy as np
from mcp.server.fastmcp.server import Context
from dataclasses import dataclass
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
from ..models.code_chunk import CodeChunk, RepoInfo
from ..strategies.base import BaseChunkStrategy
from ..providers.base import BaseEmbeddingProvider, BaseRerankProvider, BaseTranslatorProvider
from ..middleware.base import BaseMiddleware

@dataclass
class ProcessingStats:
    """处理统计信息"""
    total_files: int = 0
    total_chunks: int = 0
    total_lines: int = 0
    processing_time: float = 0.0
    embedding_time: float = 0.0
    expected_total_files: int = 0
    milvus_write_time: float = 0.0
    milvus_read_time: float = 0.0
    milvus_connect_time: float = 0.0
    milvus_flush_time: float = 0.0
    @property
    def files_per_second(self) -> float:
        return self.total_files / self.processing_time if self.processing_time > 0 else 0
        
    @property
    def chunks_per_second(self) -> float:
        return self.total_chunks / self.embedding_time if self.embedding_time > 0 else 0
        
    @property
    def lines_per_second(self) -> float:
        return self.total_lines / self.processing_time if self.processing_time > 0 else 0
        
    @property
    def progress_percentage(self) -> float:
        return (self.total_files / self.expected_total_files * 100) if self.expected_total_files > 0 else 0

    @property
    def milvus_write_ops_per_second(self) -> float:
        return self.total_chunks / self.milvus_write_time if self.milvus_write_time > 0 else 0
    
    @property
    def milvus_read_ops_per_second(self) -> float:
        return self.total_chunks / self.milvus_read_time if self.milvus_read_time > 0 else 0
    
    def get_performance_report(self) -> dict:
        """获取性能报告"""
        return {
            "文件处理": {
                "总文件数": self.total_files,
                "总代码块数": self.total_chunks,
                "总行数": self.total_lines,
                "处理进度": f"{self.progress_percentage:.2f}%",
                "每秒处理文件数": f"{self.files_per_second:.2f}",
                "每秒处理行数": f"{self.lines_per_second:.2f}"
            },
            "嵌入生成": {
                "嵌入时间": f"{self.embedding_time:.2f}秒",
                "每秒嵌入块数": f"{self.chunks_per_second:.2f}"
            },
            "Milvus性能": {
                "连接时间": f"{self.milvus_connect_time:.4f}秒",
                "写入时间": f"{self.milvus_write_time:.2f}秒",
                "读取时间": f"{self.milvus_read_time:.2f}秒",
                "每秒写入操作": f"{self.milvus_write_ops_per_second:.2f}",
                "每秒读取操作": f"{self.milvus_read_ops_per_second:.2f}"
            },
            "总体性能": {
                "总处理时间": f"{self.processing_time:.2f}秒"
            }
        }

class CodeProcessor:
    """代码处理器的核心类"""
    
    def __init__(
        self,
        db_path: str,
        chunk_strategy: BaseChunkStrategy,
        embedding_provider: BaseEmbeddingProvider,
        rerank_provider: Optional[BaseRerankProvider] = None,
        translator_provider: Optional[BaseTranslatorProvider] = None,
        table_name: str = "code_chunks",
        middlewares: Optional[List[BaseMiddleware]] = None,
        progress_callback: Optional[Callable[[ProcessingStats], None]] = None,
    ):
        # 连接到Milvus服务
        # 这里假设db_path格式为"host:port"，也可以是其他连接字符串格式
        self.stats = ProcessingStats()
        connect_start = time.time()
        if ":" in db_path:
            host, port = db_path.split(":")
            connections.connect("default", host=host, port=port)
        else:
            # 本地默认连接
            connections.connect("default", host="localhost", port="19530")
        self.stats.milvus_connect_time = time.time() - connect_start
            
        self.chunk_strategy = chunk_strategy
        self.embedding_provider = embedding_provider
        self.rerank_provider = rerank_provider
        self.translator_provider = translator_provider
        self.table_name = table_name
        self.middlewares = middlewares or []
        self.progress_callback = progress_callback
        self._last_callback_time = 0
        
        # 创建集合（如果不存在）
        self._create_collection_if_not_exists()
            
    def _create_collection_if_not_exists(self):
        """创建Milvus集合（如果不存在）"""
        if utility.has_collection(self.table_name):
            self.collection = Collection(self.table_name)
            self.collection.load()
        else:
            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="repo_name", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="start_line", dtype=DataType.INT32),
                FieldSchema(name="end_line", dtype=DataType.INT32),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="branch", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=255, max_capacity=100),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
            ]
            
            # 创建集合
            schema = CollectionSchema(fields)
            self.collection = Collection(self.table_name, schema)
            
            # 创建索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            self.collection.load()
            
    def _get_repo_info(self, path: Path) -> Optional[str]:
        """获取Git仓库信息"""
        try:
            repo = git.Repo(path, search_parent_directories=True)
            return repo.active_branch.name
        except (git.InvalidGitRepositoryError, git.NoSuchPathError):
            return None
            
    async def process_file(self, file_path: Path, repo_info: RepoInfo) -> List[CodeChunk]:
        """处理单个文件
        
        Args:
            file_path: 文件的完整路径
            repo_path: 仓库根目录路径
            repo_name: 仓库名称
        """
        start_time = time.time()
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # 读取文件内容
        content = file_path.read_text(encoding='utf-8')
        
        # 切分代码块
        chunks = list(self.chunk_strategy.chunk_file(file_path, content, repo_info))
            
        # 应用中间件
        for middleware in self.middlewares:
            chunks = await middleware.process(chunks)
        
        # 生成嵌入向量
        embedding_start = time.time()
        chunks = await self.embedding_provider.embed_chunks(chunks)
        embedding_time = time.time() - embedding_start
        
        # 更新统计信息
        self.stats.total_chunks += len(chunks)
        self.stats.total_lines += sum(chunk.end_line - chunk.start_line + 1 for chunk in chunks)
        self.stats.embedding_time += embedding_time
        self.stats.processing_time += time.time() - start_time
        
        # 检查是否需要触发进度回调
        current_time = time.time()
        if self.progress_callback and (current_time - self._last_callback_time >= 5):
            self.progress_callback(self.stats)
            self._last_callback_time = current_time
        
        return chunks
        
    async def index_files(self, files: List[Path], repo_path: Path) -> ProcessingStats:
        """索引多个文件
        
        Args:
            files: 要处理的文件路径列表
            repo_path: 仓库根目录路径
            repo_name: 仓库名称
        """
        start_time = time.time()
        
        # 重置统计信息
        self.stats = ProcessingStats(expected_total_files=len(files))
        self._last_callback_time = time.time()

        branch = self._get_repo_info(repo_path)
        repo_name = repo_path.name
        repo_info = RepoInfo(repo_path=repo_path, repo_name=repo_name, branch=branch)
        
        for file_path in files:
            if file_path.is_file():  # 确保只处理文件
                chunks = await self.process_file(file_path, repo_info)
                if len(chunks) == 0:
                    continue
                    
                # 准备批量插入数据
                entities = []
                for chunk in chunks:
                    # 查询是否存在相同内容
                    expr = f'repo_name == "{chunk.repo_name}" and file_path == "{chunk.file_path}" and start_line == {chunk.start_line} and end_line == {chunk.end_line}'
                    milvus_read_start = time.time()
                    results = self.collection.query(expr=expr, output_fields=["content", "branch"])
                    self.stats.milvus_read_time += time.time() - milvus_read_start
                    
                    if not results:
                        # 不存在，添加新记录
                        chunk_data = chunk.to_dict()
                        entities.append({
                            "repo_name": chunk_data["repo_name"],
                            "file_path": chunk_data["file_path"],
                            "start_line": chunk_data["start_line"],
                            "end_line": chunk_data["end_line"],
                            "content": chunk_data["content"],
                            "branch": chunk_data["branch"],
                            "embedding": chunk_data["embedding"]
                        })
                    else:
                        # 已存在，更新分支信息
                        matched = None
                        for result in results:
                            if result["content"] == chunk.content:
                                matched = result
                                break
                                
                        if not matched:
                            # 内容不同，添加新记录
                            chunk_data = chunk.to_dict()
                            entities.append({
                                "repo_name": chunk_data["repo_name"],
                                "file_path": chunk_data["file_path"],
                                "start_line": chunk_data["start_line"],
                                "end_line": chunk_data["end_line"],
                                "content": chunk_data["content"],
                                "branch": chunk_data["branch"],
                                "embedding": chunk_data["embedding"]
                            })
                        elif repo_info.branch not in matched['branch']:
                            # 更新分支信息
                            updated_branch = list(set(matched["branch"] + [repo_info.branch]))
                            self.collection.delete(f'id == {matched["id"]}')
                            chunk_data = chunk.to_dict()
                            entities.append({
                                "repo_name": chunk_data["repo_name"],
                                "file_path": chunk_data["file_path"],
                                "start_line": chunk_data["start_line"],
                                "end_line": chunk_data["end_line"],
                                "content": chunk_data["content"],
                                "branch": updated_branch,
                                "embedding": chunk_data["embedding"]
                            })
                
                # 批量插入数据
                if entities:
                    milvus_write_start = time.time()
                    self.collection.insert(entities)
                    self.stats.milvus_write_time += time.time() - milvus_write_start
                    
                self.stats.total_files += 1
                
        self.stats.processing_time = time.time() - start_time

        milvus_flush_start = time.time()
        self.collection.flush()
        self.stats.milvus_flush_time += time.time() - milvus_flush_start
        
        # 确保最后一次进度更新
        if self.progress_callback:
            self.progress_callback(self.stats)
            
        return self.stats
        
    async def search(self, query: str, limit: int = 5, context: Optional[Context] = None) -> Tuple[List[Tuple[CodeChunk, float]], ProcessingStats]:
        """搜索相似代码块
        
        使用多阶段搜索策略：
        1. 查询翻译：如果有翻译器，将查询翻译成英文
        2. 向量召回：分别使用原始查询和翻译后的查询进行召回
        3. 重排序：合并召回结果后根据原始查询进行重排序
        
        Returns:
            tuple: 包含两个元素的元组:
                - 按相关性排序的(CodeChunk, 相似度分数)元组列表
                - 处理统计信息
        """
        # 第一阶段：查询翻译
        translated = None
        lang = None
        if self.translator_provider:
            lang = await self.translator_provider.detect_language(query)
            if lang != "en":
                if context:
                    await context.info(f"翻译查询: {query} 到英文")
                translated = await self.translator_provider.translate(
                    query,
                    source_lang=lang,
                    target_lang="en",
                    preserve_format=True
                )
        
        # 第二阶段：向量召回
        initial_limit = limit * 4  # 每个查询获取4倍的候选结果
        
        # 使用原始查询召回
        if context:
            await context.info(f"使用原始查询召回: {query}")
        query_embedding = await self._get_query_embedding(query)
        
        self.collection.load()
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        milvus_read_start = time.time()
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=initial_limit,
            output_fields=["repo_name", "file_path", "start_line", "end_line", "content", "branch"]
        )
        self.stats.milvus_read_time += time.time() - milvus_read_start
        
        candidates = []
        for hit in results[0]:
            candidate = {
                "repo_name": hit.entity.get("repo_name"),
                "file_path": hit.entity.get("file_path"),
                "start_line": hit.entity.get("start_line"),
                "end_line": hit.entity.get("end_line"),
                "content": hit.entity.get("content"),
                "branch": hit.entity.get("branch"),
                "embedding": query_embedding,  # 原始结果中不包含向量
                "score": hit.score
            }
            candidates.append(candidate)
        
        # 如果有翻译结果，使用翻译后的查询再次召回
        if translated:
            if context:
                await context.info(f"使用翻译后的查询召回: {translated}")
            translated_embedding = await self._get_query_embedding(translated)
            
            milvus_read_start = time.time()
            results = self.collection.search(
                data=[translated_embedding],
                anns_field="embedding",
                param=search_params,
                limit=initial_limit,
                output_fields=["repo_name", "file_path", "start_line", "end_line", "content", "branch"]
            )
            self.stats.milvus_read_time += time.time() - milvus_read_start
            
            # 合并结果，去重
            seen = {f'{c["file_path"]}:{c["start_line"]}-{c["end_line"]}' for c in candidates}
            for hit in results[0]:
                identifier = f'{hit.entity.get("file_path")}:{hit.entity.get("start_line")}-{hit.entity.get("end_line")}'
                if identifier not in seen:
                    candidate = {
                        "repo_name": hit.entity.get("repo_name"),
                        "file_path": hit.entity.get("file_path"),
                        "start_line": hit.entity.get("start_line"),
                        "end_line": hit.entity.get("end_line"),
                        "content": hit.entity.get("content"),
                        "branch": hit.entity.get("branch"),
                        "embedding": translated_embedding,
                        "score": hit.score
                    }
                    candidates.append(candidate)
                    seen.add(identifier)
        
        if not candidates:
            return [], self.stats
            
        # 第三阶段：重排序
        if self.rerank_provider:
            # 使用专门的重排序提供者
            # 使用原始查询进行重排序
            if context:
                await context.info(f"使用原始查询进行重排序")
            reranked_results = await self.rerank_provider.rerank(query, candidates)
        else:
            # 使用默认的余弦相似度重排序
            reranked_results = self._default_rerank(candidates, query_embedding)
        
        # 取前limit个结果
        final_results = reranked_results[:limit]
        return [(CodeChunk.from_dict(result[0]), result[1]) for result in final_results], self.stats
        
    def _default_rerank(self, candidates: List[dict], query_embedding: List[float]) -> List[Tuple[dict, float]]:
        """默认的重排序方法，使用余弦相似度
        
        Args:
            candidates: 候选结果列表
            query_embedding: 查询的嵌入向量
            
        Returns:
            按相似度排序的(结果, 分数)元组列表
        """
        # 转换为numpy数组以提高计算效率
        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        
        scored_results = []
        for candidate in candidates:
            # Milvus搜索结果已包含分数，直接使用
            if "score" in candidate:
                scored_results.append((candidate, candidate["score"]))
            else:
                # 计算余弦相似度
                candidate_vec = np.array(candidate['embedding'])
                candidate_norm = np.linalg.norm(candidate_vec)
                
                if query_norm == 0 or candidate_norm == 0:
                    similarity = 0
                else:
                    similarity = np.dot(query_vec, candidate_vec) / (query_norm * candidate_norm)
                
                scored_results.append((candidate, similarity))
            
        # 按相似度降序排序
        return sorted(scored_results, key=lambda x: x[1], reverse=True)
        
    async def _get_query_embedding(self, query: str) -> List[float]:
        """获取查询文本的嵌入向量"""
        dummy_chunk = CodeChunk(
            file_path="",
            start_line=0,
            end_line=0,
            content=query,
            repo_name="",
            branch=""
        )
        embedded_chunks = await self.embedding_provider.embed_chunks([dummy_chunk])
        return embedded_chunks[0].embedding 

    def close(self):
        """关闭Milvus连接"""
        try:
            connections.disconnect("default")
        except Exception as e:
            print(f"关闭Milvus连接时发生错误: {e}")

    def __del__(self):
        """析构函数，确保关闭连接"""
        self.close() 