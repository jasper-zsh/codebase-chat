import asyncio
from pathlib import Path
from typing import List, Optional, Type, Any, Callable, Tuple
import git
import lancedb
import pyarrow as pa
import time
import numpy as np
from dataclasses import dataclass
from ..models.code_chunk import CodeChunk
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
        progress_callback: Optional[Callable[[ProcessingStats], None]] = None
    ):
        self.db = lancedb.connect(db_path)
        self.chunk_strategy = chunk_strategy
        self.embedding_provider = embedding_provider
        self.rerank_provider = rerank_provider
        self.translator_provider = translator_provider
        self.table_name = table_name
        self.middlewares = middlewares or []
        self.stats = ProcessingStats()
        self.progress_callback = progress_callback
        self._last_callback_time = 0
        
        schema = pa.schema([
            ("file_path", pa.string()),
            ("start_line", pa.int32()),
            ("end_line", pa.int32()),
            ("content", pa.string()),
            ("branch", pa.string()),
            ("embedding", pa.list_(pa.float32(), list_size=768)),
        ])
        self.db.create_table(table_name, schema=schema, exist_ok=True)
            
    def _get_repo_info(self, path: Path) -> Optional[str]:
        """获取Git仓库信息"""
        try:
            repo = git.Repo(path, search_parent_directories=True)
            return repo.active_branch.name
        except (git.InvalidGitRepositoryError, git.NoSuchPathError):
            return None
            
    async def process_file(self, file_path: Path) -> List[CodeChunk]:
        """处理单个文件"""
        start_time = time.time()
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # 读取文件内容
        content = file_path.read_text(encoding='utf-8')
        
        # 获取Git信息
        branch = self._get_repo_info(file_path)
        
        # 切分代码块
        chunks = list(self.chunk_strategy.chunk_file(file_path, content))
        
        # 设置分支信息
        for chunk in chunks:
            chunk.branch = branch
            
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
        
    async def index_files(self, files: List[Path]) -> ProcessingStats:
        """索引多个文件"""
        start_time = time.time()
        table = self.db.open_table(self.table_name)
        
        # 重置统计信息
        self.stats = ProcessingStats(expected_total_files=len(files))
        self._last_callback_time = time.time()
        
        for file_path in files:
            chunks = await self.process_file(file_path)
            if len(chunks) == 0:
                continue
            # 转换为字典并存储
            records = [chunk.to_dict() for chunk in chunks]
            table.add(records)
            self.stats.total_files += 1
            
        self.stats.processing_time = time.time() - start_time
        
        # 确保最后一次进度更新
        if self.progress_callback:
            self.progress_callback(self.stats)
            
        return self.stats
        
    async def search(self, query: str, limit: int = 5) -> List[Tuple[CodeChunk, float]]:
        """搜索相似代码块
        
        使用多阶段搜索策略：
        1. 查询翻译：如果有翻译器，将查询翻译成英文
        2. 向量召回：分别使用原始查询和翻译后的查询进行召回
        3. 重排序：合并召回结果后根据原始查询进行重排序
        """
        table = self.db.open_table(self.table_name)
        
        # 第一阶段：查询翻译
        translated = None
        lang = await self.translator_provider.detect_language(query)
        if lang != "en":
            translated = await self.translator_provider.translate(
                query,
                source_lang=lang,
                target_lang="en",
                preserve_format=True
            )
        
        # 第二阶段：向量召回
        initial_limit = limit * 4  # 每个查询获取4倍的候选结果
        
        # 使用原始查询召回
        query_embedding = await self._get_query_embedding(query)
        candidates = table.search(
            query_embedding,
            vector_column_name="embedding"
        ).limit(initial_limit).to_list()
        
        # 如果有翻译结果，使用翻译后的查询再次召回
        if translated:
            translated_embedding = await self._get_query_embedding(translated)
            translated_candidates = table.search(
                translated_embedding,
                vector_column_name="embedding"
            ).limit(initial_limit).to_list()
            
            # 合并结果，去重
            seen = {f'{c["file_path"]}:{c["start_line"]}-{c["end_line"]}' for c in candidates}
            for c in translated_candidates:
                if f'{c["file_path"]}:{c["start_line"]}-{c["end_line"]}' not in seen:
                    candidates.append(c)
                    seen.add(f'{c["file_path"]}:{c["start_line"]}-{c["end_line"]}')
        
        if not candidates:
            return []
            
        # 第三阶段：重排序
        if self.rerank_provider:
            # 使用专门的重排序提供者
            # 使用原始查询进行重排序
            reranked_results = await self.rerank_provider.rerank(query, candidates)
        else:
            # 使用默认的余弦相似度重排序
            reranked_results = self._default_rerank(candidates, query_embedding)
        
        # 取前limit个结果
        final_results = reranked_results[:limit]
        return [(CodeChunk.from_dict(result[0]), result[1]) for result in final_results]
        
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
            content=query
        )
        embedded_chunks = await self.embedding_provider.embed_chunks([dummy_chunk])
        return embedded_chunks[0].embedding 