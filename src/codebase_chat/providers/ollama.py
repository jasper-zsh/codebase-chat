from typing import List, Dict
import asyncio
import aiohttp
import time
from urllib.parse import urljoin
from rich.progress import Progress, TaskID
import httpx

from ..models.code_chunk import CodeChunk
from .base import BaseEmbeddingProvider

class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """使用Ollama API的嵌入策略"""
    
    def __init__(
        self,
        model: str = "codellama",
        batch_size: int = 10,
        base_url: str = "http://localhost:11434",
    ):
        """
        Args:
            model: Ollama模型名称
            batch_size: 批处理大小
            base_url: Ollama服务的基础URL
        """
        self.model = model
        self.batch_size = batch_size
        self.base_url = base_url.rstrip('/')
        self._session = None
        self.metrics: Dict[str, float] = {
            "total_chunks": 0,
            "total_tokens": 0,
            "total_time": 0,
            "avg_chunks_per_second": 0,
        }
        
    async def _ensure_session(self):
        """确保aiohttp会话已创建"""
        if self._session is None:
            self._session = aiohttp.ClientSession()
            
    async def _close_session(self):
        """关闭aiohttp会话"""
        if self._session is not None:
            await self._session.close()
            self._session = None
            
    async def _get_embedding(self, text: str) -> List[float]:
        """获取单个文本的嵌入向量"""
        await self._ensure_session()
        
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model,
            "prompt": text
        }
        
        start_time = time.time()
        async with self._session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Ollama API error: {error_text}")
                
            data = await response.json()
            self.metrics["total_time"] += time.time() - start_time
            return data['embedding']
        
    async def embed_chunks(
        self,
        chunks: List[CodeChunk],
        progress: Progress = None,
        task_id: TaskID = None
    ) -> List[CodeChunk]:
        """批量处理代码块的嵌入向量"""
        try:
            start_time = time.time()
            tasks = []
            total_chunks = len(chunks)
            chunks_processed = 0
            
            for i in range(0, total_chunks, self.batch_size):
                batch = chunks[i:i + self.batch_size]
                tasks.extend([self._get_embedding(chunk.content) for chunk in batch])
                # 等待当前批次完成
                embeddings = await asyncio.gather(*tasks)
                tasks.clear()
                
                # 更新代码块的嵌入向量
                for chunk, embedding in zip(batch, embeddings):
                    chunk.embedding = embedding
                    chunks_processed += 1
                    
                    if progress and task_id:
                        # 更新进度条
                        progress.update(task_id, completed=chunks_processed)
                        
                        # 计算并显示实时速率
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            rate = chunks_processed / elapsed
                            progress.update(
                                task_id,
                                description=f"嵌入处理中 [{rate:.2f} chunks/s]"
                            )
            
            # 更新总体指标
            total_time = time.time() - start_time
            self.metrics["total_chunks"] += total_chunks
            self.metrics["total_time"] = total_time
            self.metrics["avg_chunks_per_second"] = total_chunks / total_time if total_time > 0 else 0
            
            return chunks
        finally:
            # 确保会话被关闭
            await self._close_session()
            
    def get_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        return self.metrics.copy()

class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """使用Ollama生成嵌入向量"""
    
    def __init__(self, model: str = "codellama", batch_size: int = 10):
        self.model = model
        self.batch_size = batch_size
        self.base_url = "http://localhost:11434/api"
        
    async def _get_embedding(self, text: str) -> List[float]:
        """获取单个文本的嵌入向量"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
            return response.json()["embedding"]
            
    async def embed_chunks(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """为代码块批量生成嵌入向量"""
        tasks = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            tasks.extend([self._get_embedding(chunk.content) for chunk in batch])
            
        embeddings = await asyncio.gather(*tasks)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            
        return chunks 