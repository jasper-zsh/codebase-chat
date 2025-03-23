from typing import List, Dict, Any, Union, Optional
import asyncio
import aiohttp
import time
import json
from urllib.parse import urljoin
from rich.progress import Progress, TaskID
import httpx

from ..models.code_chunk import CodeChunk
from .base import BaseEmbeddingProvider, BaseTranslatorProvider

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

class OllamaTranslatorProvider(BaseTranslatorProvider):
    """使用Ollama的大语言模型进行翻译"""
    
    LANGUAGE_MAP = {
        "zh": {"code": "zh", "name": "Chinese", "local_name": "中文"},
        "en": {"code": "en", "name": "English", "local_name": "English"},
        "ja": {"code": "ja", "name": "Japanese", "local_name": "日本語"},
        "ko": {"code": "ko", "name": "Korean", "local_name": "한국어"},
        "fr": {"code": "fr", "name": "French", "local_name": "Français"},
        "de": {"code": "de", "name": "German", "local_name": "Deutsch"},
        "es": {"code": "es", "name": "Spanish", "local_name": "Español"},
        "ru": {"code": "ru", "name": "Russian", "local_name": "Русский"},
    }
    
    def __init__(
        self,
        model: str = "qwen:14b",
        base_url: str = "http://localhost:11434",
        batch_size: int = 5,
        system_prompt: str = "你是一个专业的翻译助手。请准确翻译用户的文本，保持格式不变。对于代码注释和文档，保持专业性和准确性。"
    ):
        """
        Args:
            model: Ollama模型名称（推荐使用支持多语言的模型）
            base_url: Ollama服务的基础URL
            batch_size: 批处理大小
            system_prompt: 系统提示词
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.batch_size = batch_size
        self.system_prompt = system_prompt
        self._session = None
        
    async def _ensure_session(self):
        """确保aiohttp会话已创建"""
        if self._session is None:
            self._session = aiohttp.ClientSession()
            
    async def _close_session(self):
        """关闭aiohttp会话"""
        if self._session is not None:
            await self._session.close()
            self._session = None
            
    async def _translate_single(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: str = "zh",
        preserve_format: bool = True
    ) -> str:
        """翻译单个文本"""
        await self._ensure_session()
        
        # 构建提示词
        if source_lang:
            prompt = f"将以下{source_lang}文本翻译成{target_lang}："
        else:
            prompt = f"将以下文本翻译成{target_lang}："
            
        if preserve_format:
            prompt += "（请保持原文本的格式，包括缩进、换行等）\n\n"
        else:
            prompt += "\n\n"
            
        prompt += text
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        
        async with self._session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Ollama API error: {error_text}")
                
            data = await response.json()
            return data['message']['content'].strip()
            
    async def translate(
        self,
        texts: Union[str, List[str]],
        source_lang: Optional[str] = None,
        target_lang: str = "zh",
        preserve_format: bool = True
    ) -> Union[str, List[str]]:
        """翻译文本
        
        Args:
            texts: 要翻译的文本或文本列表
            source_lang: 源语言代码（如果为None则自动检测）
            target_lang: 目标语言代码
            preserve_format: 是否保留原文本格式（如代码缩进、换行等）
            
        Returns:
            翻译后的文本或文本列表
        """
        try:
            # 处理单个文本的情况
            if isinstance(texts, str):
                return await self._translate_single(
                    texts,
                    source_lang,
                    target_lang,
                    preserve_format
                )
            
            # 处理文本列表的情况
            results = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                tasks = [
                    self._translate_single(
                        text,
                        source_lang,
                        target_lang,
                        preserve_format
                    )
                    for text in batch
                ]
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
            
            return results
        finally:
            await self._close_session()
            
    async def detect_language(self, text: str) -> str:
        """检测文本语言
        
        使用Ollama模型进行语言检测
        
        Args:
            text: 要检测的文本
            
        Returns:
            语言代码
        """
        await self._ensure_session()
        try:
            prompt = "请检测以下文本的语言，只返回语言代码（如'en', 'zh', 'ja'等）：\n\n" + text
            
            messages = [
                {"role": "system", "content": "你是一个语言检测助手。请只返回语言代码，不要其他解释。"},
                {"role": "user", "content": prompt}
            ]
            
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False
            }
            
            async with self._session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Ollama API error: {error_text}")
                    
                data = await response.json()
                lang_code = data['message']['content'].strip().lower()
                
                # 确保返回支持的语言代码
                return lang_code if lang_code in self.LANGUAGE_MAP else "en"
        finally:
            await self._close_session()
            
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """获取支持的语言列表"""
        return list(self.LANGUAGE_MAP.values()) 