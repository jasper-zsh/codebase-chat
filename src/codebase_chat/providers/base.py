from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from ..models.code_chunk import CodeChunk

class BaseEmbeddingProvider(ABC):
    """嵌入向量提供者的基类"""
    
    @abstractmethod
    async def embed_chunks(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """为代码块生成嵌入向量
        
        Args:
            chunks: 要处理的代码块列表
            
        Returns:
            处理后的代码块列表，每个代码块都包含嵌入向量
        """
        pass 

class BaseRerankProvider(ABC):
    """重排序提供者的基类"""
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        return_scores: bool = True
    ) -> List[Tuple[Dict[str, Any], float]]:
        """对候选结果进行重排序
        
        Args:
            query: 搜索查询
            candidates: 候选结果列表，每个结果是一个字典
            return_scores: 是否返回相似度分数
            
        Returns:
            按相关性排序的(结果, 分数)元组列表
        """
        pass