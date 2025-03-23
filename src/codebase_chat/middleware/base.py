from abc import ABC, abstractmethod
from typing import List

from ..models.code_chunk import CodeChunk

class BaseMiddleware(ABC):
    """中间件基类，用于在代码块处理过程中添加自定义逻辑"""
    
    @abstractmethod
    async def process(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """处理代码块
        
        Args:
            chunks: 待处理的代码块列表
            
        Returns:
            List[CodeChunk]: 处理后的代码块列表
        """
        pass 