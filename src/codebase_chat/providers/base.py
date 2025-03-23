from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Union
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

class BaseTranslatorProvider(ABC):
    """翻译提供者的基类"""
    
    @abstractmethod
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
            翻译后的文本或文本列表。如果输入是字符串，返回字符串；
            如果输入是列表，返回列表
        """
        pass
    
    @abstractmethod
    async def detect_language(self, text: str) -> str:
        """检测文本语言
        
        Args:
            text: 要检测的文本
            
        Returns:
            语言代码（如'en', 'zh', 'ja'等）
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """获取支持的语言列表
        
        Returns:
            支持的语言列表，每个语言是一个字典，包含：
            - code: 语言代码
            - name: 语言名称（英文）
            - local_name: 语言本地名称
        """
        pass