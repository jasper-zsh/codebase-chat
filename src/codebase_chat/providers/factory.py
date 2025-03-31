from typing import Optional, Dict, Any
from .base import BaseRerankProvider, BaseEmbeddingProvider, BaseTranslatorProvider
from .siliconflow import SiliconflowRerankProvider
from .flag_embedding import FlagEmbeddingRerankProvider
from .ollama import OllamaEmbeddingProvider, OllamaTranslatorProvider
from ..config import Config

class ProviderFactory:
    """Provider 工厂类"""
    
    @classmethod
    def create_rerank_provider(cls) -> Optional[BaseRerankProvider]:
        """创建重排序 Provider
        
        根据环境变量 USE_RERANKER 和 RERANKER_PROVIDER 创建相应的重排序提供者
        """
        if not Config.USE_RERANKER:
            return None
            
        provider_name = Config.RERANKER_PROVIDER.lower()
        
        if provider_name == "siliconflow":
            if not Config.SILICONFLOW_API_KEY:
                raise ValueError("SILICONFLOW_API_KEY 环境变量未设置")
            return SiliconflowRerankProvider(
                api_key=Config.SILICONFLOW_API_KEY,
                model=Config.SILICONFLOW_RERANKER_MODEL,
            )
        elif provider_name == "flag_embedding":
            return FlagEmbeddingRerankProvider(
                base_url=Config.FLAG_EMBEDDING_SERVER_URL,
            )
        else:
            raise ValueError(f"不支持的 Rerank Provider: {provider_name}")
            
    @classmethod
    def create_embedding_provider(cls) -> BaseEmbeddingProvider:
        """创建嵌入向量 Provider
        
        根据环境变量 EMBEDDING_PROVIDER 创建相应的嵌入向量提供者
        """
        provider_name = Config.EMBEDDING_PROVIDER.lower()
        
        if provider_name == "ollama":
            return OllamaEmbeddingProvider(
                model=Config.EMBEDDING_MODEL,
                base_url=f"http://{Config.OLLAMA_HOST}:{Config.OLLAMA_PORT}",
                batch_size=Config.BATCH_SIZE
            )
        else:
            raise ValueError(f"不支持的 Embedding Provider: {provider_name}")
            
    @classmethod
    def create_translator_provider(cls) -> Optional[BaseTranslatorProvider]:
        """创建翻译 Provider
        
        根据环境变量 USE_TRANSLATOR 和 TRANSLATOR_PROVIDER 创建相应的翻译提供者
        """
        if not Config.USE_TRANSLATOR:
            return None
            
        provider_name = Config.TRANSLATOR_PROVIDER.lower()
        
        if provider_name == "ollama":
            return OllamaTranslatorProvider(
                model=Config.TRANSLATOR_MODEL,
                base_url=f"http://{Config.OLLAMA_HOST}:{Config.OLLAMA_PORT}",
                batch_size=Config.BATCH_SIZE
            )
        else:
            raise ValueError(f"不支持的 Translator Provider: {provider_name}")
