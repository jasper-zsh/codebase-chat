from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

class Config:
    """配置类"""
    
    # 数据库配置
    DB_PATH = os.getenv("DB_PATH", "code_chunks.db")
    
    # 代码块处理配置
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "100"))
    OVERLAP = int(os.getenv("OVERLAP", "10"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    
    # 嵌入向量配置
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "codellama")
    
    # 重排序配置
    USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
    RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "siliconflow")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    
    # SiliconFlow 配置
    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
    SILICONFLOW_RERANKER_MODEL = os.getenv("SILICONFLOW_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    
    # FlagEmbedding 配置
    FLAG_EMBEDDING_SERVER_URL = os.getenv("FLAG_EMBEDDING_SERVER_URL", "http://localhost:8000")
    
    # 翻译配置
    USE_TRANSLATOR = os.getenv("USE_TRANSLATOR", "true").lower() == "true"
    TRANSLATOR_PROVIDER = os.getenv("TRANSLATOR_PROVIDER", "ollama")
    TRANSLATOR_MODEL = os.getenv("TRANSLATOR_MODEL", "qwen:14b")
    
    # Ollama 配置
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
    OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
    
    # 重排序服务配置
    RERANKER_HOST = os.getenv("RERANKER_HOST", "0.0.0.0")
    RERANKER_PORT = int(os.getenv("RERANKER_PORT", "9000")) 