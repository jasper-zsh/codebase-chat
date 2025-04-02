from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, List
from ..models.code_chunk import CodeChunk, RepoInfo

class BaseChunkStrategy(ABC):
    """代码分块策略的基类"""
    
    @abstractmethod
    def chunk_file(self, file_path: Path, content: str, repo_info: RepoInfo) -> Generator[CodeChunk, None, None]:
        """将文件内容分割成代码块
        
        Args:
            file_path: 文件路径
            content: 文件内容
            
        Returns:
            代码块生成器
        """
        pass 