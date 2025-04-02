from pathlib import Path
from typing import Iterator, Optional

from .base import BaseChunkStrategy
from ..models.code_chunk import CodeChunk, RepoInfo

class LineChunkStrategy(BaseChunkStrategy):
    """基于行数的代码切片策略"""
    
    def __init__(self, chunk_size: int = 10, overlap: int = 2):
        """
        Args:
            chunk_size: 每个代码块的行数
            overlap: 相邻代码块的重叠行数
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_file(self, file_path: Path, content: str, repo_info: RepoInfo) -> Iterator[CodeChunk]:
        lines = content.splitlines()
        total_lines = len(lines)
        
        if total_lines == 0:
            return
            
        # 计算相对路径
        rel_path = str(file_path.relative_to(repo_info.repo_path))
            
        start_line = 0
        while start_line < total_lines:
            end_line = min(start_line + self.chunk_size, total_lines)
            chunk_content = "\n".join(lines[start_line:end_line])
            
            yield CodeChunk(
                repo_name=repo_info.repo_name,
                branch=[repo_info.branch],
                file_path=rel_path,
                start_line=start_line + 1,  # 转换为1-based行号
                end_line=end_line,
                content=chunk_content
            )
            
            # 考虑重叠行数移动到下一个块
            start_line = end_line - self.overlap if end_line < total_lines else total_lines 