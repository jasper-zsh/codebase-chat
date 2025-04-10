from typing import Optional, Dict, Any
from pydantic import BaseModel
from pathlib import Path

class RepoInfo(BaseModel):
    """表示一个Git仓库的信息"""
    repo_path: Path
    repo_name: str
    branch: Optional[str] = None

class CodeChunk(BaseModel):
    """表示一个代码块的数据模型"""
    repo_name: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    branch: list[str] = []
    embedding: Optional[list[float]] = None
    metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，用于存储到向量数据库"""
        return {
            "repo_name": self.repo_name,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
            "branch": self.branch,
            "embedding": self.embedding,
            **self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeChunk":
        """从字典创建CodeChunk实例"""
        metadata = {k: v for k, v in data.items() 
                   if k not in ["file_path", "start_line", "end_line", "content", "branch", "embedding", "repo_name"]}
        return cls(
            repo_name=data.get("repo_name"),
            file_path=data["file_path"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            content=data["content"],
            branch=data.get("branch"),
            embedding=data.get("embedding"),
            metadata=metadata
        ) 