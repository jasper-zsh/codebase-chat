# Codebase Chat

一个用于代码仓库分析和向量化的工具，支持语义搜索和RAG应用。

## 功能特点

- 支持代码仓库的增量索引
- 使用LanceDB进行向量存储
- 支持Ollama API进行文本嵌入
- 可扩展的中间件系统
- 灵活的代码切片策略
- 支持Git仓库信息集成

## 安装

1. 确保已安装Python 3.9+
2. 安装uv包管理器
3. 克隆仓库并安装依赖：

```bash
git clone <repository-url>
cd codebase-chat
uv venv
uv pip install -e .
```

## 使用方法

### 索引代码文件

```bash
python -m src.main index /path/to/your/code --db-path ./codebase.db
```

可选参数：
- `--chunk-size`: 代码块大小（行数），默认为10
- `--overlap`: 相邻代码块重叠行数，默认为2
- `--model`: Ollama模型名称，默认为"codellama"
- `--batch-size`: 批处理大小，默认为10

### 搜索代码

```bash
python -m src.main search "你的搜索查询" --db-path ./codebase.db
```

可选参数：
- `--limit`: 返回结果数量，默认为5
- `--model`: Ollama模型名称，默认为"codellama"

## 扩展

### 自定义切片策略

继承 `BaseChunkStrategy` 类并实现 `chunk_file` 方法：

```python
from src.strategies.base import BaseChunkStrategy

class MyChunkStrategy(BaseChunkStrategy):
    def chunk_file(self, file_path: Path, content: str) -> Iterator[CodeChunk]:
        # 实现你的切片逻辑
        pass
```

### 自定义中间件

继承 `BaseMiddleware` 类并实现 `process` 方法：

```python
from src.middleware.base import BaseMiddleware

class MyMiddleware(BaseMiddleware):
    async def process(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        # 实现你的处理逻辑
        return chunks
```

## 许可证

MIT 