# Codebase Chat

一个强大的代码仓库语义搜索和分析工具，支持多语言检索和RAG应用。

## 功能特点

- 🔍 **智能搜索**
  - 基于语义的代码检索
  - 多语言查询支持（自动翻译）
  - 结果智能重排序
  - 支持Git仓库信息集成

- 💾 **高效存储**
  - 使用LanceDB进行向量存储
  - 支持增量索引
  - 灵活的代码切片策略
  - 可配置的块大小和重叠度

- 🤖 **AI驱动**
  - 集成Ollama API进行文本嵌入
  - 支持多种大语言模型
  - 智能翻译服务
  - 高性能重排序服务

- 🔧 **可扩展性**
  - 插件化的中间件系统
  - 自定义切片策略
  - 环境变量配置
  - HTTP API服务

## 快速开始

### 安装

1. 确保已安装`uv`
2. 安装依赖：

```bash
git clone <repository-url>
cd codebase-chat
uv sync
```

### 配置

1. 复制示例配置文件：
```bash
cp .env.example .env
```

2. 根据需要修改配置项：
```env
# 数据库配置
CODEBASE_DB_PATH=./codebase.db

# 模型配置
CODEBASE_EMBEDDING_MODEL=codellama
CODEBASE_TRANSLATOR_MODEL=qwen:14b
CODEBASE_RERANKER_MODEL=BAAI/bge-reranker-large

# 服务配置
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
```

## 使用方法

### 前置准备

#### 启动重排序服务：
```bash
python -m src.main serve
```

### 命令行工具

#### 索引代码仓库：
```bash
python -m src.main index --chunk-size 100 --overlap 10 /path/to/your/code 
```

#### 搜索代码：
```bash
python -m src.main search "查找文件处理相关的函数"
```

### MCP Server

#### 启动MCP服务器
```bash
python -m src.main mcp-server
```

这个操作将以SSE模式在8000端口启动一个MCP Server，启动后将其配置到MCP Client就可以使用了。

## 高级功能

### 自定义切片策略

```python
from src.strategies.base import BaseChunkStrategy

class MyChunkStrategy(BaseChunkStrategy):
    def chunk_file(self, file_path: Path, content: str) -> Iterator[CodeChunk]:
        # 实现自定义切片逻辑
        pass
```

### 自定义中间件

```python
from src.middleware.base import BaseMiddleware

class MyMiddleware(BaseMiddleware):
    async def process(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        # 实现自定义处理逻辑
        return chunks
```

## 环境要求

- Python 3.9+
- Ollama服务
- 足够的磁盘空间用于向量数据库
- 推荐使用GPU以获得更好的性能

## 许可证

MIT

## 贡献

欢迎提交Issue和Pull Request！

1. Fork本仓库
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request 