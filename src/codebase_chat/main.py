import asyncio
from pathlib import Path
from typing import List
import os
import typer
from rich.console import Console
from rich.table import Table
from rich.live import Live
from dotenv import load_dotenv
import gitignore_parser

from codebase_chat.core.processor import CodeProcessor, ProcessingStats
from codebase_chat.strategies.line_chunker import LineChunkStrategy
from codebase_chat.providers.flag_embedding import start_server
from codebase_chat.models.code_chunk import CodeChunk
from codebase_chat.providers.factory import ProviderFactory

# 加载.env文件
load_dotenv()

# 从环境变量获取配置，如果不存在则使用默认值
DEFAULT_DB_PATH = os.getenv("DB_PATH", "codebase.db")
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "100"))
DEFAULT_OVERLAP = int(os.getenv("OVERLAP", "10"))
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))

app = typer.Typer()
console = Console()

def create_stats_table(stats: ProcessingStats) -> Table:
    """创建性能统计表格"""
    stats_table = Table(show_header=True, header_style="bold blue")
    stats_table.add_column("指标")
    stats_table.add_column("值")
    
    stats_table.add_row("处理进度", f"{stats.progress_percentage:.1f}%")
    stats_table.add_row("处理的文件数", f"{stats.total_files}/{stats.expected_total_files}")
    stats_table.add_row("处理的代码块总数", str(stats.total_chunks))
    stats_table.add_row("处理的代码行总数", str(stats.total_lines))
    stats_table.add_row("已用时间 (秒)", f"{stats.processing_time:.2f}")
    stats_table.add_row("嵌入向量时间 (秒)", f"{stats.embedding_time:.2f}")
    stats_table.add_row("Milvus 连接时间 (秒)", f"{stats.milvus_connect_time:.2f}")
    stats_table.add_row("Milvus 写入时间 (秒)", f"{stats.milvus_write_time:.2f}")
    stats_table.add_row("Milvus 读取时间 (秒)", f"{stats.milvus_read_time:.2f}")
    stats_table.add_row("Milvus 刷新时间 (秒)", f"{stats.milvus_flush_time:.2f}")
    stats_table.add_row("文件处理速率 (files/s)", f"{stats.files_per_second:.2f}")
    stats_table.add_row("代码块处理速率 (chunks/s)", f"{stats.chunks_per_second:.2f}")
    stats_table.add_row("代码行处理速率 (lines/s)", f"{stats.lines_per_second:.2f}")
    
    return stats_table

def get_processor(
    db_path: str = DEFAULT_DB_PATH,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    progress_callback = None
) -> CodeProcessor:
    """创建代码处理器实例"""
    chunk_strategy = LineChunkStrategy(chunk_size=chunk_size, overlap=overlap)
    # chunk_strategy = GolangChunkStrategy()
    embedding_provider = ProviderFactory.create_embedding_provider()
    rerank_provider = ProviderFactory.create_rerank_provider()
    translator_provider = ProviderFactory.create_translator_provider()
    
    return CodeProcessor(
        db_path,
        chunk_strategy,
        embedding_provider,
        rerank_provider=rerank_provider,
        translator_provider=translator_provider,
        progress_callback=progress_callback
    )

@app.command()
def serve(
    host: str = typer.Option(os.getenv("FLAG_EMBEDDING_SERVER_HOST", "0.0.0.0"), help="服务器主机地址"),
    port: int = typer.Option(os.getenv("FLAG_EMBEDDING_SERVER_PORT", "9000"), help="服务器端口"),
    model: str = typer.Option(os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"), help="重排序模型名称"),
    use_fp16: bool = typer.Option(True, help="是否使用半精度浮点数"),
):
    """启动重排序服务器"""
    console.print(f"[bold green]启动重排序服务器 {host}:{port}[/bold green]")
    console.print(f"模型: {model}")
    console.print(f"使用FP16: {use_fp16}")
    start_server(host, port, model, use_fp16)

@app.command()
def index(
    paths: List[Path] = typer.Argument(..., help="要索引的文件或目录路径"),
    db_path: str = typer.Option(DEFAULT_DB_PATH, help="数据库路径"),
    chunk_size: int = typer.Option(DEFAULT_CHUNK_SIZE, help="代码块大小（行数）"),
    overlap: int = typer.Option(DEFAULT_OVERLAP, help="相邻代码块重叠行数"),
):
    """索引代码文件到向量数据库"""
    # 收集所有目录路径
    directories = [path for path in paths if path.is_dir()]
    
    # 确保至少有一个目录
    if not directories:
        console.print("请提供至少一个有效的目录路径。")
        return
    
    for repo_path in directories:
        files = []
        
        # 检查是否存在 .gitignore 文件
        gitignore_path = repo_path / '.gitignore'
        gitignore = None
        if gitignore_path.exists():
            # 解析 .gitignore 文件
            gitignore = gitignore_parser.parse_gitignore(str(gitignore_path), repo_path)
        
        for path in repo_path.rglob("*"):
            if path.is_file():
                # 根据 .gitignore 规则过滤文件
                if gitignore and gitignore(path):
                    print(f'Ignore {path}')
                    continue
                if '.git' in path.parts:
                    continue
                # 跳过二进制文件
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        f.read(16)  # 尝试读取文件开头部分
                except UnicodeDecodeError:
                    print(f'跳过二进制文件: {path}')
                    continue
                files.append(path)
        
        console.print(f"Found {len(files)} files to process in directory: {repo_path}")
        
        # 创建实时更新界面
        with Live(auto_refresh=False) as live:
            def update_progress(stats: ProcessingStats):
                live.update(create_stats_table(stats))
                live.refresh()
                
            processor = get_processor(
                db_path, chunk_size, overlap,
                progress_callback=update_progress
            )
            
            # 只传递当前目录作为 repo_name
            final_stats = asyncio.run(processor.index_files(files, repo_path))
        
    # 显示最终统计信息
    console.print("\n[bold green]处理完成！最终统计：[/bold green]")
    console.print(create_stats_table(final_stats))

@app.command()
def search(
    query: str = typer.Argument(..., help="搜索查询"),
    db_path: str = typer.Option(DEFAULT_DB_PATH, help="数据库路径"),
    limit: int = typer.Option(5, help="返回结果数量"),
):
    """搜索相似代码块"""
    processor = get_processor(
        db_path,
    )
    
    results, stats = asyncio.run(processor.search(query, limit))
    
    # 显示结果
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("相关度")
    table.add_column("文件")
    table.add_column("行号")
    table.add_column("分支")
    table.add_column("内容")
    
    for chunk, score in results:
        table.add_row(
            f"{score:.4f}",
            str(chunk.file_path),
            f"{chunk.start_line}-{chunk.end_line}",
            ','.join(chunk.branch) or "N/A",
            chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
        )
    
    console.print(table)

@app.command()
def mcp_server(
    transport: str = typer.Argument("sse", help="传输协议"),
    host: str = typer.Option(os.getenv("MCP_HOST", "0.0.0.0"), help="服务器主机地址"),
    port: int = typer.Option(os.getenv("MCP_PORT", "8000"), help="服务器端口"),
):
    """启动代码仓检索服务器"""
    from mcp.server import FastMCP
    from mcp.server.fastmcp.server import Field, Context

    server = FastMCP('CodebaseChat', host=host, port=port)
    processor = get_processor(
        DEFAULT_DB_PATH,
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_OVERLAP,
        progress_callback=None
    )

    @server.tool()
    async def search(query: str = Field(description="用户输入的原始问题，接受自然语言，不需要提取关键词"), context: Context = None) -> str:
        """搜索代码仓"""
        await context.info(f"搜索代码仓: {query}")
        results, stats = await processor.search(query, limit=10, context=context)
        # 将搜索结果转换为字典列表
        formatted_result = ''
        for chunk, score in results:
            formatted_result += f'''# {chunk.file_path} {chunk.start_line}-{chunk.end_line}

{chunk.content}

'''
        return formatted_result

    server.run(transport=transport)

if __name__ == "__main__":
    app() 