import asyncio
from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

from codebase_chat.core.processor import CodeProcessor, ProcessingStats
from codebase_chat.strategies.line_chunker import LineChunkStrategy
from codebase_chat.providers.ollama import OllamaEmbeddingProvider
from codebase_chat.providers.flag_embedding import FlagEmbeddingRerankProvider, start_server

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
    stats_table.add_row("文件处理速率 (files/s)", f"{stats.files_per_second:.2f}")
    stats_table.add_row("代码块处理速率 (chunks/s)", f"{stats.chunks_per_second:.2f}")
    stats_table.add_row("代码行处理速率 (lines/s)", f"{stats.lines_per_second:.2f}")
    
    return stats_table

def get_processor(
    db_path: str,
    chunk_size: int = 10,
    overlap: int = 2,
    model: str = "codellama",
    batch_size: int = 10,
    use_reranker: bool = True,
    reranker_url: str = "http://localhost:8000",
    progress_callback = None
) -> CodeProcessor:
    """创建代码处理器实例"""
    chunk_strategy = LineChunkStrategy(chunk_size=chunk_size, overlap=overlap)
    embedding_provider = OllamaEmbeddingProvider(model=model, batch_size=batch_size)
    rerank_provider = FlagEmbeddingRerankProvider(base_url=reranker_url) if use_reranker else None
    
    return CodeProcessor(
        db_path,
        chunk_strategy,
        embedding_provider,
        rerank_provider=rerank_provider,
        progress_callback=progress_callback
    )

@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="服务器主机地址"),
    port: int = typer.Option(8000, help="服务器端口"),
    model: str = typer.Option("BAAI/bge-reranker-large", help="重排序模型名称"),
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
    db_path: str = typer.Option("./codebase.db", help="数据库路径"),
    chunk_size: int = typer.Option(100, help="代码块大小（行数）"),
    overlap: int = typer.Option(10, help="相邻代码块重叠行数"),
    model: str = typer.Option("codellama", help="Ollama模型名称"),
    batch_size: int = typer.Option(10, help="批处理大小"),
):
    """索引代码文件到向量数据库"""
    # 收集所有文件路径
    files = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(path.rglob("*"))
            
    # 过滤掉不需要的文件
    files = [
        f for f in files 
        if f.is_file() and not any(part.startswith('.') and part not in ['.', '..'] for part in f.parts)
    ]
    
    console.print(f"Found {len(files)} files to process")
    
    # 创建实时更新界面
    with Live(auto_refresh=False) as live:
        def update_progress(stats: ProcessingStats):
            live.update(create_stats_table(stats))
            live.refresh()
            
        processor = get_processor(
            db_path, chunk_size, overlap, model, batch_size,
            progress_callback=update_progress
        )
        
        final_stats = asyncio.run(processor.index_files(files))
        
    # 显示最终统计信息
    console.print("\n[bold green]处理完成！最终统计：[/bold green]")
    console.print(create_stats_table(final_stats))

@app.command()
def search(
    query: str = typer.Argument(..., help="搜索查询"),
    db_path: str = typer.Option("./codebase.db", help="数据库路径"),
    limit: int = typer.Option(5, help="返回结果数量"),
    model: str = typer.Option("codellama", help="Ollama模型名称"),
    use_reranker: bool = typer.Option(True, help="是否使用重排序模型"),
    reranker_url: str = typer.Option("http://localhost:9000", help="重排序服务URL"),
):
    """搜索相似代码块"""
    processor = get_processor(
        db_path,
        model=model,
        use_reranker=use_reranker,
        reranker_url=reranker_url
    )
    
    results = asyncio.run(processor.search(query, limit))
    
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
            chunk.branch or "N/A",
            chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
        )
    
    console.print(table)

if __name__ == "__main__":
    app() 