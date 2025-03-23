from typing import List, Tuple, Dict, Any
import asyncio
import httpx
from codebase_chat.providers.base import BaseRerankProvider

class FlagEmbeddingRerankProvider(BaseRerankProvider):
    """使用FlagEmbedding进行重排序的提供者"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:9000",
        batch_size: int = 32
    ):
        """
        Args:
            base_url: 重排序服务的基础URL
            batch_size: 批处理大小
        """
        self.base_url = base_url
        self.batch_size = batch_size
        
    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        return_scores: bool = True
    ) -> List[Tuple[Dict[str, Any], float]]:
        """对候选结果进行重排序
        
        通过HTTP API调用FlagEmbedding服务进行重排序
        
        Args:
            query: 搜索查询
            candidates: 候选结果列表
            return_scores: 是否返回相似度分数
            
        Returns:
            按相关性排序的(结果, 分数)元组列表
        """
        # 准备文本
        texts = [candidate["content"] for candidate in candidates]
        
        # 分批处理
        all_scores = []
        async with httpx.AsyncClient(timeout=60) as client:
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                response = await client.post(
                    f"{self.base_url}/rerank",
                    json={
                        "query": query,
                        "texts": batch_texts,
                        "return_scores": return_scores
                    }
                )
                response.raise_for_status()
                all_scores.extend(response.json()["scores"])
        
        # 将分数与候选结果配对并排序
        scored_results = list(zip(candidates, all_scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return scored_results if return_scores else [r[0] for r in scored_results]

# 全局模型实例
_model = None

def start_server(
    host: str = "0.0.0.0",
    port: int = 9000,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    use_fp16: bool = True
):
    import uvicorn
    from fastapi import FastAPI, BackgroundTasks
    from FlagEmbedding import FlagReranker
    from pydantic import BaseModel
    # FastAPI应用
    app = FastAPI(title="FlagEmbedding Rerank Service")


    def get_model(model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True) -> FlagReranker:
        """获取或初始化模型实例（单例模式）"""
        global _model
        if _model is None:
            _model = FlagReranker(model_name, use_fp16=use_fp16)
        return _model

    class RerankRequest(BaseModel):
        """重排序请求模型"""
        query: str
        texts: List[str]
        return_scores: bool = True

    class RerankResponse(BaseModel):
        """重排序响应模型"""
        scores: List[float]

    
    @app.post("/rerank", response_model=RerankResponse)
    async def rerank(request: RerankRequest, background_tasks: BackgroundTasks) -> RerankResponse:
        """重排序端点"""
        model = get_model()
        # 在后台任务中运行计算密集型操作
        pairs = [(request.query, text) for text in request.texts]
        scores = await asyncio.get_event_loop().run_in_executor(
            None, model.compute_score, pairs
        )
        return RerankResponse(scores=scores)
    """启动FastAPI服务器"""
    # 预加载模型
    get_model(model_name, use_fp16)
    # 启动服务器
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
