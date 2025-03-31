from typing import List, Tuple, Dict, Any
import httpx
from .base import BaseRerankProvider

class SiliconflowRerankProvider(BaseRerankProvider):
    """使用 SiliconFlow API 进行重排序的提供者"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "BAAI/bge-reranker-v2-m3",
        base_url: str = "https://api.siliconflow.cn/v1",
    ):
        """
        Args:
            api_key: SiliconFlow API 密钥
            model: 重排序模型名称
            base_url: API 基础 URL
            batch_size: 批处理大小
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip('/')
        
    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        return_scores: bool = True
    ) -> List[Tuple[Dict[str, Any], float]]:
        """对候选结果进行重排序
        
        通过 SiliconFlow API 调用重排序服务
        
        Args:
            query: 搜索查询
            candidates: 候选结果列表
            return_scores: 是否返回相似度分数
            
        Returns:
            按相关性排序的(结果, 分数)元组列表
        """
        # 准备文本
        texts = [candidate["content"] for candidate in candidates]
        
        # 一次性处理所有文本
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": texts,
                    "top_n": len(texts),
                    "return_documents": False
                }
            )
            response.raise_for_status()
            results = response.json()["results"]
            scores = [result["relevance_score"] for result in results]

        # 将分数与候选结果配对并排序
        scored_results = list(zip(candidates, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return scored_results if return_scores else [r[0] for r in scored_results]