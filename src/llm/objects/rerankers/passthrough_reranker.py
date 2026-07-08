import time
from typing import List, Optional

from src.api.models.serializable_text_node import SerializableTextNode
from src.llm.objects.LLMs import Models
from src.llm.objects.rerankers.base import BaseReranker, RerankResult


class PassthroughReranker(BaseReranker):
    """No-op baseline: returns the top-N chunks in original retrieval order.

    Used in benchmarks to measure the uplift from actual reranking.
    """

    @property
    def name(self) -> str:
        return "no_rerank (baseline)"

    def rerank(
        self,
        query: str,
        nodes: List[SerializableTextNode],
        model: Optional[Models] = None,
        *,
        course_id: Optional[int] = None,
        module_id: Optional[int] = None,
        preranked: bool = False,
    ) -> RerankResult:
        t0 = time.perf_counter()
        result = nodes[: self.top_n]
        latency_ms = (time.perf_counter() - t0) * 1000
        return RerankResult(nodes=result, latency_ms=latency_ms, estimated_cost_eur=0.0)
