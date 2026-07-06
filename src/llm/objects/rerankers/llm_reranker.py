import time
from typing import List, Optional

from src.api.models.serializable_text_node import SerializableTextNode
from src.llm.objects.LLMs import Models
from src.llm.objects.reranker import Reranker
from src.llm.objects.rerankers.base import BaseReranker, RerankResult

# Azure GPT-4o pricing (EUR). Input: ~$2.50/1M tokens at ~1.08 EUR/USD.
_EUR_PER_INPUT_TOKEN = 2.30 / 1_000_000
_CHARS_PER_TOKEN = 4


def _estimate_cost(query: str, nodes: List[SerializableTextNode], batch_size: int) -> float:
    total_chars = len(query) + sum(len(n.text or "") for n in nodes)
    n_batches = max(1, len(nodes) // batch_size)
    overhead_chars = n_batches * 200  # prompt template per batch
    return (total_chars + overhead_chars) / _CHARS_PER_TOKEN * _EUR_PER_INPUT_TOKEN


class LLMReranker(BaseReranker):
    """LLM-based reranker — current production default. Wraps LlamaIndex LLMRerank."""

    def __init__(self, top_n: int):
        super().__init__(top_n)
        self._inner = Reranker(top_n=top_n)

    @property
    def name(self) -> str:
        return "LLM Reranker"

    def rerank(
        self,
        query: str,
        nodes: List[SerializableTextNode],
        model: Optional[Models] = None,
    ) -> RerankResult:
        if model is None:
            model = Models.AZURE_FALLBACK

        text_nodes = [n.to_text_node() for n in nodes]
        estimated_cost = _estimate_cost(query, nodes, self._inner.choice_batch_size)

        t0 = time.perf_counter()
        reranked = self._inner.rerank(query=query, nodes=text_nodes, model=model)
        latency_ms = (time.perf_counter() - t0) * 1000

        return RerankResult(
            nodes=reranked,
            latency_ms=latency_ms,
            estimated_cost_eur=estimated_cost,
            metadata={"model": model.value, "input_chunks": len(nodes)},
        )
