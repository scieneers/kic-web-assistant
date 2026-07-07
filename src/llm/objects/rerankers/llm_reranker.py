import time
from typing import List, Optional

from src.api.models.serializable_text_node import SerializableTextNode
from src.llm.objects.LLMs import Models
from src.llm.objects.reranker import Reranker
from src.llm.objects.rerankers.base import BaseReranker, RerankResult

# Marginal input-token cost (EUR) per model. GWDG-hosted models (GEMMA4_31B and
# the LLAMA3 compat alias mapping to it) are flat-rate hosted — no per-token cost.
_EUR_PER_INPUT_TOKEN: dict[Models, float] = {
    Models.AZURE_FALLBACK: 2.30 / 1_000_000,  # Azure GPT-4o: ~$2.50/1M at ~0.92 EUR/USD
    Models.MINI: 0.14 / 1_000_000,  # Azure GPT-4o-mini: ~$0.15/1M
}
_CHARS_PER_TOKEN = 4


def _estimate_cost(query: str, nodes: List[SerializableTextNode], batch_size: int, model: Models) -> float:
    eur_per_token = _EUR_PER_INPUT_TOKEN.get(model, 0.0)
    if eur_per_token == 0.0:
        return 0.0
    total_chars = len(query) + sum(len(n.text or "") for n in nodes)
    n_batches = max(1, len(nodes) // batch_size)
    overhead_chars = n_batches * 200  # prompt template per batch
    return (total_chars + overhead_chars) / _CHARS_PER_TOKEN * eur_per_token


class LLMReranker(BaseReranker):
    """LLM-based reranker — current production default. Wraps LlamaIndex LLMRerank."""

    def __init__(self, top_n: int, min_score: float = 0.0):
        super().__init__(top_n, min_score)
        self._inner = Reranker(top_n=top_n)

    @property
    def name(self) -> str:
        return "LLM Reranker"

    def configure(self, top_n: int, min_score: float) -> None:
        super().configure(top_n, min_score)
        self._inner.top_n = top_n

    def normalize_score(self, score: float) -> float:
        # LlamaIndex LLMRerank scores relevance on a 1–10 scale.
        return score / 10.0

    def rerank(
        self,
        query: str,
        nodes: List[SerializableTextNode],
        model: Optional[Models] = None,
    ) -> RerankResult:
        if model is None:
            model = Models.AZURE_FALLBACK

        text_nodes = [n.to_text_node() for n in nodes]
        estimated_cost = _estimate_cost(query, nodes, self._inner.choice_batch_size, model)

        t0 = time.perf_counter()
        reranked = self._inner.rerank(query=query, nodes=text_nodes, model=model)
        latency_ms = (time.perf_counter() - t0) * 1000

        reranked, dropped = self.apply_min_score(reranked)

        return RerankResult(
            nodes=reranked,
            latency_ms=latency_ms,
            estimated_cost_eur=estimated_cost,
            metadata={"model": model.value, "input_chunks": len(nodes), "dropped_below_min_score": dropped},
        )
