"""
BGE Cross-Encoder Reranker (local, zero per-request cost).

Model: BAAI/bge-reranker-v2-m3 (~568MB)
  - Multilingual (100+ languages including German and English)
  - 8192 token context window
  - Runs on CPU, ~30-80ms for 10 chunks

Prerequisites:
  sentence-transformers>=3.0.0,<4  (already in pyproject.toml)
  uv sync
"""

import time
from typing import List, Optional

from src.api.models.serializable_text_node import SerializableTextNode
from src.llm.objects.LLMs import Models
from src.llm.objects.rerankers.base import BaseReranker, RerankResult

BGE_LARGE_MODEL = "BAAI/bge-reranker-v2-m3"   # 568MB, multilingual (100+ languages)
BGE_SMALL_MODEL = "BAAI/bge-reranker-base"    # 280MB, faster, primarily English
_DEFAULT_MODEL = BGE_LARGE_MODEL


class BGEReranker(BaseReranker):
    """Local cross-encoder reranker using sentence-transformers. Model is downloaded
    on first use (~280MB) and cached by HuggingFace locally."""

    def __init__(self, top_n: int, model_name: str = _DEFAULT_MODEL):
        super().__init__(top_n)
        self.model_name = model_name
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_name)

    @property
    def name(self) -> str:
        return f"BGE ({self.model_name.split('/')[-1]})"

    def rerank(
        self,
        query: str,
        nodes: List[SerializableTextNode],
        model: Optional[Models] = None,
    ) -> RerankResult:
        pairs = [(query, node.text) for node in nodes]

        t0 = time.perf_counter()
        scores = self._model.predict(pairs)
        latency_ms = (time.perf_counter() - t0) * 1000

        ranked = sorted(zip(nodes, scores), key=lambda x: x[1], reverse=True)
        top_nodes = [node for node, _ in ranked[: self.top_n]]

        return RerankResult(
            nodes=top_nodes,
            latency_ms=latency_ms,
            estimated_cost_eur=0.0,
            metadata={"model": self.model_name, "input_chunks": len(nodes)},
        )
