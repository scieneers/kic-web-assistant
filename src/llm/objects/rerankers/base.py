from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from src.api.models.serializable_text_node import SerializableTextNode
from src.llm.objects.LLMs import Models


@dataclass
class RerankResult:
    nodes: List[SerializableTextNode]
    latency_ms: float
    estimated_cost_eur: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseReranker(ABC):
    """Common interface for all reranker implementations.

    top_n is set at construction (from system_config); model is passed per call
    because it comes from runtime_config and can vary per request.
    Non-LLM rerankers ignore the model argument.

    min_score is a relevance cutoff on a NORMALIZED 0–1 scale (backend-native
    scores are mapped via normalize_score). Nodes below the cutoff are dropped;
    if ALL nodes fall below it, the rerank node returns an empty list, which
    triggers the no-answer fallback in the answer node — "all hits are weak"
    then behaves like "no hits" instead of producing a weak answer.
    """

    def __init__(self, top_n: int, min_score: float = 0.0):
        self.top_n = top_n
        self.min_score = min_score

    def configure(self, top_n: int, min_score: float) -> None:
        """Update runtime parameters on a cached (singleton) instance."""
        self.top_n = top_n
        self.min_score = min_score

    def normalize_score(self, score: float) -> float:
        """Map a backend-native score to the shared 0–1 relevance scale.

        Default is identity (for backends that already score in 0–1).
        """
        return score

    def apply_min_score(self, nodes: List[SerializableTextNode]) -> tuple[List[SerializableTextNode], int]:
        """Drop nodes whose normalized score falls below min_score.

        Nodes without a score are dropped too (cannot be assessed).
        Returns (kept_nodes, dropped_count). No-op when min_score <= 0.
        """
        if self.min_score <= 0:
            return nodes, 0
        kept = [
            n for n in nodes
            if n.score is not None and self.normalize_score(n.score) >= self.min_score
        ]
        return kept, len(nodes) - len(kept)

    @abstractmethod
    def rerank(
        self,
        query: str,
        nodes: List[SerializableTextNode],
        model: Optional[Models] = None,
    ) -> RerankResult:
        """Rerank nodes by relevance to query. Returns RerankResult with timing and cost."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
