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
    """

    def __init__(self, top_n: int):
        self.top_n = top_n

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
