"""
Cohere Rerank API.

Model: rerank-multilingual-v3.0
  - Best-in-class multilingual reranking, excellent German support
  - External API call (~200ms p50)
  - Cost: ~$1 / 1000 queries (~€0.92)

Prerequisites:
  pip install cohere

  Add to pyproject.toml:
    "cohere>=5.0.0,<6"

  Set environment variable:
    COHERE_API_KEY=<your key>
"""

import time
from typing import List, Optional

from src.api.models.serializable_text_node import SerializableTextNode
from src.llm.objects.LLMs import Models
from src.llm.objects.rerankers.base import BaseReranker, RerankResult

_DEFAULT_MODEL = "rerank-multilingual-v3.0"
_EUR_PER_QUERY = 0.92 / 1000


class CohereReranker(BaseReranker):
    """
    Reranker using Cohere's dedicated reranking API.

    TODO: Implement once cohere package is added as a dependency.

    Steps to activate:
    1. Add "cohere>=5.0.0,<6" to pyproject.toml dependencies.
    2. Add COHERE_API_KEY to .env and Azure Key Vault.
    3. Add COHERE_API_KEY field to EnvHelper in env.py.
    4. Remove the NotImplementedError below and implement rerank().

    Example implementation:
        import cohere
        co = cohere.Client(env.COHERE_API_KEY)
        response = co.rerank(
            model=self.model_name,
            query=query,
            documents=[n.text for n in nodes],
            top_n=self.top_n,
        )
        ordered = [nodes[r.index] for r in response.results]
        return ordered
    """

    def __init__(self, top_n: int, model_name: str = _DEFAULT_MODEL):
        super().__init__(top_n)
        self.model_name = model_name
        raise NotImplementedError(
            "CohereReranker requires the cohere package and COHERE_API_KEY. "
            "See module docstring for setup steps."
        )

    @property
    def name(self) -> str:
        return f"Cohere ({self.model_name})"

    def rerank(
        self,
        query: str,
        nodes: List[SerializableTextNode],
        model: Optional[Models] = None,
    ) -> RerankResult:
        raise NotImplementedError
