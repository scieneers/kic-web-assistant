"""
Azure AI Search Semantic Reranker.

Differs from cross-encoder rerankers (BGE, Cohere): Azure's semantic ranker is
integrated into the search call, not a standalone scoring API. This reranker
re-issues the hybrid search with query_type="semantic" so Azure scores and
reorders results server-side using its own semantic model.

Prerequisites (already configured):
  - Standard-tier Azure Search service
  - Semantic configuration "default" added to the index (done)

Cost: ~€0.09 per 1000 semantic queries.
"""

import json
import logging
import time
from typing import List, Optional

from src.api.models.serializable_text_node import SerializableTextNode
from src.llm.objects.LLMs import LLM, Models
from src.llm.objects.rerankers.base import BaseReranker, RerankResult
from src.vectordb.azure_search import VectorDBAzureSearch
from src.env import env

logger = logging.getLogger(__name__)

_EUR_PER_QUERY = 0.09 / 1000
_SEMANTIC_CONFIG = "default"


class AzureSemanticReranker(BaseReranker):
    """
    Reranks by re-issuing the hybrid search with Azure's built-in semantic ranker.

    Results are ordered by @search.reranker_score (Azure's semantic score)
    instead of the default RRF score. The pre-retrieved nodes are used to
    extract the OData filter (course_id/module_id) so the re-issued search
    uses the same scope.
    """

    def __init__(self, top_n: int, min_score: float = 0.0):
        super().__init__(top_n)
        self._min_score = min_score
        self._llm = LLM()
        self._vector_db = VectorDBAzureSearch()

    @property
    def name(self) -> str:
        return "Azure Semantic"

    def rerank(
        self,
        query: str,
        nodes: List[SerializableTextNode],
        model: Optional[Models] = None,
    ) -> RerankResult:
        from src.llm.objects.retriever import _build_odata_filter

        # Extract scope from the first node's metadata (all nodes share the same filter)
        first_meta = nodes[0].metadata if nodes else {}
        course_id = first_meta.get("course_id")
        module_id = first_meta.get("module_id")
        odata_filter = _build_odata_filter(course_id, module_id)

        dense_embedding = self._llm.get_embedder().get_query_embedding(query)

        t0 = time.perf_counter()
        raw_results = self._vector_db.hybrid_search(
            query_text=query,
            query_vector=dense_embedding,
            index_name=env.AZURE_SEARCH_INDEX,
            odata_filter=odata_filter,
            top=self.top_n,
            use_semantic=True,
            semantic_config_name=_SEMANTIC_CONFIG,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        reranked = [self._to_node(r) for r in raw_results]

        # Sort defensively — Azure should already order by reranker_score, but be explicit
        reranked.sort(key=lambda n: n.score or 0, reverse=True)

        if self._min_score > 0:
            before = len(reranked)
            reranked = [n for n in reranked if (n.score or 0) >= self._min_score]
            dropped = before - len(reranked)
            if dropped:
                logger.debug(
                    "AzureSemanticReranker: dropped %d/%d chunks below min_score=%.2f",
                    dropped, before, self._min_score,
                )

        return RerankResult(
            nodes=reranked,
            latency_ms=latency_ms,
            estimated_cost_eur=_EUR_PER_QUERY,
            metadata={"semantic_config": _SEMANTIC_CONFIG, "input_chunks": len(nodes)},
        )

    @staticmethod
    def _to_node(result: dict) -> SerializableTextNode:
        raw_metadata = result.get("metadata_json")
        metadata = json.loads(raw_metadata) if raw_metadata else {}
        return SerializableTextNode(
            text=result.get("text", ""),
            id_=str(result.get("id")) if result.get("id") is not None else None,
            metadata=metadata,
            score=result.get("@search.reranker_score") or result.get("@search.score"),
        )
