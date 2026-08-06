"""
Azure AI Search Semantic Reranker.

Differs from cross-encoder rerankers (BGE, Cohere): Azure's semantic ranker is
integrated into the search call, not a standalone scoring API. Two modes:

1. Integrated (production, preranked=True): retrieval itself runs with
   query_type="semantic" (see KiCampusRetriever), so the nodes already carry
   @search.reranker_score. This backend then only cuts to top_n and applies
   min_score — no second search call, no second embedding.

2. Re-search (benchmark, preranked=False): re-issues the hybrid search with
   query_type="semantic" so Azure scores server-side. The REQUEST scope
   (course_id/module_id) must be passed explicitly — it is never derived from
   chunk metadata, because chunks carry the course they BELONG TO or DESCRIBE
   (e.g. Drupal course pages have a course_id), not the scope of the request.

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
    Reranks via Azure AI Search's built-in semantic ranker.

    Results are ordered by @search.reranker_score (0–4 scale) instead of the
    default RRF score. See module docstring for the two operating modes.
    """

    def __init__(self, top_n: int, min_score: float = 0.0):
        super().__init__(top_n, min_score)
        self._llm = LLM()
        self._vector_db = VectorDBAzureSearch()

    @property
    def name(self) -> str:
        return "Azure Semantic"

    def normalize_score(self, score: float) -> float:
        # Azure @search.reranker_score is on a 0–4 scale.
        return score / 4.0

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
        if preranked:
            return self._cut_preranked(nodes)
        return self._research(query, course_id=course_id, module_id=module_id)

    def _cut_preranked(self, nodes: List[SerializableTextNode]) -> RerankResult:
        """Integrated mode: retrieval already ranked with the semantic ranker —
        only sort defensively, cut to top_n and apply min_score."""
        t0 = time.perf_counter()
        ranked = sorted(nodes, key=lambda n: n.score or 0, reverse=True)[: self.top_n]
        ranked, dropped = self.apply_min_score(ranked)
        latency_ms = (time.perf_counter() - t0) * 1000
        return RerankResult(
            nodes=ranked,
            latency_ms=latency_ms,
            # The semantic surcharge was paid on the retrieval call — attribute
            # it here so cost telemetry per rerank stays comparable.
            estimated_cost_eur=_EUR_PER_QUERY,
            metadata={
                "semantic_config": _SEMANTIC_CONFIG,
                "integrated": True,
                "input_chunks": len(nodes),
                "dropped_below_min_score": dropped,
            },
        )

    def _research(
        self,
        query: str,
        course_id: Optional[int] = None,
        module_id: Optional[int] = None,
    ) -> RerankResult:
        """Benchmark mode: re-issue the hybrid search with semantic ranking.

        course_id/module_id define the search scope exactly like the original
        retrieval (None/None → Drupal-only, same as KiCampusRetriever).
        """
        from src.llm.objects.retriever import _build_odata_filter

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

        reranked, dropped = self.apply_min_score(reranked)
        if dropped:
            logger.debug(
                "AzureSemanticReranker: dropped %d chunks below min_score=%.2f (normalized)",
                dropped, self.min_score,
            )

        return RerankResult(
            nodes=reranked,
            latency_ms=latency_ms,
            estimated_cost_eur=_EUR_PER_QUERY,
            metadata={
                "semantic_config": _SEMANTIC_CONFIG,
                "scope": {"course_id": course_id, "module_id": module_id},
                "dropped_below_min_score": dropped,
            },
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
