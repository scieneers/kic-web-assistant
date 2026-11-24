"""
Node wrapper for reranking retrieved chunks.
"""

import logging

from langfuse.decorators import langfuse_context, observe

from src.llm.objects.rerankers.base import BaseReranker
from src.llm.state.models import GraphState, RerankerType

logger = logging.getLogger(__name__)


def _trace_span_metadata(**metadata) -> None:
    """Kompakte Analyse-Metadaten an den aktuellen Langfuse-Span hängen."""
    langfuse_context.update_current_observation(metadata=metadata)

# Singletons keyed by reranker_type so swapping at startup is zero-cost at runtime.
_reranker_instances: dict[str, BaseReranker] = {}


def get_reranker(reranker_type: RerankerType, top_n: int, min_score: float = 0.0) -> BaseReranker:
    """Return (or create) a singleton reranker for the given type.

    Heavy resources (BGE model, Azure clients) are created once per type;
    cheap parameters (top_n, min_score) are re-applied on every call so a
    cached instance never runs with stale configuration.
    """
    if reranker_type not in _reranker_instances:
        if reranker_type == "llm":
            from src.llm.objects.rerankers.llm_reranker import LLMReranker
            _reranker_instances[reranker_type] = LLMReranker(top_n=top_n, min_score=min_score)
        elif reranker_type == "azure_semantic":
            from src.llm.objects.rerankers.azure_semantic_reranker import AzureSemanticReranker
            _reranker_instances[reranker_type] = AzureSemanticReranker(top_n=top_n, min_score=min_score)
        elif reranker_type == "bge":
            from src.llm.objects.rerankers.bge_reranker import BGEReranker
            _reranker_instances[reranker_type] = BGEReranker(top_n=top_n, min_score=min_score)
        else:
            raise ValueError(f"Unknown reranker_type: {reranker_type!r}. Choose: llm, azure_semantic, bge")
    reranker = _reranker_instances[reranker_type]
    reranker.configure(top_n=top_n, min_score=min_score)
    return reranker


@observe()
def rerank_chunks(state: GraphState) -> dict:
    """
    Reranks retrieved chunks using the configured reranker.

    Changes:
    - Sets state.reranked (list of top-N reranked SerializableTextNodes)

    Args:
        state: Current graph state with contextualized_query, retrieved, and configs

    Returns:
        Updated state with reranked chunks
    """
    model = state["runtime_config"]["model"]
    query = state["contextualized_query"] or state["user_query"]
    rerank_top_n = state["system_config"]["rerank_top_n"]
    reranker_type = state["system_config"].get("reranker_type", "llm")
    min_score = state["system_config"].get("min_reranker_score", 0.0)
    # Request scope — index-querying backends (Azure Semantic) must search with
    # the SAME scope as the original retrieval. Never derived from chunk
    # metadata (Drupal course pages carry the course_id they describe).
    course_id = state["runtime_config"].get("course_id")
    module_id = state["runtime_config"].get("module_id")
    # Integrated mode: retrieval already ranked with Azure's semantic ranker —
    # the backend then only cuts to top_n and applies min_score.
    preranked = state.get("retrieval_semantic_ranked", False)

    logger.debug(
        "rerank_chunks: query=%r, model=%s, reranker=%s, top_n=%d, min_score=%.2f, input_chunks=%d, preranked=%s",
        query[:80] if query else None,
        model,
        reranker_type,
        rerank_top_n,
        min_score,
        len(state.get("retrieved", [])),
        preranked,
    )

    if not query:
        logger.debug("rerank_chunks: no query — skipping, returning retrieved as-is")
        _trace_span_metadata(skipped="no_query")
        return {"reranked": state.get("retrieved", [])}

    retrieved = state.get("retrieved", [])
    if not retrieved:
        logger.debug("rerank_chunks: no retrieved docs — returning empty list")
        _trace_span_metadata(skipped="no_retrieved_docs")
        return {"reranked": []}

    # Single-chunk shortcut saves an LLM call / model inference — but not in
    # integrated mode: there the chunk already carries its semantic score, so
    # cutting/thresholding is free and min_score must still apply.
    if len(retrieved) == 1 and not preranked:
        logger.debug("rerank_chunks: only 1 chunk — skipping rerank")
        _trace_span_metadata(skipped="single_chunk")
        return {"reranked": retrieved}

    if rerank_top_n >= len(retrieved):
        logger.warning(
            "rerank_chunks: rerank_top_n (%d) >= input chunks (%d) — reranking won't reduce the result set",
            rerank_top_n,
            len(retrieved),
        )

    reranker = get_reranker(reranker_type, rerank_top_n, min_score)
    result = reranker.rerank(
        query=query,
        nodes=retrieved,
        model=model,
        course_id=course_id,
        module_id=module_id,
        preranked=preranked,
    )

    reranked = result.nodes
    dropped = result.metadata.get("dropped_below_min_score", 0)
    if dropped:
        logger.info(
            "rerank_chunks: dropped %d chunk(s) below min_score=%.2f (%d remaining)",
            dropped, min_score, len(reranked),
        )
    if not reranked:
        # Empty result set — either every chunk scored below min_score ("all
        # hits are weak") or the reranker returned nothing (e.g. LLMRerank
        # format failure). Return empty so the answer node triggers the
        # no-answer fallback instead of passing through unfiltered chunks
        # (which risks weak answers and prompt injection).
        if dropped:
            logger.info(
                "rerank_chunks: all %d chunks below min_score=%.2f — triggering no-answer fallback",
                len(retrieved), min_score,
            )
        else:
            logger.warning(
                "rerank_chunks: reranker returned 0 results for %d input chunks — "
                "returning empty list to trigger no-content fallback",
                len(retrieved),
            )

    logger.debug("rerank_chunks: %d → %d chunks after rerank", len(retrieved), len(reranked))
    _trace_span_metadata(
        reranker_type=reranker_type,
        rerank_top_n=rerank_top_n,
        min_score=min_score,
        preranked=preranked,
        n_input=len(retrieved),
        n_output=len(reranked),
        dropped_below_min_score=dropped,
        no_answer_fallback_triggered=not reranked,
    )
    return {"reranked": reranked}
