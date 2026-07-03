"""
Node wrapper for reranking retrieved chunks.
"""

import logging

from langfuse.decorators import observe

from src.llm.state.models import GraphState, get_doc_as_textnodes

logger = logging.getLogger(__name__)

# Module-level singleton
_reranker_instance = None

def get_reranker(reranker_top_n: int):
    """Get or create singleton reranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        from src.llm.objects.reranker import Reranker
        _reranker_instance = Reranker(reranker_top_n)
    return _reranker_instance

@observe()
def rerank_chunks(state: GraphState) -> dict:
    """
    Reranks retrieved chunks using LLM for improved precision.
    
    Changes:
    - Sets state.reranked (list of top-N reranked TextNodes)
    
    Args:
        state: Current graph state with contextualized_query, retrieved, and configs
        
    Returns:
        Updated state with reranked chunks
    """
    # Get necessary variables from state
    model = state["runtime_config"]["model"]
    # Fallback to user_query if contextualized_query is not available
    query = state["contextualized_query"] or state["user_query"]
    rerank_top_n = state["system_config"]["rerank_top_n"]

    logger.debug(
        "rerank_chunks: query=%r, model=%s, top_n=%d, input_chunks=%d",
        query[:80] if query else None,
        model,
        rerank_top_n,
        len(state.get("retrieved", [])),
    )

    # Guard: If no query available, cannot rerank
    if not query:
        logger.debug("rerank_chunks: no query — skipping rerank, returning retrieved as-is")
        return {"reranked": state.get("retrieved", [])}

    # Guard: If no documents retrieved or too few for reranking, skip reranking
    if not state["retrieved"] or len(state["retrieved"]) == 0:
        logger.debug("rerank_chunks: no retrieved docs — returning empty list")
        return {"reranked": []}

    # If only 1 node, no need to rerank
    if len(state["retrieved"]) == 1:
        logger.debug("rerank_chunks: only 1 chunk — skipping rerank")
        return {"reranked": state["retrieved"]}

    # Get shared reranker singleton
    reranker = get_reranker(rerank_top_n)

    # Convert to TextNode for reranker component
    retrieved_nodes = get_doc_as_textnodes(state, "retrieved")

    # Rerank (return SerializableTextNode)
    reranked_nodes = reranker.rerank(
        query=query,
        nodes=retrieved_nodes,
        model=model
    )

    logger.debug("rerank_chunks: %d → %d chunks after rerank", len(retrieved_nodes), len(reranked_nodes))
    # Return top-N
    return {"reranked": reranked_nodes}