"""
Node wrapper for retrieving relevant chunks from vector database.
"""

import logging

from langfuse.decorators import observe

from src.llm.state.models import GraphState

logger = logging.getLogger(__name__)

# Module-level singleton
_retriever_instance = None

def get_retriever(use_hybrid: bool = True, n_chunks: int = 10, use_semantic: bool = False):
    """Get or create singleton retriever instance.

    Parameters are fixed on first call (they come from system_config, which is
    constant for the process lifetime).
    """
    global _retriever_instance
    if _retriever_instance is None:
        from src.llm.objects.retriever import KiCampusRetriever
        _retriever_instance = KiCampusRetriever(
            use_hybrid=use_hybrid, n_chunks=n_chunks, use_semantic=use_semantic
        )
    return _retriever_instance


@observe()
def retrieve_chunks(state: GraphState) -> dict:
    """
    Retrieves relevant document chunks using hybrid search.
    
    Changes:
    - Sets state.retrieved (list of TextNode)
    
    Args:
        state: Current graph state with contextualized_query and optional course_id/module_id in runtime_config
        
    Returns:
        Updated state with retrieved chunks
    """
    # Extract optional filters from runtime_config (set by frontend)
    course_id = state["runtime_config"]["course_id"]
    module_id = state["runtime_config"]["module_id"]
    query = state["contextualized_query"]
    retrieve_top_n = state["system_config"]["retrieve_top_n"]
    # Integrated reranking: with the Azure Semantic backend, the semantic
    # ranker runs inside THIS search call (no separate re-query later) — the
    # rerank node then only cuts to top_n and applies min_score.
    use_semantic = state["system_config"].get("reranker_type") == "azure_semantic"

    logger.debug(
        "retrieve_chunks: query=%r, course_id=%s, module_id=%s, top_n=%d, semantic=%s",
        query[:80] if query else None,
        course_id,
        module_id,
        retrieve_top_n,
        use_semantic,
    )

    # Get singleton retriever
    retriever = get_retriever(use_hybrid=True, n_chunks=retrieve_top_n, use_semantic=use_semantic)

    # Retrieve chunks (returns SerializableTextNode)
    nodes = retriever.retrieve(
        query=query,
        course_id=course_id,
        module_id=module_id
    )

    logger.debug("retrieve_chunks: returned %d chunks", len(nodes))
    return {"retrieved": nodes, "retrieval_semantic_ranked": retriever.use_semantic}