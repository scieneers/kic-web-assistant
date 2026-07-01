"""
Node wrapper for retrieving relevant chunks from vector database.
"""

import logging

from langfuse.decorators import observe

from src.llm.state.models import GraphState

logger = logging.getLogger(__name__)

# Module-level singleton
_retriever_instance = None

def get_retriever(use_hybrid: bool = True, n_chunks: int = 10):
    """Get or create singleton retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        from src.llm.objects.retriever import KiCampusRetriever
        _retriever_instance = KiCampusRetriever(use_hybrid=use_hybrid, n_chunks=n_chunks)
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

    logger.debug(
        "retrieve_chunks: query=%r, course_id=%s, module_id=%s, top_n=%d",
        query[:80] if query else None,
        course_id,
        module_id,
        retrieve_top_n,
    )

    # Get singleton retriever
    retriever = get_retriever(use_hybrid=True, n_chunks=retrieve_top_n)

    # Retrieve chunks (returns SerializableTextNode)
    nodes = retriever.retrieve(
        query=query,
        course_id=course_id,
        module_id=module_id
    )

    logger.debug("retrieve_chunks: returned %d chunks", len(nodes))
    return {"retrieved": nodes}