"""
Simple Hop Subgraph - Classic RAG workflow.

For specific factual questions answerable with a single retrieval step.
This is the standard RAG flow equivalent to the current system.
"""

from langgraph.graph import StateGraph, START, END

from src.llm.state.models import GraphState


def build_simple_hop_graph() -> StateGraph:
    """
    Builds the simple_hop subgraph for standard RAG queries.
    
    Flow:
        START ─┬→ retrieve → rerank ─┐
               └→ detect_language ───┴→ answer → citation → END

    Language detection runs in parallel with retrieval+reranking: it only needs
    the query + chat history (no dependency on the retrieved sources), so the
    LLM detection call is hidden behind retrieval latency. The answer node joins
    both branches. detected_language and the retrieval keys are disjoint state
    fields, so the parallel writes never collide (no reducer needed).
    """
    from src.llm.tools.retrieve import retrieve_chunks
    from src.llm.tools.rerank import rerank_chunks
    from src.llm.tools.language import detect_language
    from src.llm.tools.answer import generate_answer
    from src.llm.tools.citation import parse_citations

    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("retrieve_node", retrieve_chunks)
    graph.add_node("rerank_node", rerank_chunks)
    graph.add_node("detect_language_node", detect_language)
    graph.add_node("answer_node", generate_answer)
    graph.add_node("citation_node", parse_citations)

    # Fan out: retrieval pipeline and language detection run concurrently.
    graph.add_edge(START, "retrieve_node")
    graph.add_edge(START, "detect_language_node")
    graph.add_edge("retrieve_node", "rerank_node")
    # Fan in: answer waits for both the reranked sources and the detected language.
    graph.add_edge("rerank_node", "answer_node")
    graph.add_edge("detect_language_node", "answer_node")
    graph.add_edge("answer_node", "citation_node")
    graph.add_edge("citation_node", END)

    return graph.compile()