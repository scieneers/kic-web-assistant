"""
Summarize Subgraph - full-scope content summary, no relevance ranking.

For "fasse dieses Modul/diesen Kurs zusammen" requests: there is no topical
query to rank chunks against, so this fetches the complete course/module
scope (see retrieve_full_scope) instead of the "top 5 most relevant" chunks,
and skips the reranker entirely — reranking assumes a query to score
relevance against, which doesn't exist here.
"""

from langgraph.graph import StateGraph, START, END

from src.llm.state.models import GraphState


def build_summarize_graph() -> StateGraph:
    """
    Builds the summarize subgraph for content-summary requests.

    Flow:
        START ─┬→ retrieve_full_scope ─┐
               └→ detect_language ─────┴→ summarize_answer → citation → END

    Language detection runs in parallel with retrieval, same as simple_hop:
    it only needs the query + chat history, so the LLM detection call is
    hidden behind retrieval latency.
    """
    from src.llm.tools.retrieve import retrieve_full_scope
    from src.llm.tools.language import detect_language
    from src.llm.tools.summarize import generate_summary
    from src.llm.tools.citation import parse_citations

    graph = StateGraph(GraphState)

    graph.add_node("retrieve_full_scope_node", retrieve_full_scope)
    graph.add_node("detect_language_node", detect_language)
    graph.add_node("summarize_answer_node", generate_summary)
    graph.add_node("citation_node", parse_citations)

    graph.add_edge(START, "retrieve_full_scope_node")
    graph.add_edge(START, "detect_language_node")
    graph.add_edge("retrieve_full_scope_node", "summarize_answer_node")
    graph.add_edge("detect_language_node", "summarize_answer_node")
    graph.add_edge("summarize_answer_node", "citation_node")
    graph.add_edge("citation_node", END)

    return graph.compile()
