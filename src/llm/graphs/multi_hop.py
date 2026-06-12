"""
Multi-Hop Subgraph - Complex queries requiring multiple retrieval steps.

For questions that need information from multiple diverse documents
(e.g., comparisons, synthesis, multi-step reasoning).

Flow:
1. Decompose complex query into sub-queries
2. Retrieve chunks for ALL sub-queries in parallel (low latency)
3. Synthesize: Combine and deduplicate all retrieved contexts
4. Rerank: Select most relevant chunks for the original query
5. Generate answer with citations
"""

from langgraph.graph import StateGraph, START, END

from src.llm.state.models import GraphState
from src.llm.tools.decompose import decompose_query
from src.llm.tools.retrieve_multi import retrieve_multi_parallel
from src.llm.tools.synthesize import synthesize_answer
from src.llm.tools.rerank import rerank_chunks
from src.llm.tools.language import detect_language
from src.llm.tools.answer import generate_answer
from src.llm.tools.citation import parse_citations


def build_multi_hop_graph() -> StateGraph:
    """
    Builds the multi_hop subgraph for complex queries.
    
    Flow:
        START ─┬→ decompose → retrieve_multi → synthesize → rerank ─┐
               └→ detect_language ────────────────────────────────┴→ answer → citation → END

    Key features:
    - decompose_query: Breaks complex query into self-contained sub-queries
    - retrieve_multi_parallel: Retrieves chunks for all sub-queries simultaneously
    - synthesize_answer: Combines and deduplicates all contexts
    - rerank_chunks: Selects top-N most relevant for original query

    Language detection runs in parallel with the whole retrieval pipeline (it
    only needs the query + chat history), so its LLM call adds ~0 wall-clock.
    The answer node joins both branches; detected_language and the retrieval
    keys are disjoint state fields, so the parallel writes never collide.
    """
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("decompose_node", decompose_query)
    graph.add_node("retrieve_multi_node", retrieve_multi_parallel)
    graph.add_node("synthesize_node", synthesize_answer)
    graph.add_node("rerank_node", rerank_chunks)
    graph.add_node("detect_language_node", detect_language)
    graph.add_node("answer_node", generate_answer)
    graph.add_node("citation_node", parse_citations)

    # Fan out: the multi-hop retrieval pipeline and language detection run concurrently.
    graph.add_edge(START, "decompose_node")
    graph.add_edge(START, "detect_language_node")
    graph.add_edge("decompose_node", "retrieve_multi_node")
    graph.add_edge("retrieve_multi_node", "synthesize_node")
    graph.add_edge("synthesize_node", "rerank_node")
    # Fan in: answer waits for both the reranked sources and the detected language.
    graph.add_edge("rerank_node", "answer_node")
    graph.add_edge("detect_language_node", "answer_node")
    graph.add_edge("answer_node", "citation_node")
    graph.add_edge("citation_node", END)

    return graph.compile()