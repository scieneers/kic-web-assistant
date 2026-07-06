"""
Node wrapper for parsing citations and converting [docN] markers to clickable links.
"""

from langfuse.decorators import observe

from src.llm.state.models import GraphState, get_doc_as_textnodes

# Module-level singleton
_citation_parser_instance = None

def get_citation_parser():
    """Get or create singleton citation parser instance."""
    global _citation_parser_instance
    if _citation_parser_instance is None:
        from src.llm.objects.citation_parser import CitationParser
        _citation_parser_instance = CitationParser()
    return _citation_parser_instance


@observe()
def parse_citations(state: GraphState) -> dict:
    """
    Parses [docN] citations in answer and converts them to markdown links.
    
    Changes:
    - Sets state.citations_markdown (answer with clickable citation links)
    
    Args:
        state: Current graph state with answer and reranked chunks
        
    Returns:
        Updated state with parsed citations
    """
    # Fan-in guard: defer if answer_node hasn't produced an answer yet.
    answer = state.get("answer")
    if answer is None:
        return {}

    # Get singleton citation parser
    parser = get_citation_parser()
    sources = get_doc_as_textnodes(state, "reranked")
    
    # Parse citations in answer (convert SerializableTextNode to TextNode)
    parsed_answer = parser.parse(
        answer=answer,
        source_documents=sources
    )
    
    return {"citations_markdown": parsed_answer}