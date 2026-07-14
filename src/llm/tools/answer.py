"""
Node wrapper for generating answers using QuestionAnswerer.
"""

import logging

from langfuse.decorators import observe

from src.llm.state.models import GraphState, get_doc_as_textnodes
from src.llm.objects.question_answerer import NO_CONTENT_IN_MODULE, NO_CONTENT_UNSUPPORTED_TYPE

logger = logging.getLogger(__name__)

# Module-level singleton
_question_answerer_instance = None

def get_question_answerer():
    """Get or create singleton question answerer instance."""
    global _question_answerer_instance
    if _question_answerer_instance is None:
        from src.llm.objects.question_answerer import QuestionAnswerer
        _question_answerer_instance = QuestionAnswerer()
    return _question_answerer_instance


@observe()
def generate_answer(state: GraphState) -> dict:
    """
    Generates answer to user query using reranked sources.
    
    Changes:
    - Sets state.answer
    
    Args:
        state: Current graph state with user_query, chat_history, detected_language, reranked, and configs
        
    Returns:
        Updated state with generated answer
    """
    # Fan-in guard: answer_node has two predecessors (rerank and detect_language).
    # LangGraph invokes it after each branch completes; defer if rerank hasn't run yet.
    if state.get("reranked") is None:
        return {}

    # Get variables from state (convert to LlamaIndex types)
    query = state["user_query"]
    chat_history = state["chat_history"]
    language = state["detected_language"]
    sources = get_doc_as_textnodes(state, "reranked")
    model = state["runtime_config"]["model"]
    is_moodle = state["runtime_config"]["course_id"] is not None
    course_id = state["runtime_config"]["course_id"]

    # EmptyModule-Marker: Modul existiert, hat aber keinen extrahierbaren Inhalt.
    # Kein LLM-Aufruf — direkt den fest definierten Fallback-Text zurückgeben.
    # Zwei Fälle: ein bekannter, bewusst nicht unterstützter Inhaltstyp (z. B.
    # natives Moodle-Quiz — siehe Module.UNSUPPORTED_MODNAME_LABELS) benennt
    # den Grund konkret, statt fälschlich "kein Inhalt" zu suggerieren.
    if sources and all(s.metadata.get("type") == "EmptyModule" for s in sources):
        first = sources[0].metadata
        module_name = first.get("fullname", "Dieses Modul")
        unsupported_label = first.get("unsupported_label")
        if unsupported_label:
            logger.debug(
                "generate_answer: unsupported content type (%s) for %r — returning specific fallback",
                unsupported_label,
                module_name,
            )
            return {
                "answer": NO_CONTENT_UNSUPPORTED_TYPE.format(
                    module_name=module_name, label=unsupported_label, url=first.get("url", "")
                )
            }
        logger.debug("generate_answer: EmptyModule marker detected for %r — returning fixed fallback", module_name)
        return {"answer": NO_CONTENT_IN_MODULE.format(module_name=module_name)}

    # Get singleton question answerer (only when we actually need the LLM)
    answerer = get_question_answerer()

    logger.debug(
        "generate_answer: query=%r, model=%s, language=%s, sources=%d, is_moodle=%s",
        query[:80],
        model,
        language,
        len(sources),
        is_moodle,
    )

    # Generate answer
    response = answerer.answer_question(
        query=query,
        chat_history=chat_history,
        language=language,
        sources=sources,
        model=model,
        is_moodle=is_moodle,
        course_id=course_id
    )

    logger.debug("generate_answer: done, response_len=%d chars", len(response.content))
    # Extract answer text
    return {"answer": response.content}