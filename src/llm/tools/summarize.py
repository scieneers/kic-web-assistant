"""
Node wrapper for generating content summaries using SummaryAnswerer.
"""

import logging

from langfuse.decorators import observe

from src.llm.state.models import GraphState, get_doc_as_textnodes

logger = logging.getLogger(__name__)

# Module-level singleton
_summary_answerer_instance = None

def get_summary_answerer():
    """Get or create singleton summary answerer instance."""
    global _summary_answerer_instance
    if _summary_answerer_instance is None:
        from src.llm.objects.summary_answerer import SummaryAnswerer
        _summary_answerer_instance = SummaryAnswerer()
    return _summary_answerer_instance


def _scope_name(course_id, module_id) -> str:
    """Short German phrase describing the scope, for the no-content fallback
    (used only when no source exists at all to pull an actual name from)."""
    if module_id:
        return "diesen Modulen" if isinstance(module_id, list) and len(module_id) > 1 else "diesem Modul"
    if course_id is not None:
        return "diesem Kurs"
    return "diesem Bereich"


@observe()
def generate_summary(state: GraphState) -> dict:
    """
    Generates a content summary from the full retrieved scope.

    Changes:
    - Sets state.answer

    Args:
        state: Current graph state with chat_history, detected_language, reranked
            (the full scope, set directly by retrieve_full_scope), and configs

    Returns:
        Updated state with generated summary
    """
    # Fan-in guard: this node has two predecessors (retrieve_full_scope and
    # detect_language). LangGraph invokes it after each branch completes;
    # defer if retrieval hasn't run yet.
    if state.get("reranked") is None:
        return {}

    chat_history = state["chat_history"]
    language = state["detected_language"]
    sources = get_doc_as_textnodes(state, "reranked")
    model = state["runtime_config"]["model"]
    course_id = state["runtime_config"]["course_id"]
    module_id = state["runtime_config"]["module_id"]
    focus_hint = state.get("contextualized_query")

    logger.debug(
        "generate_summary: course_id=%s, module_id=%s, model=%s, language=%s, sources=%d",
        course_id,
        module_id,
        model,
        language,
        len(sources),
    )

    answerer = get_summary_answerer()
    response = answerer.summarize(
        chat_history=chat_history,
        sources=sources,
        model=model,
        language=language,
        scope_name=_scope_name(course_id, module_id),
        is_course_level=not module_id,
        focus_hint=focus_hint,
    )

    logger.debug("generate_summary: done, response_len=%d chars", len(response.content))
    return {"answer": response.content, "fallback_type": response.fallback_type}
