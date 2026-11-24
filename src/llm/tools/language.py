"""
Node wrapper for detecting user's language.
"""

from langfuse.decorators import observe

from src.llm.state.models import GraphState

# Module-level singleton
_language_detector_instance = None

def get_language_detector():
    """Get or create singleton language detector instance."""
    global _language_detector_instance
    if _language_detector_instance is None:
        from src.llm.objects.language_detector import LanguageDetector
        _language_detector_instance = LanguageDetector()
    return _language_detector_instance


@observe()
def detect_language(state: GraphState) -> GraphState:
    """
    Detects the language of the user's query.
    
    Changes:
    - Sets state.detected_language
    
    Args:
        state: Current graph state with user_query and chat_history
        
    Returns:
        Updated state with detected_language
    """
    # Get singleton language detector
    detector = get_language_detector()

    # Get necessary variables from state
    query = state["user_query"]
    chat_history = state["chat_history"]
    
    language = detector.detect(
        query=query,
        chat_history=chat_history
    )

    return {"detected_language": language}


@observe()
def detect_language_socratic_v2(state: GraphState) -> dict:
    """Language detection for a Socratic Tutor (v2) turn — the SAME logic as
    the normal chat path (detect_language above), on every turn:

    - same detector and prompt (short message → dominant language of the
      history, explicit "answer in English" wins, default German)
    - same history window: the last CHAT_HISTORY_LIMIT messages. A tutor
      session keeps up to CHAT_HISTORY_LIMIT_SOCRATIC_V2 messages in state;
      passing all of them would only make the call slower and pricier without
      improving a one-word language decision.

    Runs in parallel with retrieval in the v2 graph (like in simple_hop).

    Changes:
    - Sets detected_language (raw detector output) and v2_language (the
      session language, normalized) for the generating node of this turn
    """
    from src.env import env
    from src.llm.state.socratic_v2_language import canonical_language

    detector = get_language_detector()
    history = (state.get("chat_history") or [])[-env.CHAT_HISTORY_LIMIT:]
    language = detector.detect(query=state["user_query"], chat_history=history)
    return {"detected_language": language, "v2_language": canonical_language(language)}