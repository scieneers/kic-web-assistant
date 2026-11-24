"""Socratic v2 Translate Node — 1:1 translation of the previous tutor message.

Testing feedback (2026-11): asked to translate its last message, the tutor
ran a normal tutoring turn and paraphrased itself instead of translating. A
translation request (recognized in contextualize_and_route, see
socratic_v2_language.classify_language_request) now skips the policy and the
move generation entirely: this node translates the previous assistant message
as literally as possible into the requested language and leaves the session
state untouched — a posed quiz stays pending (answer letters are preserved),
counters and learner model don't move. The session language was already
switched to the target language by the router, so the dialogue continues in
that language.
"""

import logging

from langfuse.decorators import langfuse_context, observe

from src.llm.objects.LLMs import LLM, Models
from src.llm.prompts.prompt_loader import load_prompt
from src.llm.state.models import GraphState
from src.llm.state.socratic_v2_language import canonical_language, last_assistant_message
from src.llm.streaming import StreamPhaseContext

logger = logging.getLogger(__name__)

SOCRATIC_V2_TRANSLATE_PROMPT = load_prompt("socratic_v2_translate")

_llm = LLM()


@observe()
def socratic_v2_translate(state: GraphState) -> dict:
    """
    Translates the last tutor message into the session language.

    Changes:
    - Sets answer with the translation (the original text if anything fails)
    - Clears v2_translate_last; no other session state changes
    """
    language = canonical_language(state.get("v2_language"))
    original = last_assistant_message(state.get("chat_history") or []) or ""

    translated = None
    if original:
        try:
            # MINI: a short, literal translation is exactly what the small
            # model is good at, and it keeps the GWDG runtime model's timeout
            # budget out of this path. Streamed like any user-facing turn.
            with StreamPhaseContext("final"):
                response = _llm.chat(
                    query=original,
                    chat_history=[],
                    model=Models.MINI,
                    system_prompt=SOCRATIC_V2_TRANSLATE_PROMPT.format(language=language),
                )
            translated = (response.content or "").strip() or None
        except Exception:
            logger.warning("socratic_v2_translate failed — returning the original message", exc_info=True)

    langfuse_context.update_current_observation(
        metadata={
            "executed_move": "TRANSLATE",
            "target_language": language,
            "used_fallback": translated is None,
        }
    )

    return {
        "v2_translate_last": False,
        "v2_last_policy_move": None,  # no policy call
        "v2_last_move": "TRANSLATE",
        "answer": translated or original,
        "citations_markdown": None,
    }
