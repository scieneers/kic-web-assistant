"""Socratic v2 Consolidation Node — the learner summarizes, the tutor closes.

Final step of the v2 workflow. The previous core turn (move CONSOLIDATE) asked
the learner to summarize in their own words; this node takes that summary and
produces the session close: honest feedback with gap-filling, a visible
learning status ("Sicher: … / Noch wackelig: …" from the learner model), one
next-step recommendation, and a friendly goodbye.

Afterwards ALL v2 state is reset — the session is complete, the next message
is normal chat again (or a fresh Lernmodus start via the button).
"""

import json
import logging

from langfuse.decorators import langfuse_context, observe

from src.llm.objects.LLMs import LLM
from src.llm.prompts.prompt_loader import load_prompt
from src.llm.state.models import GraphState
from src.llm.state.socratic_v2_routing import initial_learner_model, reset_socratic_v2_state
from src.llm.streaming import StreamPhaseContext

logger = logging.getLogger(__name__)

# Load prompt once at module level
SOCRATIC_V2_CONSOLIDATION_PROMPT = load_prompt("socratic_v2_consolidation")

# Initialize LLM instance at module level
_llm = LLM()

CONSOLIDATION_FALLBACK = (
    "Danke für deine Zusammenfassung — damit ist die Lernsession abgeschlossen! "
    "Du kannst jetzt normal weiterfragen oder den Lernmodus jederzeit erneut starten."
)


@observe()
def socratic_v2_consolidation(state: GraphState) -> dict:
    """
    Closes the session based on the learner's own summary.

    Changes:
    - Resets ALL v2 state (session complete)
    - Sets answer with feedback + learning status + next step + goodbye
    """
    user_query = state["user_query"]
    chat_history = state["chat_history"]
    model = state["runtime_config"]["model"]

    session_goal = state.get("v2_session_goal")
    learner_model = state.get("v2_learner_model") or initial_learner_model()
    learning_objectives = state.get("v2_learning_objectives") or []

    query_for_llm = f"""SITZUNGS-LERNZIEL: {session_goal or "nicht explizit vereinbart"}
LERNENDEN-MODELL: {json.dumps(learner_model, ensure_ascii=False)}
MODUL-LERNZIELE: {"; ".join(learning_objectives) or "unbekannt"}

ZUSAMMENFASSUNG DER LERNENDEN PERSON: {user_query}"""

    # Stream the closing message like any other user-facing tutor turn.
    with StreamPhaseContext("final"):
        response = _llm.chat(
            query=query_for_llm,
            chat_history=chat_history,
            model=model,
            system_prompt=SOCRATIC_V2_CONSOLIDATION_PROMPT,
        )

    answer = response.content.strip() if response.content else CONSOLIDATION_FALLBACK

    # Final learner state of the session — after this the checkpoint is reset,
    # so this observation is the last place the learner model is visible.
    langfuse_context.update_current_observation(
        metadata={
            "executed_move": "CONSOLIDATION",
            "session_goal": session_goal,
            "learner_model": learner_model,
            "n_learning_objectives": len(learning_objectives),
            "used_fallback": not bool(response.content),
        }
    )

    # Tracking fields survive the reset so the final state snapshot shows the
    # session ended via consolidation (as opposed to EXIT).
    return {
        **reset_socratic_v2_state(),
        "v2_last_policy_move": None,
        "v2_last_move": "CONSOLIDATION",
        "answer": answer,
        "citations_markdown": None,
    }
