"""State helpers and dialogue policy for the Socratic v2 ("Lernmodus v2") subgraph.

Mirrors the role of socratic_routing.py for v1: LLM-based turn assessment plus
state reset helpers. v2 replaces the four-way CONTINUE/HINT/REFLECT/EXPLAIN
classifier with a move policy that also maintains a learner model.
"""

import json
import logging
import re
from typing import Any, Dict, List

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.llm.objects.LLMs import LLM, Models
from src.llm.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

# Initiate LLM instance
llm = LLM()
# Load prompt once at module level
POLICY_PROMPT = load_prompt("socratic_v2_policy")

# Same deterministic escape hatch as the v1 keyword exit in contextualize.py —
# works even if the policy LLM misbehaves. The policy additionally handles
# softer exit intents ("lass gut sein", "sag mir einfach die Antwort") itself.
V2_EXIT_KEYWORDS = ["exit", "quit", "stop", "stopp", "beende den lernmodus", "ich möchte aufhören"]

V2_EXIT_MESSAGE = "Du hast den Lernmodus verlassen. Wenn du weitere Fragen hast, stehe ich dir gerne zur Verfügung!"

V2_MOVES = {
    "FRAGE",
    "HINT",
    "MICRO_EXPLAIN",
    "ERKLAER_ZURUECK",
    "FEHLER_FINDEN",
    "ZWISCHENFAZIT",
    "ENCOURAGE",
    "CONSOLIDATE",
    "EXIT",
}
# Moves that put a question to the learner without giving anything back.
# Capped via v2_question_streak (see socratic_v2_core): after 3 in a row the
# next move is forced to HINT — "nie mehr als ~3 Fragen ohne Gegenwert".
V2_QUESTION_MOVES = {"FRAGE", "ERKLAER_ZURUECK", "FEHLER_FINDEN"}

DEFAULT_MOVE = "FRAGE"


def initial_learner_model() -> Dict[str, Any]:
    """Empty learner model, created at session opening and updated every turn."""
    return {"konzepte": {}, "missverstaendnisse": [], "affekt": "neutral"}


def reset_socratic_v2_state() -> Dict[str, Any]:
    """All v2 fields back to a clean slate (exit, consolidation done, or error)."""
    return {
        "socratic_v2_phase": None,
        "v2_learning_objectives": None,
        "v2_key_concepts": None,
        "v2_session_goal": None,
        "v2_learner_model": None,
        "v2_target_concept": None,
        "v2_hint_count": 0,
        "v2_question_streak": 0,
        "v2_scope_title": None,
    }


def format_learner_model(learner_model: Dict[str, Any]) -> str:
    """Compact German one-liner of the learner model for prompt injection."""
    concepts = learner_model.get("konzepte") or {}
    misconceptions = learner_model.get("missverstaendnisse") or []
    affect = learner_model.get("affekt") or "neutral"
    concept_str = ", ".join(f"{name}: {status}" for name, status in concepts.items()) or "noch keine Konzepte erfasst"
    misconception_str = "; ".join(misconceptions) or "keine bekannt"
    return f"Konzepte: {concept_str} | Missverständnisse: {misconception_str} | Affekt: {affect}"


def merge_learner_update(learner_model: Dict[str, Any], update: Dict[str, Any] | None) -> Dict[str, Any]:
    """Merge a policy learner_update into the learner model (non-destructive).

    Concept statuses are overwritten per concept, misconceptions are appended
    (deduplicated), affect is replaced when provided.
    """
    merged = {
        "konzepte": dict(learner_model.get("konzepte") or {}),
        "missverstaendnisse": list(learner_model.get("missverstaendnisse") or []),
        "affekt": learner_model.get("affekt") or "neutral",
    }
    if not update:
        return merged

    for name, status in (update.get("konzepte") or {}).items():
        if isinstance(name, str) and isinstance(status, str):
            merged["konzepte"][name] = status
    for misconception in update.get("missverstaendnisse") or []:
        if isinstance(misconception, str) and misconception not in merged["missverstaendnisse"]:
            merged["missverstaendnisse"].append(misconception)
    if isinstance(update.get("affekt"), str) and update["affekt"]:
        merged["affekt"] = update["affekt"]
    return merged


def parse_policy_response(content: str | None) -> Dict[str, Any]:
    """Parse the policy LLM's JSON output; degrade gracefully to a safe default.

    A malformed policy response must never break the tutoring turn — the
    fallback continues the dialogue with a plain Socratic question and no
    learner-model update.
    """
    fallback = {"move": DEFAULT_MOVE, "zielkonzept": None, "lernziel_session": None, "learner_update": None}
    if not content:
        return fallback

    # Strip markdown fences and grab the outermost JSON object.
    cleaned = re.sub(r"```(?:json)?", "", content).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return fallback
    try:
        data = json.loads(cleaned[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        logger.warning("Socratic v2 policy returned unparseable JSON: %r", content[:200])
        return fallback
    if not isinstance(data, dict):
        return fallback

    move = str(data.get("move", "")).strip().upper()
    if move not in V2_MOVES:
        logger.warning("Socratic v2 policy returned unknown move %r — falling back to %s", move, DEFAULT_MOVE)
        move = DEFAULT_MOVE

    zielkonzept = data.get("zielkonzept")
    lernziel_session = data.get("lernziel_session")
    learner_update = data.get("learner_update")
    return {
        "move": move,
        "zielkonzept": zielkonzept if isinstance(zielkonzept, str) and zielkonzept.strip() else None,
        "lernziel_session": lernziel_session if isinstance(lernziel_session, str) and lernziel_session.strip() else None,
        "learner_update": learner_update if isinstance(learner_update, dict) else None,
    }


def run_policy(
    user_query: str,
    chat_history: List[SerializableChatMessage],
    learning_objectives: List[str],
    session_goal: str | None,
    learner_model: Dict[str, Any],
    hint_count: int,
    question_streak: int,
    model: Models,
) -> Dict[str, Any]:
    """One policy LLM call: decide the next move and update the learner model."""
    policy_query = f"""MODUL-LERNZIELE: {"; ".join(learning_objectives) or "unbekannt"}
SITZUNGS-LERNZIEL: {session_goal or "noch nicht vereinbart"}
LERNENDEN-MODELL: {json.dumps(learner_model, ensure_ascii=False)}
HINWEISE ZUM AKTUELLEN KONZEPT: {hint_count}
FRAGEN IN FOLGE OHNE GEGENWERT: {question_streak}

LETZTE ANTWORT DER LERNENDEN PERSON: {user_query}"""

    response = llm.chat(
        query=policy_query,
        chat_history=chat_history,
        model=model,
        system_prompt=POLICY_PROMPT,
    )
    return parse_policy_response(response.content)
