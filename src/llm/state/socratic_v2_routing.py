"""State helpers and dialogue policy for the Socratic v2 ("Lernmodus v2") subgraph.

Mirrors the role of socratic_routing.py for v1: LLM-based turn assessment plus
state reset helpers. v2 replaces the four-way CONTINUE/HINT/REFLECT/EXPLAIN
classifier with a move policy that also maintains a learner model.
"""

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional

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
    "QUIZ",
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
        "v2_core_turns": 0,
        "v2_scope_title": None,
        "v2_quiz_items": None,
        "v2_pending_quiz": None,
        "v2_last_policy_move": None,
        "v2_last_move": None,
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
    fallback = {
        "move": DEFAULT_MOVE,
        "zielkonzept": None,
        "lernziel_session": None,
        "learner_update": None,
        "begruendung": None,
    }
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
    begruendung = data.get("begruendung")
    return {
        "move": move,
        "zielkonzept": zielkonzept if isinstance(zielkonzept, str) and zielkonzept.strip() else None,
        "lernziel_session": lernziel_session if isinstance(lernziel_session, str) and lernziel_session.strip() else None,
        "learner_update": learner_update if isinstance(learner_update, dict) else None,
        # Prompt asks for a one-sentence rationale — kept for Langfuse analysis.
        "begruendung": begruendung if isinstance(begruendung, str) and begruendung.strip() else None,
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
    available_quiz_items: int = 0,
) -> Dict[str, Any]:
    """One policy LLM call: decide the next move and update the learner model."""
    policy_query = f"""MODUL-LERNZIELE: {"; ".join(learning_objectives) or "unbekannt"}
SITZUNGS-LERNZIEL: {session_goal or "noch nicht vereinbart"}
LERNENDEN-MODELL: {json.dumps(learner_model, ensure_ascii=False)}
HINWEISE ZUM AKTUELLEN KONZEPT: {hint_count}
FRAGEN IN FOLGE OHNE GEGENWERT: {question_streak}
VERFÜGBARE QUIZFRAGEN: {available_quiz_items}

LETZTE ANTWORT DER LERNENDEN PERSON: {user_query}"""

    response = llm.chat(
        query=policy_query,
        chat_history=chat_history,
        model=model,
        system_prompt=POLICY_PROMPT,
    )
    return parse_policy_response(response.content)


# ---------------------------------------------------------------------------
# QUIZ move: real course quiz items, graded deterministically
# ---------------------------------------------------------------------------

# Only kinds with a discrete, checkable answer set are used by the QUIZ move —
# summary/blanks payloads exist in the index but need free-text grading.
GRADEABLE_QUIZ_KINDS = ("multichoice", "singlechoice", "truefalse")

# German synonyms for grading free-text answers to Wahr/Falsch questions.
_TRUTHY = {"wahr", "richtig", "true", "ja", "stimmt", "korrekt"}
_FALSY = {"falsch", "false", "nein", "stimmt nicht", "nicht korrekt", "unwahr"}


def select_quiz_item(quiz_items: List[Dict[str, Any]], target_concept: Optional[str]) -> Optional[int]:
    """Index of the next unasked quiz item, preferring a target-concept match."""
    if not quiz_items:
        return None
    if target_concept:
        needle = target_concept.strip().lower()
        for idx, item in enumerate(quiz_items):
            if item.get("asked"):
                continue
            haystack = " ".join(
                [item.get("question") or ""]
                + (item.get("correct_answers") or [])
                + (item.get("incorrect_answers") or [])
            ).lower()
            if needle and needle in haystack:
                return idx
    for idx, item in enumerate(quiz_items):
        if not item.get("asked"):
            return idx
    return None


def build_pending_quiz(payload: Dict[str, Any], concept: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Lettered option list for one quiz payload; None if it can't be posed.

    Option order is shuffled deterministically (hash-based) so the correct
    answer's position carries no signal but stays stable for a given question.
    """
    kind = payload.get("kind")
    question = (payload.get("question") or "").strip()
    if not question or kind not in GRADEABLE_QUIZ_KINDS:
        return None

    if kind == "truefalse":
        correct = payload.get("correct_answers") or []
        options = [
            {"letter": "A", "text": "Wahr", "correct": "Wahr" in correct},
            {"letter": "B", "text": "Falsch", "correct": "Falsch" in correct},
        ]
    else:
        pairs = [(text, True) for text in payload.get("correct_answers") or []] + [
            (text, False) for text in payload.get("incorrect_answers") or []
        ]
        if len(pairs) < 2 or not any(correct for _, correct in pairs):
            return None
        pairs.sort(key=lambda pair: hashlib.sha256((question + pair[0]).encode()).hexdigest())
        options = [
            {"letter": chr(65 + i), "text": text, "correct": correct}
            for i, (text, correct) in enumerate(pairs)
        ]

    return {"question": question, "kind": kind, "concept": concept, "options": options}


def render_quiz_message(pending: Dict[str, Any]) -> str:
    """Deterministic quiz rendering — no LLM involved, nothing can leak."""
    lines = ["📝 Quizfrage aus dem Kurs:", "", pending["question"], ""]
    lines += [f"{option['letter']}) {option['text']}" for option in pending["options"]]
    lines += ["", "Antworte mit dem Buchstaben oder deiner Antwort."]
    return "\n".join(lines)


def grade_quiz_answer(pending: Dict[str, Any], user_answer: str) -> Dict[str, Any]:
    """Deterministic grading against the known solution — no LLM, no hallucination.

    Matches a bare letter ("b", "B)", "Antwort b"), the option text itself, or
    Wahr/Falsch synonyms. An unrecognized reply is reported as such so the
    feedback LLM can react gracefully instead of grading small talk as wrong.
    """
    answer = (user_answer or "").strip().lower().rstrip(".!?")
    options = pending.get("options") or []
    matched = None

    letter_match = re.fullmatch(r"(?:antwort\s+)?([a-z])[).]?", answer)
    if letter_match:
        letter = letter_match.group(1).upper()
        matched = next((o for o in options if o["letter"] == letter), None)

    if matched is None:
        for option in options:
            normalized = option["text"].strip().lower()
            if answer == normalized or (len(normalized) > 3 and normalized in answer):
                matched = option
                break

    if matched is None and pending.get("kind") == "truefalse":
        if answer in _FALSY:  # check FALSY first — "stimmt nicht" contains "stimmt"
            matched = next((o for o in options if o["text"] == "Falsch"), None)
        elif answer in _TRUTHY:
            matched = next((o for o in options if o["text"] == "Wahr"), None)

    return {
        "recognized": matched is not None,
        "correct": bool(matched and matched.get("correct")),
        "matched_option": matched["text"] if matched else None,
        "correct_options": [o["text"] for o in options if o.get("correct")],
    }
