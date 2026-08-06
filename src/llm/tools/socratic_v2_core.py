"""Socratic v2 Core Node — move-policy tutoring loop.

Heart of the v2 workflow. Each turn:
1. A policy LLM call decides the next MOVE (question, hint, micro-explain,
   explain-back, find-the-error, interim summary, encourage, consolidate,
   exit) and updates the learner model.
2. Deterministic guards enforce the pedagogical ladder regardless of what the
   policy says: 3 questions in a row → forced HINT, 2 hints on the same
   concept → forced MICRO_EXPLAIN (which always ends in a transfer check),
   CONSOLIDATE only from the turn floor on (lowered when the learner model
   already shows mastery, so a confident session can close early).
3. A generation LLM call produces the actual tutor message for that move,
   grounded in the retrieved course material, streamed to the user through a
   peek-buffered guard (see _PeekLeakGuard) that screens the opening of the
   response for a leaked-prompt signature before committing to live streaming.

Unlike v1's EXPLAIN branch, an explanation never resets the session — the
micro-explain includes a transfer question and the dialogue continues.
"""

import logging

from langfuse.decorators import langfuse_context, observe

from src.llm.objects.LLMs import LLM, Models
from src.llm.prompts.prompt_loader import load_prompt
from src.llm.state.models import GraphState
from src.llm.state.socratic_v2_routing import (
    V2_EXIT_MESSAGE,
    V2_QUESTION_MOVES,
    build_pending_quiz,
    grade_quiz_answer,
    initial_learner_model,
    format_learner_model,
    mastery_reached,
    merge_learner_update,
    render_quiz_message,
    reset_socratic_v2_state,
    run_policy,
    select_quiz_item,
)
from src.llm.streaming import StreamPhaseContext, TokenCallbackContext, token_callback_var

logger = logging.getLogger(__name__)

# Load prompt once at module level
SOCRATIC_V2_GENERATE_PROMPT = load_prompt("socratic_v2_generate")

# Initialize LLM instance at module level
_llm = LLM()

# Guardrails of the pedagogical ladder — deterministic, not up to the policy:
MAX_QUESTION_STREAK = 3  # question moves in a row before a HINT is forced
MAX_HINTS_PER_CONCEPT = 2  # hints on one concept before a MICRO_EXPLAIN is forced
MIN_CORE_TURNS_BEFORE_CONSOLIDATE = 4  # core exchanges before CONSOLIDATE is allowed to fire
# Lowered floor for learners who are already demonstrably safe (see
# mastery_reached) — a fast, confident session may close early; the floor stays
# as a safety net against a policy that consolidates after a single exchange.
MIN_CORE_TURNS_WHEN_MASTERED = 2

GENERATION_FALLBACK = "Kannst du deine Überlegung genauer erklären?"

# Tell-tale field labels from the engineered generation query / system prompt.
# Smaller models occasionally echo their instructions or the query template
# instead of answering (observed with Gemma4) — any of these appearing is a
# strong signal that happened, since a genuine short German tutor reply about
# an AI concept would never contain these exact label strings.
_LEAK_MARKERS = (
    "ZIELKONZEPT:",
    "SITZUNGS-LERNZIEL:",
    "LERNSTAND:",
    "KURSMATERIAL:",
    "ZUG:",
    "STRIKTE REGELN",
    "SO FÜHRST DU DEN ZUG AUS",
    "# ROLLE #",
    "# EINGABE #",
)

# Chars buffered before a stream commits to live token-by-token output. Large
# enough to contain the observed leak's opening ("# ROLLE #\nDu bist ein
# herausragender sokratischer Tutor...") but small enough that a clean
# response still starts streaming almost immediately.
_LEAK_PEEK_THRESHOLD = 120


def _looks_like_leaked_prompt(text: str) -> bool:
    """True if a generation response echoes the prompt/query instead of answering."""
    return any(marker in text for marker in _LEAK_MARKERS)


class _PeekLeakGuard:
    """Buffers the start of a token stream to screen for a leaked-prompt
    signature before forwarding anything to the real callback.

    Mirrors question_answerer.py's SmartStreamCallback peek-then-commit
    pattern (used there for the "NO ANSWER FOUND" sentinel), applied to a
    different failure mode. Once the buffer clears the check, every
    subsequent token is forwarded immediately — the common case streams live
    with only a small initial delay, and a leak starting anywhere in the
    checked prefix never reaches the learner's screen at all.
    """

    def __init__(self, outer_callback) -> None:
        self._outer_callback = outer_callback
        self._buffer = ""
        self._peeking = True
        self.leaked = False

    def feed(self, delta: str) -> None:
        if self.leaked:
            return  # already aborted this attempt — drop any further tokens
        if not self._peeking:
            self._outer_callback(delta)
            return
        self._buffer += delta
        if _looks_like_leaked_prompt(self._buffer):
            self.leaked = True
            return
        if len(self._buffer) >= _LEAK_PEEK_THRESHOLD:
            self._peeking = False
            self._outer_callback(self._buffer)

    def finalize_clean(self) -> None:
        """Call once generation finished without tripping the guard — flushes
        a still-buffered prefix (response shorter than the peek threshold)."""
        if self._peeking and self._buffer and not self.leaked:
            self._outer_callback(self._buffer)


def _generate(query_for_llm: str, chat_history: list, model, *, stream: bool) -> tuple[str | None, bool]:
    """One LLM call for the generation step. Returns ``(text, leaked)``:
    ``text`` is None only for a genuinely empty response (no content at all —
    distinct from a leak, and not worth retrying); ``leaked`` is True if a
    leaked-prompt signature was detected (streamed: caught live by the peek
    guard: buffered: checked on the full text). ``stream=False`` forces the
    buffered path regardless of any outer streaming context — used for
    retries, where correctness matters more than typing effect."""
    outer_callback = token_callback_var.get()
    if not stream or outer_callback is None:
        with TokenCallbackContext(None):
            response = _llm.chat(
                query=query_for_llm, chat_history=chat_history, model=model, system_prompt=SOCRATIC_V2_GENERATE_PROMPT
            )
        text = response.content.strip() if response.content else None
        return text, bool(text and _looks_like_leaked_prompt(text))

    guard = _PeekLeakGuard(outer_callback)
    with TokenCallbackContext(guard.feed), StreamPhaseContext("final"):
        response = _llm.chat(
            query=query_for_llm, chat_history=chat_history, model=model, system_prompt=SOCRATIC_V2_GENERATE_PROMPT
        )
    text = response.content.strip() if response.content else None
    leaked = guard.leaked or bool(text and _looks_like_leaked_prompt(text))
    if not leaked:
        guard.finalize_clean()
    return text, leaked


def _generate_validated(query_for_llm: str, chat_history: list, model) -> str:
    """Generate the tutor message with a streamed peek-guard; on a detected
    leak, retry once (buffered, no live streaming) against the more reliable
    Azure fallback model; if that also leaks, use the safe static fallback.
    A genuinely empty response (not a leak) goes straight to the static
    fallback without a retry. A leaked response never reaches the learner."""
    text, leaked = _generate(query_for_llm, chat_history, model, stream=True)
    if not leaked:
        return text if text is not None else GENERATION_FALLBACK

    logger.warning("socratic_v2 generation looked like a leaked prompt — retrying with Azure fallback")
    retry_text, retry_leaked = _generate(query_for_llm, chat_history, Models.AZURE_FALLBACK, stream=False)
    if retry_text is None or retry_leaked:
        logger.warning("socratic_v2 generation still leaked after retry — using static fallback")
        return GENERATION_FALLBACK
    return retry_text


def _grade_pending_quiz(
    *,
    pending_quiz: dict,
    user_query: str,
    chat_history: list,
    model,
    learner_model: dict,
    session_goal: str | None,
    reranked: list,
    hint_count: int,
    target_concept: str | None,
    core_turns: int,
) -> dict:
    """Resolve a posed quiz question: deterministic grading, evidence-based
    learner-model update, LLM only phrases the feedback."""
    verdict = grade_quiz_answer(pending_quiz, user_query)

    # Evidence-based learner-model update — a graded answer is a stronger
    # signal than any LLM assessment.
    concept = pending_quiz.get("concept") or target_concept
    if concept and verdict["recognized"]:
        learner_model = merge_learner_update(
            learner_model, {"konzepte": {concept: "sicher" if verdict["correct"] else "wackelig"}}
        )

    grading_summary = (
        f"ERKANNTE ANTWORT: {verdict['matched_option'] or 'keine der Optionen erkannt'}\n"
        f"BEWERTUNG: {'RICHTIG' if verdict['correct'] else 'FALSCH' if verdict['recognized'] else 'NICHT ERKANNT'}\n"
        f"KORREKTE ANTWORT(EN): {', '.join(verdict['correct_options'])}"
    )
    query_for_llm = f"""ZUG: QUIZ_FEEDBACK
QUIZFRAGE: {pending_quiz.get('question')}
{grading_summary}
SITZUNGS-LERNZIEL: {session_goal or "noch nicht vereinbart"}
LERNSTAND: {format_learner_model(learner_model)}

KURSMATERIAL:
{chr(10).join(f"[Material {i + 1}]{chr(10)}{chunk.text}" for i, chunk in enumerate(reranked)) or "Kein spezifisches Kursmaterial gefunden."}

ANTWORT DER LERNENDEN PERSON: {user_query}"""

    answer = _generate_validated(query_for_llm, chat_history, model)

    langfuse_context.update_current_observation(
        metadata={
            "executed_move": "QUIZ_FEEDBACK",
            "quiz_correct": verdict["correct"],
            "quiz_recognized": verdict["recognized"],
            "target_concept": concept,
            "learner_model": learner_model,
        }
    )

    return {
        "socratic_v2_phase": "core",
        "v2_learner_model": learner_model,
        "v2_pending_quiz": None,
        "v2_hint_count": hint_count,
        "v2_question_streak": 0,  # the resolution IS the Gegenwert
        "v2_core_turns": core_turns + 1,
        "v2_last_policy_move": None,  # no policy call — grading is deterministic
        "v2_last_move": "QUIZ_FEEDBACK",
        "answer": answer,
        "citations_markdown": None,
    }


def _generate_move_response(
    move: str,
    target_concept: str | None,
    session_goal: str | None,
    learner_model: dict,
    reranked: list,
    user_query: str,
    chat_history: list,
    model,
) -> str:
    """Second LLM call: produce the tutor message for the chosen move."""
    materials = (
        "\n\n".join(f"[Material {i + 1}]\n{chunk.text}" for i, chunk in enumerate(reranked))
        if reranked
        else "Kein spezifisches Kursmaterial gefunden — arbeite mit dem bisherigen Gespräch."
    )

    query_for_llm = f"""ZUG: {move}
ZIELKONZEPT: {target_concept or "noch offen"}
SITZUNGS-LERNZIEL: {session_goal or "noch nicht vereinbart — gerade in Klärung"}
LERNSTAND: {format_learner_model(learner_model)}

KURSMATERIAL:
{materials}

LETZTE ANTWORT DER LERNENDEN PERSON: {user_query}"""

    return _generate_validated(query_for_llm, chat_history, model)


@observe(name="socratic_v2_core")
def socratic_v2_core(state: GraphState) -> dict:
    """
    One tutoring turn: policy → guards → grounded generation.

    Flow transitions (via socratic_v2_phase):
    - → "core" (default): dialogue continues
    - → "consolidation": policy chose CONSOLIDATE (only honored from the turn
      floor on — MIN_CORE_TURNS_BEFORE_CONSOLIDATE normally,
      MIN_CORE_TURNS_WHEN_MASTERED once the learner model shows mastery; a
      premature CONSOLIDATE is overridden to FRAGE, same as the other
      guards), the learner was asked to summarize — the next user turn is
      handled by the consolidation node
    - → None (reset): policy chose EXIT

    Changes:
    - Updates v2_learner_model, v2_session_goal, v2_target_concept and the
      hint/question-streak counters
    - Sets answer with the tutor message
    """
    user_query = state["user_query"]
    chat_history = state["chat_history"]
    model = state["runtime_config"]["model"]
    reranked = state.get("reranked") or []

    learning_objectives = state.get("v2_learning_objectives") or []
    session_goal = state.get("v2_session_goal")
    learner_model = state.get("v2_learner_model") or initial_learner_model()
    hint_count = state.get("v2_hint_count", 0)
    question_streak = state.get("v2_question_streak", 0)
    previous_concept = state.get("v2_target_concept")
    quiz_items = state.get("v2_quiz_items") or []
    pending_quiz = state.get("v2_pending_quiz")
    core_turns = state.get("v2_core_turns", 0)

    # 0) A posed quiz question is graded DETERMINISTICALLY against the known
    #    solution — no policy call, no LLM judgement on correctness. Only the
    #    feedback wording comes from the generation LLM.
    if pending_quiz:
        return _grade_pending_quiz(
            pending_quiz=pending_quiz,
            user_query=user_query,
            chat_history=chat_history,
            model=model,
            learner_model=learner_model,
            session_goal=session_goal,
            reranked=reranked,
            hint_count=hint_count,
            target_concept=previous_concept,
            core_turns=core_turns,
        )

    unasked_quiz_items = sum(1 for item in quiz_items if not item.get("asked"))

    # 1) Policy: which move, and what did we learn about the learner?
    policy = run_policy(
        user_query=user_query,
        chat_history=chat_history,
        learning_objectives=learning_objectives,
        session_goal=session_goal,
        learner_model=learner_model,
        hint_count=hint_count,
        question_streak=question_streak,
        model=model,
        available_quiz_items=unasked_quiz_items,
    )
    move = policy["move"]
    policy_move = move  # the policy's raw choice, before quiz fallback and guards

    if move == "EXIT":
        logger.debug("socratic_v2_core: policy chose EXIT — resetting v2 state")
        langfuse_context.update_current_observation(
            metadata={
                "policy_move": "EXIT",
                "executed_move": "EXIT",
                "policy_reasoning": policy.get("begruendung"),
                "learner_model": learner_model,
                "core_turns": core_turns,
                "next_phase": None,
            }
        )
        # Tracking fields survive the reset so the final state snapshot still
        # shows how the session ended (nothing in the app reads them).
        return {
            **reset_socratic_v2_state(),
            "v2_last_policy_move": "EXIT",
            "v2_last_move": "EXIT",
            "answer": V2_EXIT_MESSAGE,
            "citations_markdown": None,
        }

    # Apply state updates from the policy before the guards.
    learner_model = merge_learner_update(learner_model, policy.get("learner_update"))
    if policy.get("lernziel_session"):
        session_goal = policy["lernziel_session"]
    target_concept = policy.get("zielkonzept") or previous_concept
    concept_changed = bool(previous_concept) and target_concept != previous_concept
    effective_hint_count = 0 if concept_changed else hint_count

    # QUIZ: pose a REAL course quiz question — rendered deterministically
    # (no LLM, nothing can leak), graded on the next turn. Falls back to
    # FRAGE when no unasked item is left despite the policy's choice.
    if move == "QUIZ":
        item_index = select_quiz_item(quiz_items, target_concept)
        pending = build_pending_quiz(quiz_items[item_index], concept=target_concept) if item_index is not None else None
        if pending:
            updated_items = [dict(item) for item in quiz_items]
            updated_items[item_index]["asked"] = True
            langfuse_context.update_current_observation(
                metadata={
                    "policy_move": "QUIZ",
                    "executed_move": "QUIZ",
                    "policy_reasoning": policy.get("begruendung"),
                    "quiz_question": pending["question"],
                    "target_concept": target_concept,
                }
            )
            return {
                "socratic_v2_phase": "core",
                "v2_session_goal": session_goal,
                "v2_learner_model": learner_model,
                "v2_target_concept": target_concept,
                "v2_hint_count": effective_hint_count,
                # A quiz gives its resolution next turn — it does not count
                # into the question streak.
                "v2_question_streak": question_streak,
                "v2_quiz_items": updated_items,
                "v2_pending_quiz": pending,
                "v2_core_turns": core_turns + 1,
                "v2_last_policy_move": "QUIZ",
                "v2_last_move": "QUIZ",
                "answer": render_quiz_message(pending),
                "citations_markdown": None,
            }
        logger.debug("socratic_v2_core: QUIZ chosen but no usable item — falling back to FRAGE")
        move = "FRAGE"

    # 2) Deterministic guards — the ladder holds even if the policy drifts.
    original_move = move
    if move in V2_QUESTION_MOVES and question_streak >= MAX_QUESTION_STREAK:
        move = "HINT"
    if move == "HINT" and effective_hint_count >= MAX_HINTS_PER_CONCEPT:
        move = "MICRO_EXPLAIN"
    if move == "CONSOLIDATE":
        min_turns = (
            MIN_CORE_TURNS_WHEN_MASTERED
            if mastery_reached(learner_model)
            else MIN_CORE_TURNS_BEFORE_CONSOLIDATE
        )
        if core_turns < min_turns:
            move = "FRAGE"
    if move != original_move:
        logger.debug(
            "socratic_v2_core: guard override %s → %s (streak=%d, hints=%d, core_turns=%d)",
            original_move,
            move,
            question_streak,
            effective_hint_count,
            core_turns,
        )

    # 3) Generate the tutor message for the (possibly overridden) move.
    answer = _generate_move_response(
        move=move,
        target_concept=target_concept,
        session_goal=session_goal,
        learner_model=learner_model,
        reranked=reranked,
        user_query=user_query,
        chat_history=chat_history,
        model=model,
    )

    # 4) Update the counters for the next turn.
    if move == "HINT":
        new_hint_count = effective_hint_count + 1
    elif move == "MICRO_EXPLAIN":
        new_hint_count = 0  # explanation resolves the ladder for this concept
    else:
        new_hint_count = effective_hint_count
    new_question_streak = question_streak + 1 if move in V2_QUESTION_MOVES else 0

    next_phase = "consolidation" if move == "CONSOLIDATE" else "core"

    langfuse_context.update_current_observation(
        metadata={
            "policy_move": policy_move,
            "executed_move": move,
            "policy_reasoning": policy.get("begruendung"),
            "target_concept": target_concept,
            "session_goal": session_goal,
            "hint_count": new_hint_count,
            "question_streak": new_question_streak,
            "learner_model": learner_model,
            "next_phase": next_phase,
            "core_turns": core_turns + 1,
        }
    )

    return {
        "socratic_v2_phase": next_phase,
        "v2_session_goal": session_goal,
        "v2_learner_model": learner_model,
        "v2_target_concept": target_concept,
        "v2_hint_count": new_hint_count,
        "v2_question_streak": new_question_streak,
        "v2_core_turns": core_turns + 1,
        "v2_last_policy_move": policy_move,
        "v2_last_move": move,
        "answer": answer,
        "citations_markdown": None,
    }
