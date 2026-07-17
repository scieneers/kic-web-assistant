"""Unit tests for the Socratic v2 ("Lernmodus v2") workflow.

Covers:
- socratic_v2_opening: content loading, objective extraction, scope guards
- socratic_v2_core: policy execution, deterministic guards (hint ladder,
  question streak), learner-model updates, phase transitions, exit
- socratic_v2_consolidation: session close + full state reset
- socratic_v2_routing: policy JSON parsing and learner-model merging
- contextualize_and_route: v2 entry/exit/continue routing
"""

from unittest.mock import MagicMock, patch

import pytest

from src.api.models.serializable_text_node import SerializableTextNode
from src.llm.objects.LLMs import Models
from src.llm.state.socratic_v2_routing import (
    V2_EXIT_MESSAGE,
    build_pending_quiz,
    grade_quiz_answer,
    initial_learner_model,
    merge_learner_update,
    parse_policy_response,
    reset_socratic_v2_state,
    select_quiz_item,
)
from src.llm.tools.contextualize import contextualize_and_route
from src.llm.tools.socratic_v2_consolidation import socratic_v2_consolidation
from src.llm.tools.socratic_v2_core import (
    GENERATION_FALLBACK,
    _generate_validated,
    _looks_like_leaked_prompt,
    _PeekLeakGuard,
    socratic_v2_core,
)
from src.llm.tools.socratic_v2_opening import (
    NO_CONTENT_MESSAGE,
    NO_SCOPE_MESSAGE,
    socratic_v2_opening,
)


def _base_state(**overrides) -> dict:
    state = {
        "user_query": "Was ist Overfitting?",
        "chat_history": [],
        "runtime_config": {
            "model": Models.AZURE_FALLBACK,
            "course_id": 42,
            "module_id": 7,
            "thread_id": "test-socratic-v2",
            "start_socratic": False,
            "start_socratic_v2": False,
        },
        "system_config": {
            "rerank_top_n": 3,
            "retrieve_top_n": 10,
            "enable_socratic": False,
            "enable_socratic_v2": True,
        },
        "socratic_v2_phase": "core",
        "v2_learning_objectives": ["Overfitting erkennen", "Gegenmaßnahmen kennen"],
        "v2_key_concepts": ["Overfitting", "Regularisierung"],
        "v2_session_goal": "Overfitting verstehen",
        "v2_learner_model": initial_learner_model(),
        "v2_target_concept": "Overfitting",
        "v2_hint_count": 0,
        "v2_question_streak": 0,
        "v2_scope_title": "Modul Überwachtes Lernen",
        "reranked": [],
    }
    state.update(overrides)
    return state


def _mock_llm_response(content: str | None) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    return msg


def _make_chunk(text: str, metadata: dict | None = None) -> SerializableTextNode:
    return SerializableTextNode(text=text, metadata=metadata or {})


def _policy(move: str = "FRAGE", **overrides) -> dict:
    policy = {"move": move, "zielkonzept": "Overfitting", "lernziel_session": None, "learner_update": None}
    policy.update(overrides)
    return policy


# ---------------------------------------------------------------------------
# Policy parsing / learner model helpers
# ---------------------------------------------------------------------------


class TestParsePolicyResponse:
    def test_parses_valid_json(self):
        result = parse_policy_response(
            '{"move": "HINT", "zielkonzept": "Overfitting", "lernziel_session": "Ziel", '
            '"learner_update": {"konzepte": {"Overfitting": "wackelig"}}, "begruendung": "Person hängt fest."}'
        )
        assert result["move"] == "HINT"
        assert result["zielkonzept"] == "Overfitting"
        assert result["lernziel_session"] == "Ziel"
        assert result["learner_update"]["konzepte"]["Overfitting"] == "wackelig"
        assert result["begruendung"] == "Person hängt fest."

    def test_missing_begruendung_is_none(self):
        assert parse_policy_response('{"move": "HINT"}')["begruendung"] is None

    def test_strips_markdown_fences(self):
        result = parse_policy_response('```json\n{"move": "CONSOLIDATE"}\n```')
        assert result["move"] == "CONSOLIDATE"

    def test_lowercase_move_is_normalized(self):
        result = parse_policy_response('{"move": "exit"}')
        assert result["move"] == "EXIT"

    def test_unknown_move_falls_back_to_frage(self):
        result = parse_policy_response('{"move": "TANZEN"}')
        assert result["move"] == "FRAGE"

    def test_garbage_falls_back_to_frage(self):
        result = parse_policy_response("Ich bin ein LLM ohne JSON")
        assert result["move"] == "FRAGE"
        assert result["learner_update"] is None

    def test_none_content_falls_back(self):
        assert parse_policy_response(None)["move"] == "FRAGE"

    def test_json_embedded_in_prose(self):
        result = parse_policy_response('Hier mein Ergebnis: {"move": "ENCOURAGE"} — viel Erfolg!')
        assert result["move"] == "ENCOURAGE"


class TestMergeLearnerUpdate:
    def test_merges_concept_status(self):
        merged = merge_learner_update(initial_learner_model(), {"konzepte": {"Overfitting": "sicher"}})
        assert merged["konzepte"]["Overfitting"] == "sicher"

    def test_appends_misconceptions_deduplicated(self):
        base = initial_learner_model()
        base["missverstaendnisse"] = ["A"]
        merged = merge_learner_update(base, {"missverstaendnisse": ["A", "B"]})
        assert merged["missverstaendnisse"] == ["A", "B"]

    def test_none_update_keeps_model(self):
        base = initial_learner_model()
        base["konzepte"]["X"] = "wackelig"
        merged = merge_learner_update(base, None)
        assert merged["konzepte"] == {"X": "wackelig"}

    def test_affect_replaced(self):
        merged = merge_learner_update(initial_learner_model(), {"affekt": "frustriert"})
        assert merged["affekt"] == "frustriert"


# ---------------------------------------------------------------------------
# Opening node
# ---------------------------------------------------------------------------


class TestSocraticV2Opening:
    def _run(self, state, nodes, llm_content="LERNZIEL: Overfitting erkennen\nKONZEPT: Overfitting", quiz_nodes=None):
        retriever = MagicMock()
        retriever.retrieve_all.return_value = nodes
        retriever.retrieve_items.return_value = quiz_nodes or []
        with patch("src.llm.tools.socratic_v2_opening.get_retriever", return_value=retriever), patch(
            "src.llm.tools.socratic_v2_opening._llm.chat", return_value=_mock_llm_response(llm_content)
        ):
            return socratic_v2_opening(state), retriever

    def test_no_scope_does_not_start_session(self):
        state = _base_state()
        state["runtime_config"]["course_id"] = None
        state["runtime_config"]["module_id"] = None
        result = socratic_v2_opening(state)
        assert result["answer"] == NO_SCOPE_MESSAGE
        assert result["socratic_v2_phase"] is None

    def test_empty_module_does_not_start_session(self):
        result, _ = self._run(_base_state(), [_make_chunk("x", {"type": "EmptyModule"})])
        assert result["answer"] == NO_CONTENT_MESSAGE
        assert result["socratic_v2_phase"] is None

    def test_starts_session_with_extracted_objectives(self):
        result, _ = self._run(
            _base_state(),
            [_make_chunk("Inhalt über Overfitting", {"fullname": "ML Modul"})],
            llm_content="LERNZIEL: Overfitting erkennen\nLERNZIEL: Regularisierung anwenden\nKONZEPT: Overfitting\nKONZEPT: Regularisierung",
        )
        assert result["socratic_v2_phase"] == "core"
        assert result["v2_learning_objectives"] == ["Overfitting erkennen", "Regularisierung anwenden"]
        assert result["v2_key_concepts"] == ["Overfitting", "Regularisierung"]
        assert result["v2_scope_title"] == "ML Modul"
        assert "Overfitting erkennen" in result["answer"]
        assert "Lernmodus" in result["answer"]

    def test_initializes_learner_model_and_counters(self):
        result, _ = self._run(_base_state(), [_make_chunk("Inhalt")])
        assert result["v2_learner_model"] == initial_learner_model()
        assert result["v2_hint_count"] == 0
        assert result["v2_question_streak"] == 0
        assert result["v2_session_goal"] is None
        assert result["v2_last_move"] is None
        assert result["v2_last_policy_move"] is None

    def test_extraction_failure_still_starts_session(self):
        result, _ = self._run(_base_state(), [_make_chunk("Inhalt", {"fullname": "ML Modul"})], llm_content="kaputt")
        assert result["socratic_v2_phase"] == "core"
        assert len(result["v2_learning_objectives"]) == 1  # fallback objective

    def test_module_list_uses_all_selected_modules(self):
        # Multi-select means all chosen modules are worked on equally — the
        # full list must reach the retriever, not just the last entry.
        state = _base_state()
        state["runtime_config"]["module_id"] = [3, 5, 9]
        _, retriever = self._run(state, [_make_chunk("Inhalt")])
        retriever.retrieve_all.assert_called_once_with(course_id=42, module_id=[3, 5, 9])


# ---------------------------------------------------------------------------
# Core node
# ---------------------------------------------------------------------------


class TestSocraticV2Core:
    def _run(self, state, policy, llm_content="Gute Überlegung! Was folgt daraus?"):
        with patch("src.llm.tools.socratic_v2_core.run_policy", return_value=policy) as mock_policy, patch(
            "src.llm.tools.socratic_v2_core._llm.chat", return_value=_mock_llm_response(llm_content)
        ) as mock_generate:
            result = socratic_v2_core(state)
        return result, mock_policy, mock_generate

    def test_frage_stays_in_core_and_increments_streak(self):
        result, _, _ = self._run(_base_state(), _policy("FRAGE"))
        assert result["socratic_v2_phase"] == "core"
        assert result["v2_question_streak"] == 1
        assert result["answer"] == "Gute Überlegung! Was folgt daraus?"

    def test_move_tracking_without_guard_override(self):
        result, _, _ = self._run(_base_state(), _policy("FRAGE"))
        assert result["v2_last_policy_move"] == "FRAGE"
        assert result["v2_last_move"] == "FRAGE"

    def test_move_tracking_records_guard_override(self):
        # streak guard forces HINT — policy choice and executed move diverge
        result, _, _ = self._run(_base_state(v2_question_streak=3), _policy("FRAGE"))
        assert result["v2_last_policy_move"] == "FRAGE"
        assert result["v2_last_move"] == "HINT"

    def test_non_question_move_resets_streak(self):
        result, _, _ = self._run(_base_state(v2_question_streak=2), _policy("ZWISCHENFAZIT"))
        assert result["v2_question_streak"] == 0

    def test_question_streak_guard_forces_hint(self):
        result, _, mock_generate = self._run(_base_state(v2_question_streak=3), _policy("FRAGE"))
        generated_query = mock_generate.call_args.kwargs["query"]
        assert "ZUG: HINT" in generated_query
        assert result["v2_hint_count"] == 1
        assert result["v2_question_streak"] == 0

    def test_hint_guard_forces_micro_explain_after_two_hints(self):
        result, _, mock_generate = self._run(_base_state(v2_hint_count=2), _policy("HINT"))
        generated_query = mock_generate.call_args.kwargs["query"]
        assert "ZUG: MICRO_EXPLAIN" in generated_query
        assert result["v2_hint_count"] == 0  # explanation resolves the ladder

    def test_hint_increments_hint_count(self):
        result, _, _ = self._run(_base_state(v2_hint_count=1), _policy("HINT"))
        assert result["v2_hint_count"] == 2

    def test_concept_change_resets_hint_count(self):
        result, _, _ = self._run(
            _base_state(v2_hint_count=2, v2_target_concept="Overfitting"),
            _policy("HINT", zielkonzept="Regularisierung"),
        )
        # new concept → ladder starts fresh, so the hint goes through (no forced explain)
        assert result["v2_hint_count"] == 1
        assert result["v2_target_concept"] == "Regularisierung"

    def test_consolidate_transitions_to_consolidation_phase(self):
        result, _, _ = self._run(_base_state(v2_core_turns=4), _policy("CONSOLIDATE"))
        assert result["socratic_v2_phase"] == "consolidation"
        assert result["v2_last_move"] == "CONSOLIDATE"

    def test_premature_consolidate_is_overridden_to_frage(self):
        # fewer core exchanges than MIN_CORE_TURNS_BEFORE_CONSOLIDATE → guard
        result, _, _ = self._run(_base_state(v2_core_turns=1), _policy("CONSOLIDATE"))
        assert result["socratic_v2_phase"] == "core"
        assert result["v2_last_policy_move"] == "CONSOLIDATE"
        assert result["v2_last_move"] == "FRAGE"

    def test_exit_resets_all_v2_state(self):
        result, _, _ = self._run(_base_state(), _policy("EXIT"))
        assert result["socratic_v2_phase"] is None
        assert result["v2_learner_model"] is None
        assert result["answer"] == V2_EXIT_MESSAGE
        # tracking fields survive the reset so analysis sees how the session ended
        assert result["v2_last_move"] == "EXIT"
        assert result["v2_last_policy_move"] == "EXIT"

    def test_learner_update_is_merged(self):
        result, _, _ = self._run(
            _base_state(),
            _policy("FRAGE", learner_update={"konzepte": {"Overfitting": "wackelig"}, "affekt": "motiviert"}),
        )
        assert result["v2_learner_model"]["konzepte"]["Overfitting"] == "wackelig"
        assert result["v2_learner_model"]["affekt"] == "motiviert"

    def test_session_goal_set_by_policy(self):
        result, _, _ = self._run(
            _base_state(v2_session_goal=None), _policy("FRAGE", lernziel_session="Regularisierung verstehen")
        )
        assert result["v2_session_goal"] == "Regularisierung verstehen"

    def test_generation_failure_uses_fallback(self):
        result, _, _ = self._run(_base_state(), _policy("FRAGE"), llm_content=None)
        assert len(result["answer"]) > 0

    def test_materials_passed_to_generation(self):
        state = _base_state(reranked=[_make_chunk("Overfitting bedeutet ...")])
        _, _, mock_generate = self._run(state, _policy("FRAGE"))
        assert "Overfitting bedeutet ..." in mock_generate.call_args.kwargs["query"]


# ---------------------------------------------------------------------------
# QUIZ move: deterministic posing + grading of real course quiz items
# ---------------------------------------------------------------------------


def _quiz_payload(**overrides) -> dict:
    payload = {
        "kind": "multichoice",
        "question": "Was ist Overfitting?",
        "correct_answers": ["Auswendiglernen der Trainingsdaten"],
        "incorrect_answers": ["Ein zu kleines Modell"],
        "asked": False,
    }
    payload.update(overrides)
    return payload


class TestQuizMove:
    def _run(self, state, policy, llm_content="Feedback."):
        with patch("src.llm.tools.socratic_v2_core.run_policy", return_value=policy) as mock_policy, patch(
            "src.llm.tools.socratic_v2_core._llm.chat", return_value=_mock_llm_response(llm_content)
        ) as mock_generate:
            result = socratic_v2_core(state)
        return result, mock_policy, mock_generate

    def test_quiz_move_poses_question_deterministically(self):
        state = _base_state(v2_quiz_items=[_quiz_payload()])
        result, _, mock_generate = self._run(state, _policy("QUIZ"))
        mock_generate.assert_not_called()  # rendering is deterministic, no LLM
        assert "Was ist Overfitting?" in result["answer"]
        assert "A)" in result["answer"] and "B)" in result["answer"]
        assert "Korrekt" not in result["answer"]  # nothing leaks
        assert result["v2_pending_quiz"] is not None
        assert result["v2_quiz_items"][0]["asked"] is True
        assert result["v2_last_move"] == "QUIZ"
        assert result["v2_last_policy_move"] == "QUIZ"

    def test_quiz_options_carry_ground_truth_in_state_only(self):
        state = _base_state(v2_quiz_items=[_quiz_payload()])
        result, _, _ = self._run(state, _policy("QUIZ"))
        options = result["v2_pending_quiz"]["options"]
        assert sum(1 for o in options if o["correct"]) == 1
        correct_option = next(o for o in options if o["correct"])
        assert correct_option["text"] == "Auswendiglernen der Trainingsdaten"

    def test_quiz_without_items_falls_back_to_frage(self):
        state = _base_state(v2_quiz_items=[])
        result, _, mock_generate = self._run(state, _policy("QUIZ"), llm_content="Was denkst du?")
        assert "ZUG: FRAGE" in mock_generate.call_args.kwargs["query"]
        assert result.get("v2_pending_quiz") is None

    def test_policy_receives_available_quiz_count(self):
        state = _base_state(v2_quiz_items=[_quiz_payload(), _quiz_payload(asked=True)])
        _, mock_policy, _ = self._run(state, _policy("FRAGE"))
        assert mock_policy.call_args.kwargs["available_quiz_items"] == 1

    def _pending_state(self):
        pending = build_pending_quiz(_quiz_payload(), concept="Overfitting")
        return _base_state(v2_pending_quiz=pending), pending

    def test_grading_correct_letter_updates_learner_model(self):
        state, pending = self._pending_state()
        correct_letter = next(o["letter"] for o in pending["options"] if o["correct"])
        state["user_query"] = correct_letter.lower()
        with patch("src.llm.tools.socratic_v2_core.run_policy") as mock_policy, patch(
            "src.llm.tools.socratic_v2_core._llm.chat", return_value=_mock_llm_response("Richtig!")
        ) as mock_generate:
            result = socratic_v2_core(state)
        mock_policy.assert_not_called()  # grading is deterministic
        assert result["v2_learner_model"]["konzepte"]["Overfitting"] == "sicher"
        assert result["v2_pending_quiz"] is None
        assert "BEWERTUNG: RICHTIG" in mock_generate.call_args.kwargs["query"]
        assert result["v2_last_move"] == "QUIZ_FEEDBACK"
        assert result["v2_last_policy_move"] is None  # no policy ran

    def test_grading_wrong_answer_marks_concept_wackelig(self):
        state, pending = self._pending_state()
        wrong = next(o for o in pending["options"] if not o["correct"])
        state["user_query"] = wrong["text"]
        with patch("src.llm.tools.socratic_v2_core.run_policy"), patch(
            "src.llm.tools.socratic_v2_core._llm.chat", return_value=_mock_llm_response("Leider falsch.")
        ) as mock_generate:
            result = socratic_v2_core(state)
        assert result["v2_learner_model"]["konzepte"]["Overfitting"] == "wackelig"
        assert "BEWERTUNG: FALSCH" in mock_generate.call_args.kwargs["query"]

    def test_grading_unrecognized_answer_does_not_touch_learner_model(self):
        state, _ = self._pending_state()
        state["user_query"] = "warum fragst du mich das?"
        with patch("src.llm.tools.socratic_v2_core.run_policy"), patch(
            "src.llm.tools.socratic_v2_core._llm.chat", return_value=_mock_llm_response("Kein Problem!")
        ) as mock_generate:
            result = socratic_v2_core(state)
        assert "Overfitting" not in result["v2_learner_model"]["konzepte"]
        assert "BEWERTUNG: NICHT ERKANNT" in mock_generate.call_args.kwargs["query"]


class TestQuizHelpers:
    def test_grade_truefalse_synonyms(self):
        pending = build_pending_quiz(
            {"kind": "truefalse", "question": "F?", "correct_answers": ["Falsch"], "incorrect_answers": ["Wahr"]}
        )
        assert grade_quiz_answer(pending, "stimmt nicht")["correct"] is True
        assert grade_quiz_answer(pending, "ja")["correct"] is False
        assert grade_quiz_answer(pending, "b")["correct"] is True  # B) Falsch

    def test_build_pending_quiz_shuffles_deterministically(self):
        payload = _quiz_payload(incorrect_answers=["Falsch 1", "Falsch 2", "Falsch 3"])
        first = build_pending_quiz(payload)
        second = build_pending_quiz(payload)
        assert [o["text"] for o in first["options"]] == [o["text"] for o in second["options"]]
        assert len(first["options"]) == 4

    def test_build_pending_quiz_rejects_ungradeable_kind(self):
        assert build_pending_quiz({"kind": "blanks", "question": "F?", "text_with_blanks": "*x*"}) is None

    def test_select_quiz_item_prefers_target_concept(self):
        items = [
            _quiz_payload(question="Was ist ein Neuron?"),
            _quiz_payload(question="Was ist Overfitting?"),
        ]
        assert select_quiz_item(items, "Overfitting") == 1
        assert select_quiz_item(items, None) == 0

    def test_select_quiz_item_skips_asked(self):
        items = [_quiz_payload(asked=True), _quiz_payload()]
        assert select_quiz_item(items, None) == 1
        assert select_quiz_item([_quiz_payload(asked=True)], None) is None


class TestOpeningQuizLoading:
    def test_opening_loads_gradeable_quiz_items(self):
        node_gradeable = MagicMock()
        node_gradeable.metadata = {"payload": {"kind": "truefalse", "question": "F?", "correct_answers": ["Wahr"]}}
        node_blanks = MagicMock()
        node_blanks.metadata = {"payload": {"kind": "blanks", "question": "L?", "text_with_blanks": "*x*"}}

        retriever = MagicMock()
        retriever.retrieve_all.return_value = [MagicMock(metadata={"type": "module", "fullname": "M"}, text="Inhalt")]
        retriever.retrieve_items.return_value = [node_gradeable, node_blanks]
        with patch("src.llm.tools.socratic_v2_opening.get_retriever", return_value=retriever), patch(
            "src.llm.tools.socratic_v2_opening._llm.chat",
            return_value=_mock_llm_response("LERNZIEL: X\nKONZEPT: Y"),
        ):
            result = socratic_v2_opening(_base_state())
        assert len(result["v2_quiz_items"]) == 1  # blanks filtered out
        assert result["v2_quiz_items"][0]["asked"] is False
        assert result["v2_pending_quiz"] is None

    def test_opening_survives_quiz_fetch_failure(self):
        retriever = MagicMock()
        retriever.retrieve_all.return_value = [MagicMock(metadata={"type": "module", "fullname": "M"}, text="Inhalt")]
        retriever.retrieve_items.side_effect = RuntimeError("index down")
        with patch("src.llm.tools.socratic_v2_opening.get_retriever", return_value=retriever), patch(
            "src.llm.tools.socratic_v2_opening._llm.chat",
            return_value=_mock_llm_response("LERNZIEL: X"),
        ):
            result = socratic_v2_opening(_base_state())
        assert result["socratic_v2_phase"] == "core"
        assert result["v2_quiz_items"] == []


# ---------------------------------------------------------------------------
# Leaked-prompt guard: a small model (observed with Gemma4) occasionally
# echoes its own system prompt / the engineered query labels instead of
# generating the tutor message. This must never reach the learner, whether
# streaming is active or not.
# ---------------------------------------------------------------------------


LEAKED_TEXT = (
    "# ROLLE #\nDu bist ein herausragender sokratischer Tutor auf dem KI-Campus.\n"
    "ZUG: FRAGE\nZIELKONZEPT: Funktionsweise von LLMs\nSITZUNGS-LERNZIEL: ...\n"
    "LERNSTAND: ...\nKURSMATERIAL:\n...\n"
)
CLEAN_TEXT = "Das ist nicht ganz richtig. Was denkst du, woher das Modell seine Vorhersage nimmt?"


class TestLooksLikeLeakedPrompt:
    def test_detects_field_label(self):
        assert _looks_like_leaked_prompt("Text mit ZIELKONZEPT: Overfitting drin") is True

    def test_detects_role_header(self):
        assert _looks_like_leaked_prompt("# ROLLE #\nDu bist ein Tutor...") is True

    def test_clean_tutor_reply_is_not_flagged(self):
        assert _looks_like_leaked_prompt(CLEAN_TEXT) is False

    def test_full_leaked_example_is_flagged(self):
        assert _looks_like_leaked_prompt(LEAKED_TEXT) is True


class TestPeekLeakGuard:
    def test_short_clean_response_flushes_on_finalize(self):
        forwarded = []
        guard = _PeekLeakGuard(forwarded.append)
        guard.feed("Kurze ")
        guard.feed("Antwort.")
        assert forwarded == []  # still buffered — under threshold, not finalized yet
        guard.finalize_clean()
        assert forwarded == ["Kurze Antwort."]
        assert guard.leaked is False

    def test_long_clean_response_streams_live_after_threshold(self):
        forwarded = []
        guard = _PeekLeakGuard(forwarded.append)
        long_clean = "Das ist eine ausführliche, aber völlig unauffällige Tutor-Antwort ohne jede Markierung. " * 3
        for chunk in [long_clean[i : i + 10] for i in range(0, len(long_clean), 10)]:
            guard.feed(chunk)
        assert guard.leaked is False
        assert "".join(forwarded) == long_clean  # nothing dropped, just re-chunked around the threshold
        assert len(forwarded) > 1  # buffered prefix once, then individual chunks live
        # tail tokens fed after commit are forwarded immediately, one by one
        guard.feed(" Noch mehr Text.")
        assert forwarded[-1] == " Noch mehr Text."

    def test_leak_in_prefix_is_never_forwarded(self):
        forwarded = []
        guard = _PeekLeakGuard(forwarded.append)
        for chunk in ["# ROLLE #\n", "Du bist ein ", "ZIELKONZEPT: X"]:
            guard.feed(chunk)
        assert guard.leaked is True
        assert forwarded == []
        # further tokens after detection are dropped, not forwarded
        guard.feed("noch mehr")
        assert forwarded == []

    def test_finalize_clean_does_nothing_after_leak(self):
        forwarded = []
        guard = _PeekLeakGuard(forwarded.append)
        guard.feed("ZIELKONZEPT: X SITZUNGS-LERNZIEL: Y")
        guard.finalize_clean()
        assert forwarded == []


class TestGenerateValidated:
    """No streaming context is active in these tests (token_callback_var is
    unset), so _generate() always takes the buffered path — this exercises
    the retry/fallback ladder directly against mocked LLM responses."""

    def test_clean_first_attempt_returned_as_is(self):
        with patch("src.llm.tools.socratic_v2_core._llm.chat", return_value=_mock_llm_response(CLEAN_TEXT)):
            result = _generate_validated("query", [], Models.GEMMA4_31B)
        assert result == CLEAN_TEXT

    def test_leaked_first_attempt_retries_with_azure_fallback(self):
        responses = [_mock_llm_response(LEAKED_TEXT), _mock_llm_response(CLEAN_TEXT)]
        with patch("src.llm.tools.socratic_v2_core._llm.chat", side_effect=responses) as mock_chat:
            result = _generate_validated("query", [], Models.GEMMA4_31B)
        assert result == CLEAN_TEXT
        assert mock_chat.call_args_list[1].kwargs["model"] == Models.AZURE_FALLBACK

    def test_leaked_on_both_attempts_falls_back_to_static_message(self):
        responses = [_mock_llm_response(LEAKED_TEXT), _mock_llm_response(LEAKED_TEXT)]
        with patch("src.llm.tools.socratic_v2_core._llm.chat", side_effect=responses):
            result = _generate_validated("query", [], Models.GEMMA4_31B)
        assert result == GENERATION_FALLBACK

    def test_none_content_falls_back_without_retry(self):
        with patch("src.llm.tools.socratic_v2_core._llm.chat", return_value=_mock_llm_response(None)) as mock_chat:
            result = _generate_validated("query", [], Models.GEMMA4_31B)
        assert result == GENERATION_FALLBACK
        mock_chat.assert_called_once()  # None content is not a leak — no retry needed


class TestCoreNeverLeaksToUser:
    """End-to-end through socratic_v2_core: even if the generation LLM leaks
    on the first attempt, the answer that reaches the user is always clean."""

    def test_frage_move_answer_is_never_a_leak(self):
        state = _base_state()
        responses = [_mock_llm_response(LEAKED_TEXT), _mock_llm_response(CLEAN_TEXT)]
        with patch("src.llm.tools.socratic_v2_core.run_policy", return_value=_policy("FRAGE")), patch(
            "src.llm.tools.socratic_v2_core._llm.chat", side_effect=responses
        ):
            result = socratic_v2_core(state)
        assert result["answer"] == CLEAN_TEXT
        assert _looks_like_leaked_prompt(result["answer"]) is False


# ---------------------------------------------------------------------------
# Consolidation node
# ---------------------------------------------------------------------------


class TestSocraticV2Consolidation:
    def _run(self, state, llm_content="Starke Zusammenfassung! **Sicher:** Overfitting."):
        with patch(
            "src.llm.tools.socratic_v2_consolidation._llm.chat", return_value=_mock_llm_response(llm_content)
        ) as mock_chat:
            return socratic_v2_consolidation(state), mock_chat

    def test_resets_all_v2_state(self):
        result, _ = self._run(_base_state(socratic_v2_phase="consolidation"))
        # v2_last_move survives the reset on purpose: it marks that the
        # session ended via consolidation (vs. EXIT) for analysis/benchmark.
        expected = {**reset_socratic_v2_state(), "v2_last_move": "CONSOLIDATION"}
        for key, value in expected.items():
            assert result[key] == value

    def test_answer_from_llm(self):
        result, _ = self._run(_base_state(socratic_v2_phase="consolidation"))
        assert result["answer"] == "Starke Zusammenfassung! **Sicher:** Overfitting."

    def test_llm_failure_uses_fallback(self):
        result, _ = self._run(_base_state(socratic_v2_phase="consolidation"), llm_content=None)
        assert "abgeschlossen" in result["answer"]

    def test_learner_summary_passed_to_llm(self):
        state = _base_state(socratic_v2_phase="consolidation", user_query="Overfitting heißt auswendig lernen.")
        _, mock_chat = self._run(state)
        assert "Overfitting heißt auswendig lernen." in mock_chat.call_args.kwargs["query"]


# ---------------------------------------------------------------------------
# Routing (contextualize_and_route)
# ---------------------------------------------------------------------------


class TestSocraticV2Routing:
    def _contextualizer_mock(self):
        contextualizer = MagicMock()
        contextualizer.contextualize_socratic.return_value = "kontextualisierte Frage"
        contextualizer.classify_scenario.return_value = "no_vectordb"
        return contextualizer

    def test_entry_via_flag(self):
        state = _base_state(socratic_v2_phase=None)
        state["runtime_config"]["start_socratic_v2"] = True
        with patch("src.llm.tools.contextualize.get_contextualizer", return_value=self._contextualizer_mock()):
            result = contextualize_and_route(state)
        assert result["mode"] == "socratic_v2"
        assert result["socratic_v2_phase"] == "opening"

    def test_entry_ignored_when_disabled(self):
        state = _base_state(socratic_v2_phase=None)
        state["system_config"]["enable_socratic_v2"] = False
        state["runtime_config"]["start_socratic_v2"] = True
        with patch("src.llm.tools.contextualize.get_contextualizer", return_value=self._contextualizer_mock()):
            result = contextualize_and_route(state)
        assert result["mode"] != "socratic_v2"

    def test_active_core_session_routes_to_socratic_v2_with_contextualization(self):
        state = _base_state(socratic_v2_phase="core")
        contextualizer = self._contextualizer_mock()
        with patch("src.llm.tools.contextualize.get_contextualizer", return_value=contextualizer):
            result = contextualize_and_route(state)
        assert result["mode"] == "socratic_v2"
        assert result["contextualized_query"] == "kontextualisierte Frage"
        contextualizer.contextualize_socratic.assert_called_once()

    def test_active_consolidation_session_skips_contextualization(self):
        state = _base_state(socratic_v2_phase="consolidation")
        contextualizer = self._contextualizer_mock()
        with patch("src.llm.tools.contextualize.get_contextualizer", return_value=contextualizer):
            result = contextualize_and_route(state)
        assert result["mode"] == "socratic_v2"
        assert result["contextualized_query"] is None
        contextualizer.contextualize_socratic.assert_not_called()

    def test_exit_keyword_resets_and_completes(self):
        state = _base_state(socratic_v2_phase="core", user_query="stopp")
        with patch("src.llm.tools.contextualize.get_contextualizer", return_value=self._contextualizer_mock()):
            result = contextualize_and_route(state)
        assert result["mode"] == "exit_complete"
        assert result["socratic_v2_phase"] is None
        assert result["answer"] == V2_EXIT_MESSAGE

    def test_active_v1_session_takes_precedence(self):
        state = _base_state(socratic_v2_phase=None, socratic_mode="contract")
        state["system_config"]["enable_socratic"] = True
        state["runtime_config"]["start_socratic_v2"] = True
        with patch("src.llm.tools.contextualize.get_contextualizer", return_value=self._contextualizer_mock()):
            result = contextualize_and_route(state)
        assert result["mode"] == "socratic"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def test_build_socratic_v2_graph_compiles():
    from src.llm.graphs.socratic_v2 import build_socratic_v2_graph

    graph = build_socratic_v2_graph()
    assert graph is not None
