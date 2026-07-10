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
    initial_learner_model,
    merge_learner_update,
    parse_policy_response,
    reset_socratic_v2_state,
)
from src.llm.tools.contextualize import contextualize_and_route
from src.llm.tools.socratic_v2_consolidation import socratic_v2_consolidation
from src.llm.tools.socratic_v2_core import socratic_v2_core
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
            '"learner_update": {"konzepte": {"Overfitting": "wackelig"}}}'
        )
        assert result["move"] == "HINT"
        assert result["zielkonzept"] == "Overfitting"
        assert result["lernziel_session"] == "Ziel"
        assert result["learner_update"]["konzepte"]["Overfitting"] == "wackelig"

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
    def _run(self, state, nodes, llm_content="LERNZIEL: Overfitting erkennen\nKONZEPT: Overfitting"):
        retriever = MagicMock()
        retriever.retrieve_all.return_value = nodes
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

    def test_extraction_failure_still_starts_session(self):
        result, _ = self._run(_base_state(), [_make_chunk("Inhalt", {"fullname": "ML Modul"})], llm_content="kaputt")
        assert result["socratic_v2_phase"] == "core"
        assert len(result["v2_learning_objectives"]) == 1  # fallback objective

    def test_module_list_targets_last_module(self):
        state = _base_state()
        state["runtime_config"]["module_id"] = [3, 5, 9]
        _, retriever = self._run(state, [_make_chunk("Inhalt")])
        retriever.retrieve_all.assert_called_once_with(course_id=42, module_id=9)


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
        result, _, _ = self._run(_base_state(), _policy("CONSOLIDATE"))
        assert result["socratic_v2_phase"] == "consolidation"

    def test_exit_resets_all_v2_state(self):
        result, _, _ = self._run(_base_state(), _policy("EXIT"))
        assert result["socratic_v2_phase"] is None
        assert result["v2_learner_model"] is None
        assert result["answer"] == V2_EXIT_MESSAGE

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
        for key, value in reset_socratic_v2_state().items():
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
