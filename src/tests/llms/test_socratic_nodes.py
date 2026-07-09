"""Unit tests for the Socratic workflow nodes.

Covers:
- socratic_contract: Pure state initialization, no LLM call
- socratic_diagnose: LLM-driven diagnostic question + learning objective extraction
- generate_hint_text: Graduated hint generation
- generate_reflection_text: Pure congratulatory message (no LLM)
- socratic_explain: LLM-driven explanation generation
- socratic_core: Core dialogue loop with 4 mode branches (CONTINUE, HINT, REFLECT, EXPLAIN)
"""

from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.llms import MessageRole

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.api.models.serializable_text_node import SerializableTextNode
from src.llm.objects.LLMs import Models
from src.llm.tools.socratic_contract import socratic_contract
from src.llm.tools.socratic_diagnose import socratic_diagnose
from src.llm.tools.socratic_hinting import generate_hint_text
from src.llm.tools.socratic_reflection import generate_reflection_text
from src.llm.tools.socratic_explain import socratic_explain
from src.llm.tools.socratic_core import socratic_core


def _base_state(**overrides) -> dict:
    state = {
        "user_query": "Was ist Machine Learning?",
        "chat_history": [],
        "runtime_config": {
            "model": Models.AZURE_FALLBACK,
            "course_id": None,
            "module_id": None,
            "thread_id": "test-socratic",
        },
        "system_config": {"rerank_top_n": 3, "retrieve_top_n": 10},
        "socratic_mode": "contract",
        "attempt_count": 0,
        "number_given_hints": 0,
        "learning_objective": "Verstehe Machine Learning Grundkonzepte",
        "reranked": [],
    }
    state.update(overrides)
    return state


def _mock_llm_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    return msg


def _make_chunk(text: str) -> SerializableTextNode:
    return SerializableTextNode(text=text, metadata={})


class TestSocraticContract:
    def test_sets_socratic_mode_to_diagnose(self):
        result = socratic_contract(_base_state())
        assert result["socratic_mode"] == "diagnose"

    def test_initializes_counters_to_zero(self):
        result = socratic_contract(_base_state())
        assert result["hint_level"] == 0
        assert result["attempt_count"] == 0
        assert result["number_given_hints"] == 0

    def test_sets_goal_achieved_false(self):
        result = socratic_contract(_base_state())
        assert result["goal_achieved"] is False

    def test_answer_is_welcome_message(self):
        result = socratic_contract(_base_state())
        assert result["answer"] is not None
        assert len(result["answer"]) > 20
        assert "Lernmodus" in result["answer"]

    def test_clears_citations(self):
        result = socratic_contract(_base_state())
        assert result["citations_markdown"] is None

    def test_contract_disables_explain_and_hint_by_default(self):
        result = socratic_contract(_base_state())
        contract = result["socratic_contract"]
        assert contract["allow_explain"] is False
        assert contract["allow_hint"] is False


class TestSocraticDiagnose:
    def _call_with_mock(self, llm_content: str) -> dict:
        state = _base_state()
        with patch("src.llm.tools.socratic_diagnose._llm.chat", return_value=_mock_llm_response(llm_content)):
            return socratic_diagnose(state)

    def test_sets_socratic_mode_to_core(self):
        result = self._call_with_mock("LEARNING_OBJECTIVE: ML Grundlagen\nDIAGNOSTIC_QUESTION: Was weißt du?")
        assert result["socratic_mode"] == "core"

    def test_extracts_learning_objective(self):
        result = self._call_with_mock("LEARNING_OBJECTIVE: Neuronale Netze\nDIAGNOSTIC_QUESTION: Was ist ein Neuron?")
        assert result["learning_objective"] == "Neuronale Netze"

    def test_extracts_diagnostic_question_as_answer(self):
        result = self._call_with_mock("LEARNING_OBJECTIVE: ML Grundlagen\nDIAGNOSTIC_QUESTION: Was weißt du bereits?")
        assert result["answer"] == "Was weißt du bereits?"

    def test_fallback_on_missing_learning_objective(self):
        result = self._call_with_mock("Ich bin ein LLM ohne Struktur")
        assert "Was ist Machine Learning" in result["learning_objective"]

    def test_fallback_on_missing_diagnostic_question(self):
        result = self._call_with_mock("LEARNING_OBJECTIVE: ML")
        assert result["answer"] == "Was weißt du bereits über dieses Thema?"

    def test_initializes_student_model(self):
        result = self._call_with_mock("LEARNING_OBJECTIVE: ML\nDIAGNOSTIC_QUESTION: Test?")
        assert "mastery" in result["student_model"]
        assert result["student_model"]["mastery"] == "unknown"


class TestGenerateHintText:
    def test_returns_hint_string(self):
        mock_resp = _mock_llm_response("💡 Denke an Gradient Descent!")
        with patch("src.llm.tools.socratic_hinting.LLM") as MockLLM:
            MockLLM.return_value.chat.return_value = mock_resp
            result = generate_hint_text(
                learning_objective="ML verstehen",
                number_given_hints=1,
                user_query="Ich verstehe das nicht",
                reranked_chunks=[],
                chat_history=[],
                model=Models.AZURE_FALLBACK,
            )
        assert result == "💡 Denke an Gradient Descent!"

    def test_fallback_on_none_content(self):
        mock_resp = _mock_llm_response(None)
        mock_resp.content = None
        with patch("src.llm.tools.socratic_hinting.LLM") as MockLLM:
            MockLLM.return_value.chat.return_value = mock_resp
            result = generate_hint_text(
                learning_objective="ML",
                number_given_hints=1,
                user_query="?",
                reranked_chunks=[],
                chat_history=[],
                model=Models.AZURE_FALLBACK,
            )
        assert "Hinweis" in result

    def test_uses_reranked_chunks_in_prompt(self):
        mock_resp = _mock_llm_response("Hinweis basierend auf Materialien")
        chunks = [_make_chunk("Kapitel 1: Grundlagen"), _make_chunk("Kapitel 2: Training")]
        with patch("src.llm.tools.socratic_hinting.LLM") as MockLLM:
            MockLLM.return_value.chat.return_value = mock_resp
            generate_hint_text(
                learning_objective="ML",
                number_given_hints=1,
                user_query="Test",
                reranked_chunks=chunks,
                chat_history=[],
                model=Models.AZURE_FALLBACK,
            )
            call_args = MockLLM.return_value.chat.call_args
        assert "Kapitel 1: Grundlagen" in call_args.kwargs["query"]


class TestGenerateReflectionText:
    def test_returns_non_empty_string(self):
        result = generate_reflection_text(learning_objective="ML Grundlagen verstehen")
        assert isinstance(result, str)
        assert len(result) > 20

    def test_contains_positive_reinforcement(self):
        result = generate_reflection_text(learning_objective="ML Grundlagen verstehen")
        assert "Ausgezeichnet" in result or "Lernmodus" in result

    def test_mentions_learning_objective(self):
        result = generate_reflection_text(learning_objective="ML Grundlagen verstehen")
        assert "ML Grundlagen verstehen" in result

    def test_is_deterministic(self):
        assert generate_reflection_text(learning_objective="X") == generate_reflection_text(learning_objective="X")


class TestSocraticExplain:
    def test_returns_explanation_string(self):
        mock_resp = _mock_llm_response("Machine Learning ist ein Teilgebiet der KI.")
        with patch("src.llm.tools.socratic_explain.llm.chat", return_value=mock_resp):
            result = socratic_explain(
                learning_objective="ML verstehen",
                user_query="Ich gebe auf",
                reranked_chunks=[],
                chat_history=[],
                number_given_hints=2,
                attempt_count=3,
                model=Models.AZURE_FALLBACK,
            )
        assert "Machine Learning ist ein Teilgebiet der KI." in result

    def test_appends_closing_text(self):
        mock_resp = _mock_llm_response("Erklärung hier.")
        with patch("src.llm.tools.socratic_explain.llm.chat", return_value=mock_resp):
            result = socratic_explain(
                learning_objective="ML",
                user_query="Test",
                reranked_chunks=[],
                chat_history=[],
                number_given_hints=0,
                attempt_count=1,
                model=Models.AZURE_FALLBACK,
            )
        assert "Lernmodus" in result or "quit" in result

    def test_fallback_on_none_content(self):
        mock_resp = MagicMock()
        mock_resp.content = None
        with patch("src.llm.tools.socratic_explain.llm.chat", return_value=mock_resp):
            result = socratic_explain(
                learning_objective="Neuronale Netze",
                user_query="?",
                reranked_chunks=[],
                chat_history=[],
                number_given_hints=0,
                attempt_count=1,
                model=Models.AZURE_FALLBACK,
            )
        assert "Neuronale Netze" in result


class TestSocraticCore:
    def _call_core(self, state: dict, evaluate_mode: str, question_text: str = "Nächste Frage?") -> dict:
        """Helper that mocks the internal functions of socratic_core."""
        with (
            patch("src.llm.tools.socratic_core.evaluate_user_response", return_value=evaluate_mode),
            patch("src.llm.tools.socratic_core.generate_socratic_question", return_value=question_text),
            patch("src.llm.tools.socratic_core.generate_hint_text", return_value="Hier ist ein Hinweis"),
            patch("src.llm.tools.socratic_core.generate_reflection_text", return_value="Sehr gut!"),
            patch("src.llm.tools.socratic_core.socratic_explain", return_value="Die Antwort ist 42"),
        ):
            return socratic_core(state)

    def test_continue_mode_stays_in_core(self):
        state = _base_state(attempt_count=0, number_given_hints=0)
        result = self._call_core(state, "CONTINUE")
        assert result["socratic_mode"] == "core"

    def test_continue_mode_increments_attempt_count(self):
        state = _base_state(attempt_count=2, number_given_hints=0)
        result = self._call_core(state, "CONTINUE")
        assert result["attempt_count"] == 3

    def test_continue_mode_sets_answer(self):
        state = _base_state(attempt_count=0, number_given_hints=0)
        result = self._call_core(state, "CONTINUE", question_text="Kannst du das erklären?")
        assert result["answer"] == "Kannst du das erklären?"

    def test_hint_mode_increments_hint_count(self):
        state = _base_state(attempt_count=1, number_given_hints=0)
        result = self._call_core(state, "HINT")
        assert result["number_given_hints"] == 1

    def test_hint_mode_stays_in_core(self):
        state = _base_state(attempt_count=1, number_given_hints=0)
        result = self._call_core(state, "HINT")
        assert result["socratic_mode"] == "core"

    def test_hint_mode_sets_hint_answer(self):
        state = _base_state(attempt_count=1, number_given_hints=0)
        result = self._call_core(state, "HINT")
        assert result["answer"] == "Hier ist ein Hinweis"

    def test_reflect_mode_resets_to_diagnose(self):
        state = _base_state(attempt_count=3, number_given_hints=1)
        result = self._call_core(state, "REFLECT")
        assert result["socratic_mode"] == "diagnose"

    def test_reflect_mode_resets_counters(self):
        state = _base_state(attempt_count=3, number_given_hints=1)
        result = self._call_core(state, "REFLECT")
        assert result["attempt_count"] == 0
        assert result["number_given_hints"] == 0

    def test_explain_auto_triggered_after_two_hints(self):
        state = _base_state(attempt_count=2, number_given_hints=2)
        with (
            patch("src.llm.tools.socratic_core.generate_socratic_question") as mock_q,
            patch("src.llm.tools.socratic_core.generate_hint_text") as mock_hint,
            patch("src.llm.tools.socratic_core.generate_reflection_text") as mock_reflect,
            patch("src.llm.tools.socratic_core.socratic_explain", return_value="Hier ist die Erklärung"),
        ):
            result = socratic_core(state)
        mock_q.assert_not_called()
        mock_hint.assert_not_called()
        mock_reflect.assert_not_called()
        assert result["answer"] == "Hier ist die Erklärung"
        assert result["socratic_mode"] == "diagnose"
