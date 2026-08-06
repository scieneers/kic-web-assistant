"""
Unit tests for the No-Answer-Logik in QuestionAnswerer.

Tests the three-layer logic:
1. LLM returns "NO ANSWER FOUND" (first time) → polite clarification request
2. LLM returns "NO ANSWER FOUND" (second time in a row, Drupal) → support email
3. LLM returns "NO ANSWER FOUND" (second time in a row, Moodle) → course link
4. Normal LLM response → passed through unchanged
5. EmptyModule marker in reranked sources → fixed fallback without LLM call
"""

from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.llms import MessageRole
from llama_index.core.schema import TextNode

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.api.models.serializable_text_node import SerializableTextNode
from src.llm.objects.LLMs import Models
from src.llm.objects.question_answerer import (
    ANSWER_NOT_FOUND_FIRST_TIME,
    ANSWER_NOT_FOUND_SECOND_TIME_DRUPAL,
    NO_CONTENT_IN_MODULE,
    NO_CONTENT_UNSUPPORTED_TYPE,
    get_fallback_type,
    QuestionAnswerer,
)
from src.llm.tools.answer import generate_answer


def make_message(role: str, content: str) -> SerializableChatMessage:
    return SerializableChatMessage(role=role, content=content)


def make_sources() -> list[TextNode]:
    return [TextNode(text="Dummy source content")]


def call_answer(answerer, llm_response: str, chat_history=None, is_moodle=False, course_id=None):
    """Helper that calls answer_question with a mocked LLM response."""
    mock_response = MagicMock()
    mock_response.content = llm_response

    with patch.object(answerer.llm, "chat", return_value=mock_response):
        return answerer.answer_question(
            query="Testfrage",
            chat_history=chat_history or [],
            sources=make_sources(),
            model=Models.AZURE_FALLBACK,
            language="German",
            is_moodle=is_moodle,
            course_id=course_id,
        )


@pytest.fixture
def answerer():
    with patch("src.llm.objects.question_answerer.load_prompt", return_value="system prompt {language}"):
        qa = QuestionAnswerer()
    return qa


class TestFirstNoAnswer:
    def test_empty_history_returns_first_time_message(self, answerer):
        result = call_answer(answerer, "NO ANSWER FOUND", chat_history=[])
        assert result.content == ANSWER_NOT_FOUND_FIRST_TIME

    def test_history_without_no_answer_returns_first_time_message(self, answerer):
        history = [
            make_message(MessageRole.USER, "Hallo"),
            make_message(MessageRole.ASSISTANT, "Willkommen!"),
        ]
        result = call_answer(answerer, "NO ANSWER FOUND", chat_history=history)
        assert result.content == ANSWER_NOT_FOUND_FIRST_TIME


class TestSecondNoAnswer:
    def test_drupal_second_no_answer_returns_support_email(self, answerer):
        history = [
            make_message(MessageRole.USER, "Ich verstehe nicht"),
            make_message(MessageRole.ASSISTANT, ANSWER_NOT_FOUND_FIRST_TIME),
        ]
        result = call_answer(answerer, "NO ANSWER FOUND", chat_history=history, is_moodle=False)
        assert result.content == ANSWER_NOT_FOUND_SECOND_TIME_DRUPAL

    def test_moodle_second_no_answer_returns_course_link_with_id(self, answerer):
        history = [
            make_message(MessageRole.USER, "Ich verstehe nicht"),
            make_message(MessageRole.ASSISTANT, ANSWER_NOT_FOUND_FIRST_TIME),
        ]
        result = call_answer(answerer, "NO ANSWER FOUND", chat_history=history, is_moodle=True, course_id=42)
        assert "42" in result.content
        assert "moodle.ki-campus.org" in result.content

    def test_moodle_second_no_answer_without_course_id_uses_unknown(self, answerer):
        history = [
            make_message(MessageRole.USER, "Ich verstehe nicht"),
            make_message(MessageRole.ASSISTANT, ANSWER_NOT_FOUND_FIRST_TIME),
        ]
        result = call_answer(answerer, "NO ANSWER FOUND", chat_history=history, is_moodle=True, course_id=None)
        assert "UNKNOWN" in result.content

    def test_only_last_assistant_message_is_checked(self, answerer):
        """A no-answer buried in earlier history does not count as 'previous'."""
        history = [
            make_message(MessageRole.ASSISTANT, ANSWER_NOT_FOUND_FIRST_TIME),
            make_message(MessageRole.USER, "Noch eine Frage"),
            make_message(MessageRole.ASSISTANT, "Eine normale Antwort"),
        ]
        result = call_answer(answerer, "NO ANSWER FOUND", chat_history=history, is_moodle=False)
        assert result.content == ANSWER_NOT_FOUND_FIRST_TIME


class TestNormalAnswer:
    def test_normal_response_passed_through_unchanged(self, answerer):
        result = call_answer(answerer, "Deep Learning ist ein Teilgebiet des maschinellen Lernens.")
        assert result.content == "Deep Learning ist ein Teilgebiet des maschinellen Lernens."

    def test_normal_response_with_no_answer_history_still_passes_through(self, answerer):
        history = [
            make_message(MessageRole.ASSISTANT, ANSWER_NOT_FOUND_FIRST_TIME),
        ]
        result = call_answer(answerer, "Eine echte Antwort diesmal.", chat_history=history)
        assert result.content == "Eine echte Antwort diesmal."


class TestEdgeCases:
    def test_history_ending_with_user_message_still_detects_second_no_answer(self, answerer):
        """The reversed loop must skip USER messages and still find the previous ASSISTANT turn."""
        history = [
            make_message(MessageRole.USER, "Erste Frage"),
            make_message(MessageRole.ASSISTANT, ANSWER_NOT_FOUND_FIRST_TIME),
            make_message(MessageRole.USER, "Nochmal die selbe Frage"),
        ]
        result = call_answer(answerer, "NO ANSWER FOUND", chat_history=history, is_moodle=False)
        assert result.content == ANSWER_NOT_FOUND_SECOND_TIME_DRUPAL

    def test_history_with_only_user_messages_returns_first_time(self, answerer):
        """No ASSISTANT message in history → can never be a second no-answer."""
        history = [
            make_message(MessageRole.USER, "Hallo"),
            make_message(MessageRole.USER, "Ich habe eine Frage"),
        ]
        result = call_answer(answerer, "NO ANSWER FOUND", chat_history=history)
        assert result.content == ANSWER_NOT_FOUND_FIRST_TIME

    def test_moodle_second_no_answer_trailing_user_message_uses_course_link(self, answerer):
        """Moodle path: last ASSISTANT was FIRST_TIME, followed by a USER message."""
        history = [
            make_message(MessageRole.USER, "Was ist KI?"),
            make_message(MessageRole.ASSISTANT, ANSWER_NOT_FOUND_FIRST_TIME),
            make_message(MessageRole.USER, "Kannst du es nochmal versuchen?"),
        ]
        result = call_answer(answerer, "NO ANSWER FOUND", chat_history=history, is_moodle=True, course_id=99)
        assert "99" in result.content
        assert "moodle.ki-campus.org" in result.content


# ---------------------------------------------------------------------------
# EmptyModule marker — handled in generate_answer() before QuestionAnswerer
# ---------------------------------------------------------------------------

def make_state(reranked: list, course_id=None) -> dict:
    return {
        "reranked": reranked,
        "retrieved": [],
        "user_query": "Was ist in diesem Modul?",
        "chat_history": [],
        "detected_language": "German",
        "contextualized_query": None,
        "runtime_config": {
            "model": Models.AZURE_FALLBACK,
            "course_id": course_id,
            "module_id": None,
            "thread_id": "test-thread",
        },
        "system_config": {
            "rerank_top_n": 5,
            "retrieve_top_n": 10,
            "enable_socratic": False,
            "reranker_type": "llm",
            "min_reranker_score": 0.0,
        },
    }


def make_empty_module_node(fullname: str = "KI-Grundlagen") -> SerializableTextNode:
    return SerializableTextNode(
        text="",
        metadata={"type": "EmptyModule", "fullname": fullname},
    )


class TestEmptyModule:
    def test_single_empty_module_returns_fixed_message(self):
        state = make_state([make_empty_module_node("KI-Grundlagen")], course_id=42)
        with patch("src.llm.tools.answer.get_question_answerer"):
            result = generate_answer(state)
        assert "KI-Grundlagen" in result["answer"]
        assert result["answer"] == NO_CONTENT_IN_MODULE.format(module_name="KI-Grundlagen")

    def test_multiple_empty_module_nodes_uses_first_name(self):
        state = make_state(
            [make_empty_module_node("Modul A"), make_empty_module_node("Modul B")],
            course_id=1,
        )
        with patch("src.llm.tools.answer.get_question_answerer"):
            result = generate_answer(state)
        assert "Modul A" in result["answer"]

    def test_empty_module_without_fullname_uses_default(self):
        node = SerializableTextNode(text="", metadata={"type": "EmptyModule"})
        state = make_state([node], course_id=5)
        with patch("src.llm.tools.answer.get_question_answerer"):
            result = generate_answer(state)
        assert "Dieses Modul" in result["answer"]

    def test_mixed_sources_do_not_trigger_empty_module_path(self):
        """One real chunk alongside an EmptyModule marker → normal LLM path."""
        real_node = SerializableTextNode(text="KI steht für Künstliche Intelligenz.", metadata={})
        state = make_state([make_empty_module_node("Modul X"), real_node], course_id=7)
        mock_response = MagicMock()
        mock_response.content = "Eine echte Antwort."
        mock_answerer = MagicMock()
        mock_answerer.answer_question.return_value = mock_response
        with patch("src.llm.tools.answer.get_question_answerer", return_value=mock_answerer):
            result = generate_answer(state)
        assert result["answer"] == "Eine echte Antwort."
        mock_answerer.answer_question.assert_called_once()

    def test_unsupported_content_type_names_the_reason(self):
        """A known-unsupported modname (e.g. native Moodle quiz) gets a specific
        message naming why, instead of the generic 'kein Inhalt' fallback."""
        node = SerializableTextNode(
            text="",
            metadata={
                "type": "EmptyModule",
                "fullname": "Lernziel-Check II",
                "unsupported_label": "natives Moodle-Quiz",
                "url": "https://moodle.ki-campus.org/mod/quiz/view.php?id=1406",
            },
        )
        state = make_state([node], course_id=36)
        with patch("src.llm.tools.answer.get_question_answerer"):
            result = generate_answer(state)
        assert result["answer"] == NO_CONTENT_UNSUPPORTED_TYPE.format(
            module_name="Lernziel-Check II",
            label="natives Moodle-Quiz",
            url="https://moodle.ki-campus.org/mod/quiz/view.php?id=1406",
        )
        assert "natives Moodle-Quiz" in result["answer"]
        assert "kein Inhalt" not in result["answer"] and "keinen weiteren Inhalt" not in result["answer"]

    def test_empty_reranked_list_does_not_trigger_empty_module_path(self):
        """Empty list has no sources at all — EmptyModule check requires sources to be non-empty."""
        state = make_state([], course_id=3)
        mock_response = MagicMock()
        mock_response.content = ANSWER_NOT_FOUND_FIRST_TIME
        mock_answerer = MagicMock()
        mock_answerer.answer_question.return_value = mock_response
        with patch("src.llm.tools.answer.get_question_answerer", return_value=mock_answerer):
            result = generate_answer(state)
        assert NO_CONTENT_IN_MODULE.format(module_name="") not in result["answer"]


class TestGetFallbackType:
    def test_classifies_empty_module(self):
        text = NO_CONTENT_IN_MODULE.format(module_name="KI-Grundlagen")
        assert get_fallback_type(text) == "empty_module"

    def test_classifies_unsupported_content_type(self):
        text = NO_CONTENT_UNSUPPORTED_TYPE.format(
            module_name="Lernziel-Check II", label="natives Moodle-Quiz", url="https://x/1406"
        )
        assert get_fallback_type(text) == "unsupported_content_type"

    def test_real_answer_is_not_a_fallback(self):
        assert get_fallback_type("Overfitting bedeutet, dass ein Modell auswendig lernt.") is None
