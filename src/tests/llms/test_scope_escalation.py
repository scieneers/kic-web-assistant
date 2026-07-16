"""Unit tests for the module→course scope escalation.

Flow under test:
1. Module-scoped question finds no answer → generate_answer returns the
   NO_ANSWER_IN_MODULE_OFFER_COURSE fallback and sets pending_scope_escalation.
2. Next turn "Ja" → contextualize_and_route deterministically re-enters
   simple_hop with the stored query and scope_escalated=True (no LLM call).
3. retrieve_chunks drops the module filter but keeps the course filter.
"""

from unittest.mock import MagicMock, patch

import pytest

import src.llm.tools.contextualize as contextualize_module
import src.llm.tools.retrieve as retrieve_module
from src.llm.objects.LLMs import Models
from src.llm.objects.question_answerer import (
    ANSWER_NOT_FOUND_FIRST_TIME,
    NO_ANSWER_IN_MODULE_OFFER_COURSE,
    NO_RELEVANT_CONTENT,
    get_fallback_type,
    QuestionAnswerer,
)
from src.llm.tools.answer import generate_answer
from src.llm.tools.contextualize import contextualize_and_route
from src.llm.tools.retrieve import retrieve_chunks


def make_state(**overrides) -> dict:
    state = {
        "user_query": "Was ist eine Dienstvereinbarung?",
        "chat_history": [],
        "detected_language": "German",
        "contextualized_query": "Was ist eine Dienstvereinbarung?",
        "retrieved": [],
        "reranked": [],
        "runtime_config": {
            "model": Models.AZURE_FALLBACK,
            "course_id": 344,
            "module_id": 28678,
            "thread_id": "test-thread",
            "start_socratic": False,
            "start_socratic_v2": False,
        },
        "system_config": {
            "rerank_top_n": 5,
            "retrieve_top_n": 10,
            "enable_socratic": False,
            "enable_socratic_v2": False,
            "reranker_type": "llm",
            "min_reranker_score": 0.0,
        },
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# QuestionAnswerer: offer instead of dead-end fallback
# ---------------------------------------------------------------------------


@pytest.fixture
def answerer():
    with patch("src.llm.objects.question_answerer.load_prompt", return_value="system prompt {language}"):
        return QuestionAnswerer()


class TestOfferFallback:
    def test_no_sources_module_scope_offers_course_search(self, answerer):
        result = answerer.answer_question(
            query="Was ist eine Dienstvereinbarung?",
            chat_history=[],
            sources=[],
            model=Models.AZURE_FALLBACK,
            language="German",
            is_moodle=True,
            course_id=344,
            offer_course_escalation=True,
        )
        assert result.content == NO_ANSWER_IN_MODULE_OFFER_COURSE

    def test_no_sources_without_module_scope_keeps_generic_fallback(self, answerer):
        result = answerer.answer_question(
            query="Frage",
            chat_history=[],
            sources=[],
            model=Models.AZURE_FALLBACK,
            language="German",
            is_moodle=True,
            course_id=344,
        )
        assert result.content == NO_RELEVANT_CONTENT

    def test_llm_no_answer_module_scope_offers_course_search(self, answerer):
        mock_response = MagicMock()
        mock_response.content = "NO ANSWER FOUND"
        with patch.object(answerer.llm, "chat", return_value=mock_response):
            result = answerer.answer_question(
                query="Frage",
                chat_history=[],
                sources=[MagicMock(metadata={}, get_text=lambda: "content")],
                model=Models.AZURE_FALLBACK,
                language="German",
                is_moodle=True,
                course_id=344,
                offer_course_escalation=True,
            )
        assert result.content == NO_ANSWER_IN_MODULE_OFFER_COURSE

    def test_declined_offer_then_second_no_answer_uses_course_link(self, answerer):
        """After the offer was already given, the next no-answer escalates to the
        second-time Moodle fallback instead of offering in a loop."""
        from src.api.models.serializable_chat_message import SerializableChatMessage

        history = [
            SerializableChatMessage(role="user", content="Was ist eine Dienstvereinbarung?"),
            SerializableChatMessage(role="assistant", content=NO_ANSWER_IN_MODULE_OFFER_COURSE),
        ]
        result = answerer.answer_question(
            query="Andere Frage ohne Antwort",
            chat_history=history,
            sources=[],
            model=Models.AZURE_FALLBACK,
            language="German",
            is_moodle=True,
            course_id=344,
            offer_course_escalation=True,
        )
        assert "344" in result.content
        assert "moodle.ki-campus.org" in result.content

    def test_offer_is_classified_as_fallback(self):
        assert get_fallback_type(NO_ANSWER_IN_MODULE_OFFER_COURSE) == "no_answer_module_offer_course"


# ---------------------------------------------------------------------------
# generate_answer: sets pending_scope_escalation when the offer was made
# ---------------------------------------------------------------------------


class TestGenerateAnswerSetsPending:
    def _run(self, state, answer_content):
        mock_response = MagicMock()
        mock_response.content = answer_content
        mock_answerer = MagicMock()
        mock_answerer.answer_question.return_value = mock_response
        with patch("src.llm.tools.answer.get_question_answerer", return_value=mock_answerer):
            return generate_answer(state), mock_answerer

    def test_offer_fallback_sets_pending_with_query(self):
        state = make_state()
        result, _ = self._run(state, NO_ANSWER_IN_MODULE_OFFER_COURSE)
        assert result["pending_scope_escalation"] == {
            "contextualized_query": "Was ist eine Dienstvereinbarung?"
        }

    def test_real_answer_sets_no_pending(self):
        state = make_state()
        result, _ = self._run(state, "Eine echte Antwort.")
        assert "pending_scope_escalation" not in result

    def test_escalated_turn_does_not_offer_again_and_uses_stored_query(self):
        """On the escalated turn user_query is just 'Ja' — the answerer must get
        the stored question and must not offer escalation a second time."""
        state = make_state(user_query="Ja", scope_escalated=True)
        _, mock_answerer = self._run(state, ANSWER_NOT_FOUND_FIRST_TIME)
        kwargs = mock_answerer.answer_question.call_args.kwargs
        assert kwargs["query"] == "Was ist eine Dienstvereinbarung?"
        assert kwargs["offer_course_escalation"] is False

    def test_course_scope_without_module_never_offers(self):
        state = make_state()
        state["runtime_config"]["module_id"] = None
        _, mock_answerer = self._run(state, "Antwort")
        assert mock_answerer.answer_question.call_args.kwargs["offer_course_escalation"] is False


# ---------------------------------------------------------------------------
# contextualize_and_route: consuming the pending offer
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_contextualizer(monkeypatch):
    mock = MagicMock()
    mock.classify_scenario.return_value = "simple_hop"
    mock.contextualize.return_value = "kontextualisierte Frage"
    # Second-stage LLM consent check defaults to "no consent" — individual
    # tests flip it to exercise the free-form acceptance path.
    mock.classify_escalation_acceptance.return_value = False
    monkeypatch.setattr(contextualize_module, "_contextualizer_instance", mock)
    return mock


PENDING = {"contextualized_query": "Was ist eine Dienstvereinbarung?"}


class TestEscalationRouting:
    @pytest.mark.parametrize("reply", ["Ja", "ja gerne", "Ja bitte!", "ok", "Okay.", "JA"])
    def test_affirmative_keyword_escalates_without_any_llm_call(self, mock_contextualizer, reply):
        state = make_state(user_query=reply, pending_scope_escalation=PENDING)
        result = contextualize_and_route(state)
        assert result["mode"] == "simple_hop"
        assert result["scope_escalated"] is True
        assert result["contextualized_query"] == PENDING["contextualized_query"]
        assert result["pending_scope_escalation"] is None
        mock_contextualizer.classify_scenario.assert_not_called()
        mock_contextualizer.contextualize.assert_not_called()
        mock_contextualizer.classify_escalation_acceptance.assert_not_called()

    def test_free_form_consent_escalates_via_llm_check(self, mock_contextualizer):
        mock_contextualizer.classify_escalation_acceptance.return_value = True
        state = make_state(user_query="klar, mach mal", pending_scope_escalation=PENDING)
        result = contextualize_and_route(state)
        assert result["mode"] == "simple_hop"
        assert result["scope_escalated"] is True
        assert result["contextualized_query"] == PENDING["contextualized_query"]
        mock_contextualizer.classify_escalation_acceptance.assert_called_once()
        mock_contextualizer.classify_scenario.assert_not_called()

    def test_other_reply_clears_pending_and_routes_normally(self, mock_contextualizer):
        state = make_state(user_query="Was ist Overfitting?", pending_scope_escalation=PENDING)
        result = contextualize_and_route(state)
        assert result["mode"] == "simple_hop"
        assert result["pending_scope_escalation"] is None
        assert "scope_escalated" not in result
        mock_contextualizer.classify_escalation_acceptance.assert_called_once()
        mock_contextualizer.classify_scenario.assert_called_once()

    def test_no_pending_never_calls_acceptance_check(self, mock_contextualizer):
        state = make_state(user_query="Was ist Overfitting?")
        contextualize_and_route(state)
        mock_contextualizer.classify_escalation_acceptance.assert_not_called()

    def test_no_pending_affirmative_routes_normally(self, mock_contextualizer):
        mock_contextualizer.classify_scenario.return_value = "no_vectordb"
        state = make_state(user_query="Ja")
        result = contextualize_and_route(state)
        assert result["mode"] == "no_vectordb"

    def test_module_scope_passed_to_classifier(self, mock_contextualizer):
        state = make_state()
        contextualize_and_route(state)
        assert mock_contextualizer.classify_scenario.call_args.kwargs["scope"] == "module"

    def test_global_scope_passed_to_classifier(self, mock_contextualizer):
        state = make_state()
        state["runtime_config"]["course_id"] = None
        state["runtime_config"]["module_id"] = None
        contextualize_and_route(state)
        assert mock_contextualizer.classify_scenario.call_args.kwargs["scope"] == "global"

    def test_course_scope_with_empty_module_list(self, mock_contextualizer):
        state = make_state()
        state["runtime_config"]["module_id"] = []
        contextualize_and_route(state)
        assert mock_contextualizer.classify_scenario.call_args.kwargs["scope"] == "course"


# ---------------------------------------------------------------------------
# retrieve_chunks: escalated turn drops the module filter, keeps the course
# ---------------------------------------------------------------------------


class TestEscalatedRetrieve:
    def _run(self, state):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_retriever.use_semantic = False
        monkey = patch.object(retrieve_module, "_retriever_instance", mock_retriever)
        with monkey:
            retrieve_chunks(state)
        return mock_retriever

    def test_escalated_drops_module_filter_keeps_course(self):
        retriever = self._run(make_state(scope_escalated=True))
        kwargs = retriever.retrieve.call_args.kwargs
        assert kwargs["module_id"] is None
        assert kwargs["course_id"] == 344

    def test_not_escalated_keeps_module_filter(self):
        retriever = self._run(make_state(scope_escalated=False))
        assert retriever.retrieve.call_args.kwargs["module_id"] == 28678


# ---------------------------------------------------------------------------
# Integration: free-form consent classification with the real LLM
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_contextualizer():
    from src.llm.objects.contextualizer import Contextualizer

    return Contextualizer()


@pytest.mark.integration
@pytest.mark.parametrize("reply", [
    pytest.param("klar, mach mal", id="informal_consent"),
    pytest.param("ja bitte such im ganzen Kurs", id="elaborated_consent"),
    pytest.param("okay, warum nicht", id="casual_consent"),
    pytest.param("jup", id="colloquial_yes"),
])
def test_free_form_consent_accepted(real_contextualizer, reply):
    assert real_contextualizer.classify_escalation_acceptance(query=reply, model=Models.AZURE_FALLBACK) is True


@pytest.mark.integration
@pytest.mark.parametrize("reply", [
    pytest.param("nein danke", id="decline"),
    pytest.param("nö, lass mal", id="informal_decline"),
    pytest.param("Was ist Overfitting?", id="new_question"),
    pytest.param("Was ist denn eine Dienstvereinbarung?", id="rephrased_same_question"),
    pytest.param("asdfkjhgfds", id="gibberish"),
])
def test_non_consent_classified_as_other(real_contextualizer, reply):
    assert real_contextualizer.classify_escalation_acceptance(query=reply, model=Models.AZURE_FALLBACK) is False
