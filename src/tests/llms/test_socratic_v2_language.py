"""Tests for the Socratic Tutor fixes from the testing feedback (2026-11).

Covers:
- session language: resolution at opening, re-detection per turn, injection
  into generation/consolidation/extraction, localized fixed texts
- translation requests: prefilter, LLM confirmation, routing, 1:1 translate node
- quiz: multi-select grading, clarification instead of revealing, give-up,
  instruction line (single vs. multiple correct answers)
- learning-mode status for the frontend toggle
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.llms import MessageRole

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.llm.assistant import KICampusAssistant
from src.llm.objects.LLMs import Models
from src.llm.state.socratic_v2_language import (
    canonical_language,
    classify_language_request,
    localize,
    looks_like_language_request,
    normalize_language,
    resolve_opening_language,
)
from src.llm.state.socratic_v2_routing import (
    build_pending_quiz,
    grade_quiz_answer,
    initial_learner_model,
    is_quiz_give_up,
    render_quiz_message,
)
from src.llm.tools.contextualize import contextualize_and_route
from src.llm.tools.language import detect_language_socratic_v2
from src.llm.tools.socratic_v2_consolidation import socratic_v2_consolidation
from src.llm.tools.socratic_v2_core import socratic_v2_core
from src.llm.tools.socratic_v2_opening import socratic_v2_opening
from src.llm.tools.socratic_v2_translate import socratic_v2_translate


@pytest.fixture(autouse=True)
def detector():
    """Language detector stub — default "German", tests override detect.return_value."""
    mock = MagicMock()
    mock.detect.return_value = "German"
    # Shared entry point of the normal chat path and the tutor (opening + node).
    with patch("src.llm.tools.language.get_language_detector", return_value=mock):
        yield mock


def _msg(role: MessageRole, content: str) -> SerializableChatMessage:
    return SerializableChatMessage(role=role, content=content)


def _llm_response(content):
    response = MagicMock()
    response.content = content
    return response


def _state(**overrides) -> dict:
    state = {
        "user_query": "Was ist Overfitting?",
        "chat_history": [],
        "runtime_config": {
            "model": Models.AZURE_FALLBACK,
            "course_id": 42,
            "module_id": 7,
            "thread_id": "t",
            "start_socratic": False,
            "start_socratic_v2": False,
        },
        "system_config": {"enable_socratic": False, "enable_socratic_v2": True},
        "socratic_v2_phase": "core",
        "v2_learning_objectives": ["Overfitting erkennen"],
        "v2_session_goal": "Overfitting verstehen",
        "v2_learner_model": initial_learner_model(),
        "v2_target_concept": "Overfitting",
        "v2_hint_count": 0,
        "v2_question_streak": 0,
        "v2_core_turns": 0,
        "reranked": [],
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Language helpers
# ---------------------------------------------------------------------------


class TestNormalizeLanguage:
    @pytest.mark.parametrize(
        "value, expected",
        [("en", "English"), ("en-US", "English"), ("de_DE", "German"), ("Englisch", "English"), ("deutsch", "German")],
    )
    def test_known_hints(self, value, expected):
        assert normalize_language(value) == expected

    @pytest.mark.parametrize("value", [None, "", "xx", "klingonisch"])
    def test_unknown_hints(self, value):
        assert normalize_language(value) is None

    def test_canonical_keeps_unknown_detector_output(self):
        assert canonical_language("Swedish") == "Swedish"
        assert canonical_language(None) == "German"


class TestLanguageRequestPrefilter:
    @pytest.mark.parametrize(
        "text",
        [
            "Kannst du das auf Englisch übersetzen?",
            "auf englisch bitte",
            "Bitte übersetze das ins Englische",
            "translate that into German please",
            "Can we continue in English?",
        ],
    )
    def test_requests(self, text):
        assert looks_like_language_request(text)

    @pytest.mark.parametrize(
        "text",
        [
            "the order of the words does not affect BoW",
            "Das Modell übersetzt Texte",  # no language named, 3rd person verb
            "kannst du englisch?",  # no translate verb, no "auf/in <language>"
            "Ich habe einmal gelesen, dass maschinelle Übersetzung vom Deutschen ins Englische mit "
            "Encoder-Decoder-Modellen funktioniert, und zwar seit vielen Jahren",  # too long for a request
        ],
    )
    def test_non_requests(self, text):
        assert not looks_like_language_request(text)

    def test_classification_returns_target_language(self):
        with patch("src.llm.objects.LLMs.LLM.chat", return_value=_llm_response("English")):
            assert classify_language_request("auf englisch bitte", []) == "English"

    def test_classification_none(self):
        with patch("src.llm.objects.LLMs.LLM.chat", return_value=_llm_response("NONE")):
            assert classify_language_request("Übersetzung ins Englische", []) is None

    def test_classification_failure_is_not_a_request(self):
        with patch("src.llm.objects.LLMs.LLM.chat", side_effect=RuntimeError("down")):
            assert classify_language_request("auf englisch bitte", []) is None


class TestResolveOpeningLanguage:
    def test_fresh_chat_uses_ui_language(self, detector):
        assert resolve_opening_language("Starte den Lernmodus", [], ui_language="en-US", started_via_flag=True) == "English"
        detector.detect.assert_not_called()

    def test_fresh_chat_without_hint_defaults_to_german(self):
        assert resolve_opening_language("Starte den Lernmodus", [], started_via_flag=True) == "German"

    def test_flag_start_ignores_signal_text_and_uses_conversation(self, detector):
        detector.detect.return_value = "English"
        history = [
            _msg(MessageRole.USER, "What is a bag of words model?"),
            _msg(MessageRole.ASSISTANT, "A bag of words model ..."),
        ]
        language = resolve_opening_language("Starte den Lernmodus", history, ui_language="de", started_via_flag=True)
        assert language == "English"
        assert detector.detect.call_args.kwargs["query"] == "What is a bag of words model?"


class TestDetectLanguageNode:
    """Per-turn detection in the tutor = the normal chat path's logic."""

    def test_every_turn_uses_detector_even_for_short_replies(self, detector):
        # No word-count shortcut: the detector itself decides short messages
        # from the history (same as outside the tutor).
        detector.detect.return_value = "English"
        result = detect_language_socratic_v2(_state(user_query="b"))
        detector.detect.assert_called_once()
        assert result == {"detected_language": "English", "v2_language": "English"}

    def test_same_history_window_as_normal_chat(self, detector):
        from src.env import env

        history = [_msg(MessageRole.USER, f"Nachricht {i}") for i in range(30)]
        detect_language_socratic_v2(_state(chat_history=history))
        passed = detector.detect.call_args.kwargs["chat_history"]
        assert passed == history[-env.CHAT_HISTORY_LIMIT:]

    def test_detector_output_normalized(self, detector):
        detector.detect.return_value = "Englisch"
        assert detect_language_socratic_v2(_state())["v2_language"] == "English"


class TestLocalize:
    def test_german_and_english_are_authored(self):
        assert "Sokratischen Tutor" in localize("exit", "German")
        assert "Socratic Tutor" in localize("exit", "English")

    def test_other_language_is_translated_from_german(self):
        with patch("src.llm.objects.question_answerer.translate_fallback_text", return_value="FR") as mock_translate:
            assert localize("exit", "French") == "FR"
        assert mock_translate.call_args.args[1] == "French"


# ---------------------------------------------------------------------------
# Opening / core / consolidation use the session language
# ---------------------------------------------------------------------------


class TestOpeningLanguage:
    def _run(self, state):
        retriever = MagicMock()
        retriever.retrieve_all.return_value = [MagicMock(metadata={"fullname": "ML Modul"}, text="Inhalt")]
        retriever.retrieve_items.return_value = []
        with patch("src.llm.tools.socratic_v2_opening.get_retriever", return_value=retriever), patch(
            "src.llm.tools.socratic_v2_opening._llm.chat",
            return_value=_llm_response("LERNZIEL: Recognize overfitting\nKONZEPT: Overfitting"),
        ) as mock_chat:
            return socratic_v2_opening(state), mock_chat

    def test_english_session_from_ui_language(self):
        state = _state(socratic_v2_phase="opening", user_query="Starte den Lernmodus")
        state["runtime_config"].update(start_socratic_v2=True, ui_language="en")
        result, mock_chat = self._run(state)
        assert result["v2_language"] == "English"
        assert result["answer"].startswith("🎓 Welcome to the Socratic Tutor!")
        assert "Recognize overfitting" in result["answer"]
        assert "Willkommen" not in result["answer"]
        assert mock_chat.call_args.kwargs["query"].startswith("SPRACHE: English")

    def test_english_session_from_conversation(self, detector):
        detector.detect.return_value = "English"
        state = _state(
            socratic_v2_phase="opening",
            user_query="Starte den Lernmodus",
            chat_history=[_msg(MessageRole.USER, "What is overfitting exactly?")],
        )
        state["runtime_config"]["start_socratic_v2"] = True
        result, _ = self._run(state)
        assert result["v2_language"] == "English"
        assert "Welcome" in result["answer"]

    def test_no_scope_message_localized(self):
        state = _state(socratic_v2_phase="opening")
        state["runtime_config"].update(course_id=None, module_id=None, ui_language="en")
        result = socratic_v2_opening(state)
        assert result["answer"].startswith("To start the Socratic Tutor")


class TestCoreLanguage:
    def test_generation_gets_session_language(self):
        policy = {"move": "FRAGE", "zielkonzept": "Overfitting", "lernziel_session": None, "learner_update": None}
        with patch("src.llm.tools.socratic_v2_core.run_policy", return_value=policy), patch(
            "src.llm.tools.socratic_v2_core._llm.chat", return_value=_llm_response("Good. Why?")
        ) as mock_generate:
            socratic_v2_core(_state(v2_language="English"))
        assert "SPRACHE: English" in mock_generate.call_args.kwargs["query"]

    def test_policy_exit_localized(self):
        policy = {"move": "EXIT", "zielkonzept": None, "lernziel_session": None, "learner_update": None}
        with patch("src.llm.tools.socratic_v2_core.run_policy", return_value=policy):
            result = socratic_v2_core(_state(v2_language="English"))
        assert result["answer"] == localize("exit", "English")
        assert result["v2_language"] is None  # full reset

    def test_quiz_rendered_in_session_language(self):
        payload = {"kind": "truefalse", "question": "Is overfitting good?", "correct_answers": ["Falsch"]}
        policy = {"move": "QUIZ", "zielkonzept": "Overfitting", "lernziel_session": None, "learner_update": None}
        with patch("src.llm.tools.socratic_v2_core.run_policy", return_value=policy):
            result = socratic_v2_core(_state(v2_language="English", v2_quiz_items=[{**payload, "asked": False}]))
        assert result["answer"].startswith("📝 Quiz question from the course:")
        assert "- **A)** True" in result["answer"] and "- **B)** False" in result["answer"]

    def test_generation_fallback_localized(self):
        policy = {"move": "FRAGE", "zielkonzept": "Overfitting", "lernziel_session": None, "learner_update": None}
        with patch("src.llm.tools.socratic_v2_core.run_policy", return_value=policy), patch(
            "src.llm.tools.socratic_v2_core._llm.chat", return_value=_llm_response(None)
        ):
            result = socratic_v2_core(_state(v2_language="English"))
        assert result["answer"] == localize("generation_fallback", "English")


class TestConsolidationLanguage:
    def test_language_in_query_and_fallback(self):
        with patch(
            "src.llm.tools.socratic_v2_consolidation._llm.chat", return_value=_llm_response(None)
        ) as mock_chat:
            result = socratic_v2_consolidation(_state(socratic_v2_phase="consolidation", v2_language="English"))
        assert mock_chat.call_args.kwargs["query"].startswith("SPRACHE: English")
        assert result["answer"] == localize("consolidation_fallback", "English")


# ---------------------------------------------------------------------------
# Routing: exit, translation request, language switch
# ---------------------------------------------------------------------------


class TestRoutingLanguage:
    @pytest.fixture
    def contextualizer(self):
        mock = MagicMock()
        mock.contextualize_socratic.return_value = "kontextualisiert"
        with patch("src.llm.tools.contextualize.get_contextualizer", return_value=mock):
            yield mock

    def test_exit_message_in_session_language(self, contextualizer):
        result = contextualize_and_route(_state(user_query="Stop!", v2_language="English"))
        assert result["mode"] == "exit_complete"
        assert result["answer"] == localize("exit", "English")

    def test_new_exit_keyword_after_rename(self, contextualizer):
        result = contextualize_and_route(_state(user_query="Beende den Tutor"))
        assert result["mode"] == "exit_complete"

    def test_translation_request_routes_to_translate(self, contextualizer):
        history = [_msg(MessageRole.ASSISTANT, "Das ist vollkommen richtig erkannt.")]
        state = _state(user_query="kannst du das auf englisch übersetzen?", chat_history=history)
        with patch("src.llm.tools.contextualize.classify_language_request", return_value="English") as mock_classify:
            result = contextualize_and_route(state)
        mock_classify.assert_called_once()
        assert result["mode"] == "socratic_v2"
        assert result["v2_translate_last"] is True
        assert result["v2_language"] == "English"
        contextualizer.contextualize_socratic.assert_not_called()

    def test_rejected_translation_request_continues_normally(self, contextualizer):
        history = [_msg(MessageRole.ASSISTANT, "Wofür nutzt man Seq2Seq?")]
        state = _state(user_query="Für die Übersetzung ins Englische", chat_history=history)
        with patch("src.llm.tools.contextualize.classify_language_request", return_value=None):
            result = contextualize_and_route(state)
        assert not result.get("v2_translate_last")
        assert result["contextualized_query"] == "kontextualisiert"

    def test_routing_itself_does_not_detect_language(self, contextualizer, detector):
        # Detection runs inside the v2 graph, parallel to retrieval.
        result = contextualize_and_route(_state(user_query="the order of the words does not affect BoW"))
        assert result["mode"] == "socratic_v2"
        assert "v2_language" not in result
        detector.detect.assert_not_called()


class TestGraphLanguageDetection:
    """The v2 graph detects the language every core/consolidation turn and
    joins it with retrieval before the tutor node runs."""

    def _run_graph(self, state, detector, language="English"):
        from src.llm.graphs import socratic_v2 as graph_module

        detector.detect.return_value = language
        seen = {}

        def fake_core(s):
            seen.setdefault("calls", []).append({"v2_language": s.get("v2_language"), "reranked": s.get("reranked")})
            return {"answer": "ok"}

        def fake_consolidation(s):
            seen.setdefault("calls", []).append({"v2_language": s.get("v2_language")})
            return {"answer": "bye"}

        with patch.object(graph_module, "retrieve_chunks", return_value={"retrieved": []}), patch.object(
            graph_module, "rerank_chunks", return_value={"reranked": [{"text": "chunk", "metadata": {}}]}
        ), patch.object(graph_module, "socratic_v2_core", side_effect=fake_core), patch.object(
            graph_module, "socratic_v2_consolidation", side_effect=fake_consolidation
        ):
            graph_module.build_socratic_v2_graph().invoke(state)
        return seen["calls"]

    def test_core_runs_once_after_retrieval_and_detection(self, detector):
        # reranked=[] mimics the per-turn reset of a follow-up turn — a plain
        # two-edge fan-in would fire core early here with empty sources.
        calls = self._run_graph(_state(user_query="the order does not matter", reranked=[]), detector)
        assert len(calls) == 1
        assert calls[0]["v2_language"] == "English"
        assert calls[0]["reranked"]  # rerank output present

    def test_consolidation_gets_detected_language(self, detector):
        calls = self._run_graph(_state(socratic_v2_phase="consolidation"), detector)
        assert calls == [{"v2_language": "English"}]


# ---------------------------------------------------------------------------
# Translate node + graph routing
# ---------------------------------------------------------------------------


class TestTranslateNode:
    def test_translates_last_tutor_message_literally(self):
        history = [
            _msg(MessageRole.ASSISTANT, "Willkommen!"),
            _msg(MessageRole.USER, "ok"),
            _msg(MessageRole.ASSISTANT, "Das ist vollkommen richtig erkannt."),
        ]
        state = _state(chat_history=history, v2_language="English", v2_translate_last=True)
        with patch(
            "src.llm.tools.socratic_v2_translate._llm.chat", return_value=_llm_response("That is exactly right.")
        ) as mock_chat:
            result = socratic_v2_translate(state)
        assert mock_chat.call_args.kwargs["query"] == "Das ist vollkommen richtig erkannt."
        assert mock_chat.call_args.kwargs["model"] == Models.MINI
        assert "into English" in mock_chat.call_args.kwargs["system_prompt"]
        assert result["answer"] == "That is exactly right."
        assert result["v2_translate_last"] is False
        # Session state is untouched — the dialogue (or a pending quiz) continues.
        assert "socratic_v2_phase" not in result
        assert "v2_pending_quiz" not in result

    def test_failure_returns_original(self):
        state = _state(chat_history=[_msg(MessageRole.ASSISTANT, "Original")], v2_language="English")
        with patch("src.llm.tools.socratic_v2_translate._llm.chat", side_effect=RuntimeError("down")):
            result = socratic_v2_translate(state)
        assert result["answer"] == "Original"

    def test_graph_routes_translate_flag_to_translate_node(self):
        from src.llm.graphs.socratic_v2 import build_socratic_v2_graph

        state = _state(
            chat_history=[_msg(MessageRole.ASSISTANT, "Hallo")],
            v2_language="English",
            v2_translate_last=True,
        )
        with patch("src.llm.tools.socratic_v2_translate._llm.chat", return_value=_llm_response("Hello")), patch(
            "src.llm.tools.socratic_v2_core.socratic_v2_core"
        ) as mock_core:
            result = build_socratic_v2_graph().invoke(state)
        assert result["answer"] == "Hello"
        assert result["socratic_v2_phase"] == "core"  # phase unchanged
        mock_core.assert_not_called()


# ---------------------------------------------------------------------------
# Quiz helpers
# ---------------------------------------------------------------------------


def _multi_pending():
    return build_pending_quiz(
        {
            "kind": "multichoice",
            "question": "Was droht ohne Reflexion der Daten?",
            "correct_answers": ["Vorurteile fließen ein", "Diverser Blick geht verloren"],
            "incorrect_answers": ["Systeme werden autonom", "Daten werden archiviert"],
        }
    )


def _letters(pending, correct=True):
    return [o["letter"] for o in pending["options"] if o["correct"] is correct]


class TestQuizRendering:
    def test_multi_correct_announced_up_front(self):
        text = render_quiz_message(_multi_pending())
        assert "Mehrere Antworten können richtig sein" in text

    def test_single_correct_announced_up_front(self):
        pending = build_pending_quiz(
            {"kind": "multichoice", "question": "F?", "correct_answers": ["Ja"], "incorrect_answers": ["Nein"]}
        )
        assert "Genau eine Antwort ist richtig" in render_quiz_message(pending)

    def test_english_frame_keeps_course_options(self):
        text = render_quiz_message(_multi_pending(), "English")
        assert "More than one answer may be correct" in text
        assert "Vorurteile fließen ein" in text  # course content shown as authored


class TestQuizGrading:
    @pytest.mark.parametrize("template", ["{0}, {1}", "Antwort: {0} und {1}", "{0} & {1}", "{0} {1}", "{0} und {1} sind richtig"])
    def test_multi_letter_formats(self, template):
        pending = _multi_pending()
        a, b = _letters(pending)
        verdict = grade_quiz_answer(pending, template.format(a.lower(), b))
        assert verdict["recognized"] and verdict["correct"]

    def test_correct_plus_wrong_is_partial(self):
        pending = _multi_pending()
        verdict = grade_quiz_answer(pending, f"{_letters(pending)[0]}, {_letters(pending, False)[0]}")
        assert verdict["recognized"] and not verdict["correct"] and verdict["partially_correct"]

    def test_only_wrong_is_false(self):
        pending = _multi_pending()
        verdict = grade_quiz_answer(pending, _letters(pending, False)[0])
        assert verdict["recognized"] and not verdict["correct"] and not verdict["partially_correct"]

    def test_undecided_or_is_not_an_answer(self):
        pending = _multi_pending()
        a, b = _letters(pending)
        assert not grade_quiz_answer(pending, f"{a} oder {b}")["recognized"]

    def test_meta_question_is_not_an_answer(self):
        verdict = grade_quiz_answer(_multi_pending(), "ist eine mehrfachantwort möglich oder ist nur eine lösung korrekt?")
        assert not verdict["recognized"]

    def test_single_letter_with_suffix(self):
        pending = build_pending_quiz(
            {"kind": "multichoice", "question": "F?", "correct_answers": ["Ja"], "incorrect_answers": ["Nein"]}
        )
        correct = _letters(pending)[0]
        assert grade_quiz_answer(pending, f"{correct} ist richtig")["correct"]

    def test_truefalse_unwahr_is_false_not_true(self):
        pending = build_pending_quiz({"kind": "truefalse", "question": "F?", "correct_answers": ["Falsch"]})
        verdict = grade_quiz_answer(pending, "unwahr")
        assert verdict["correct"] and verdict["matched_options"] == ["Falsch"]

    def test_truefalse_english_synonyms(self):
        pending = build_pending_quiz({"kind": "truefalse", "question": "F?", "correct_answers": ["Wahr"]})
        assert grade_quiz_answer(pending, "true")["correct"]
        assert not grade_quiz_answer(pending, "no")["correct"]

    @pytest.mark.parametrize("text", ["keine Ahnung", "Lösung", "sag mir die Lösung bitte", "I don't know", "skip"])
    def test_give_up_phrases(self, text):
        assert is_quiz_give_up(text)

    def test_question_is_not_give_up(self):
        assert not is_quiz_give_up("Ist eine Mehrfachantwort möglich?")


# ---------------------------------------------------------------------------
# Learning-mode status (frontend toggle)
# ---------------------------------------------------------------------------


class TestLearningModeStatus:
    def _assistant_with_state(self, values):
        fake = SimpleNamespace(graph=MagicMock())
        fake.graph.get_state.return_value = SimpleNamespace(values=values)
        return fake

    @pytest.mark.parametrize(
        "values, expected",
        [
            ({"socratic_v2_phase": "core"}, True),
            ({"socratic_v2_phase": "consolidation"}, True),
            ({"socratic_mode": "contract"}, True),
            ({"socratic_v2_phase": None, "socratic_mode": None}, False),
            ({}, False),
        ],
    )
    def test_status_from_checkpoint(self, values, expected):
        fake = self._assistant_with_state(values)
        assert KICampusAssistant.is_learning_mode_active(fake, "thread") is expected

    def test_no_thread_is_inactive(self):
        assert KICampusAssistant.is_learning_mode_active(SimpleNamespace(graph=MagicMock()), None) is False

    def test_checkpoint_error_is_inactive(self):
        fake = SimpleNamespace(graph=MagicMock())
        fake.graph.get_state.side_effect = RuntimeError("redis down")
        assert KICampusAssistant.is_learning_mode_active(fake, "thread") is False


class TestQuizMarkdown:
    """Options must be a real Markdown list — plain lines collapse into one
    paragraph in Streamlit and Moodle (testing 2026-11)."""

    def test_options_render_as_markdown_list_with_bold_question(self):
        pending = _multi_pending()
        text = render_quiz_message(pending)
        assert f"**{pending['question']}**" in text
        for option in pending["options"]:
            assert f"- **{option['letter']})** {option['text']}" in text

    def test_gender_stars_are_escaped_for_display_only(self):
        # Two gender stars in one line would render the text between them in
        # italics ("Nutzer*innen und Lehrer*innen").
        pending = build_pending_quiz(
            {
                "kind": "multichoice",
                "question": "Was dürfen Nutzer*innen und Lehrer*innen?",
                "correct_answers": ["Daten der Patient*innen einsehen"],
                "incorrect_answers": ["Nichts"],
            }
        )
        text = render_quiz_message(pending)
        assert "**Was dürfen Nutzer\\*innen und Lehrer\\*innen?**" in text
        assert "Daten der Patient\\*innen einsehen" in text
        # State keeps the raw text — grading by option text still works.
        assert pending["question"] == "Was dürfen Nutzer*innen und Lehrer*innen?"
        verdict = grade_quiz_answer(pending, "Daten der Patient*innen einsehen")
        assert verdict["correct"] is True
