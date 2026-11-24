"""Sokratischer Tutor per Toggle: Start/Ende über Request-Flags ohne sichtbare
Nachricht im Verlauf (Testing-Feedback S. 15).

- start_socratic_v2 / stop_socratic_v2 sind reine Signale: ein mitgeschickter
  Text landet nicht in der chat_history (z. B. nach Page-Reload), die Antwort
  des Tutors (Begrüßung / Beenden-Bestätigung) schon.
- stop_socratic_v2 ist idempotent: ohne laufende Session wird nur bestätigt,
  der (leere) Signaltext läuft nicht als Chatfrage durch den RAG-Pfad.
- Per Stichwort getipptes Beenden ("Stopp") bleibt eine sichtbare Nachricht.
"""

from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from llama_index.core.llms import MessageRole

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.api.models.serializable_text_node import SerializableTextNode
from src.llm.assistant import KICampusAssistant
from src.llm.objects.LLMs import Models
from src.llm.state.socratic_v2_language import localize
from src.llm.tools.contextualize import contextualize_and_route


def _msg(role, content):
    return SerializableChatMessage(role=role, content=content)


def _role(message) -> str:
    return message.role.value if hasattr(message.role, "value") else message.role


def _state(*, phase="core", query="", start=False, stop=False, ui_language=None, **overrides) -> dict:
    state = {
        "user_query": query,
        "chat_history": [],
        "runtime_config": {
            "model": Models.AZURE_FALLBACK,
            "course_id": 42,
            "module_id": 7,
            "thread_id": "t",
            "start_socratic": False,
            "start_socratic_v2": start,
            "stop_socratic_v2": stop,
            "ui_language": ui_language,
        },
        "system_config": {"enable_socratic": False, "enable_socratic_v2": True},
        "socratic_v2_phase": phase,
    }
    state.update(overrides)
    return state


@pytest.fixture
def contextualizer():
    mock = MagicMock()
    mock.contextualize_socratic.return_value = "kontextualisiert"
    mock.classify_scenario.return_value = "no_vectordb"
    with patch("src.llm.tools.contextualize.get_contextualizer", return_value=mock):
        yield mock


# ---------------------------------------------------------------------------
# Routing (contextualize_and_route)
# ---------------------------------------------------------------------------


class TestToggleRouting:
    def test_stop_flag_ends_active_session(self, contextualizer):
        result = contextualize_and_route(_state(stop=True, v2_language="English"))
        assert result["mode"] == "exit_complete"
        assert result["socratic_v2_phase"] is None
        assert result["answer"] == localize("exit", "English")
        assert result["signal_turn"] is True
        contextualizer.contextualize_socratic.assert_not_called()

    def test_stop_flag_without_session_is_idempotent(self, contextualizer):
        result = contextualize_and_route(_state(phase=None, stop=True, ui_language="en"))
        assert result["mode"] == "exit_complete"
        assert result["answer"] == localize("exit", "English")
        assert result["signal_turn"] is True
        # The empty signal text must not run through the normal chat path.
        contextualizer.classify_scenario.assert_not_called()
        contextualizer.contextualize.assert_not_called()

    def test_stop_flag_ignored_when_v2_disabled(self, contextualizer):
        state = _state(phase=None, stop=True, query="Hallo")
        state["system_config"]["enable_socratic_v2"] = False
        result = contextualize_and_route(state)
        assert result["mode"] == "no_vectordb"
        assert not result.get("signal_turn")

    def test_start_flag_marks_signal_turn(self, contextualizer):
        result = contextualize_and_route(_state(phase=None, start=True))
        assert result["mode"] == "socratic_v2"
        assert result["socratic_v2_phase"] == "opening"
        assert result["signal_turn"] is True

    def test_trigger_phrase_is_not_a_signal_turn(self, contextualizer):
        # Typed by the learner → visible message, stays in the history.
        result = contextualize_and_route(_state(phase=None, query="starte lernmodus v2"))
        assert result["socratic_v2_phase"] == "opening"
        assert result["signal_turn"] is False

    def test_keyword_exit_is_not_a_signal_turn(self, contextualizer):
        result = contextualize_and_route(_state(query="Stopp"))
        assert result["mode"] == "exit_complete"
        assert result["signal_turn"] is False


# ---------------------------------------------------------------------------
# History handling
# ---------------------------------------------------------------------------


class TestAppendTurnToHistory:
    def _answer(self):
        return _msg(MessageRole.ASSISTANT, "Antwort")

    def test_normal_turn_stores_user_and_assistant(self):
        history = KICampusAssistant._append_turn_to_history({"chat_history": []}, "Frage", self._answer())
        assert [(_role(m), m.content) for m in history] == [("user", "Frage"), ("assistant", "Antwort")]

    def test_signal_turn_stores_only_assistant(self):
        history = KICampusAssistant._append_turn_to_history(
            {"chat_history": [], "signal_turn": True}, "Sokratischen Tutor starten", self._answer()
        )
        assert [(_role(m), m.content) for m in history] == [("assistant", "Antwort")]


# ---------------------------------------------------------------------------
# End-to-end through the real assistant graph
# ---------------------------------------------------------------------------


def _resp(text):
    response = MagicMock()
    response.content = text
    return response


@pytest.fixture
def assistant():
    """Real KICampusAssistant (main graph, checkpointer, routing, v2 subgraph);
    only LLM calls and retrieval are stubbed."""
    import src.llm.graphs.socratic_v2 as v2_graph

    detector = MagicMock()
    detector.detect.side_effect = lambda query, chat_history: (
        "English" if any(w in query.lower() for w in ("the ", "what")) else "German"
    )
    chunk = SerializableTextNode(text="BoW ignoriert die Reihenfolge.", metadata={})
    retriever = MagicMock()
    retriever.retrieve_all.return_value = [MagicMock(metadata={"fullname": "ML Modul", "module_id": 7}, text="Inhalt")]
    retriever.retrieve_items.return_value = []
    ctx = MagicMock()
    ctx.contextualize_socratic.return_value = "kontext"
    policy = {"move": "FRAGE", "zielkonzept": "BoW", "lernziel_session": "BoW", "learner_update": None, "begruendung": None}

    with patch("src.llm.assistant.build_checkpointer", return_value=MemorySaver()), \
         patch.object(v2_graph, "retrieve_chunks", return_value={"retrieved": [chunk], "retrieval_semantic_ranked": False}), \
         patch.object(v2_graph, "rerank_chunks", return_value={"reranked": [chunk]}), \
         patch("src.llm.tools.language.get_language_detector", return_value=detector), \
         patch("src.llm.tools.contextualize.get_contextualizer", return_value=ctx), \
         patch("src.llm.tools.contextualize.classify_language_request", return_value=None), \
         patch("src.llm.tools.socratic_v2_opening.get_retriever", return_value=retriever), \
         patch("src.llm.tools.socratic_v2_opening._llm.chat", return_value=_resp("LERNZIEL: BoW verstehen\nKONZEPT: BoW")), \
         patch("src.llm.tools.socratic_v2_core.run_policy", return_value=policy), \
         patch("src.llm.tools.socratic_v2_core._llm.chat", return_value=_resp("Good. What else?")):
        yield KICampusAssistant(enable_socratic_v2=True)


class TestToggleEndToEnd:
    THREAD = "toggle-thread"

    def _turn(self, assistant, query, **flags):
        message, _ = assistant.chat_with_course(
            query=query, model=Models.AZURE_FALLBACK, course_id=42, module_id=7, thread_id=self.THREAD, **flags
        )
        return message

    def _history(self, assistant):
        values = assistant.graph.get_state({"configurable": {"thread_id": self.THREAD}}).values
        return values.get("chat_history", [])

    def _seed_english_chat(self, assistant):
        assistant.graph.update_state(
            {"configurable": {"thread_id": self.THREAD}},
            {
                "chat_history": [
                    _msg(MessageRole.USER, "What is the bag of words model?"),
                    _msg(MessageRole.ASSISTANT, "It represents text as word counts."),
                ],
                "system_config": assistant.system_config,
            },
        )

    def test_toggle_on_and_off_leaves_no_signal_text_in_history(self, assistant):
        self._seed_english_chat(assistant)

        welcome = self._turn(assistant, "", start_socratic_v2=True)
        # Session language from the prior chat, not from the (empty) signal.
        assert welcome.content.startswith("🎓 Welcome to the Socratic Tutor")
        assert assistant.is_learning_mode_active(self.THREAD)

        self._turn(assistant, "the order of the words does not matter")

        bye = self._turn(assistant, "", stop_socratic_v2=True)
        assert bye.content == localize("exit", "English")
        assert not assistant.is_learning_mode_active(self.THREAD)

        history = self._history(assistant)
        assert [(_role(m), m.content) for m in history][-5:] == [
            ("assistant", "It represents text as word counts."),
            ("assistant", welcome.content),
            ("user", "the order of the words does not matter"),
            ("assistant", "Good. What else?"),
            ("assistant", bye.content),
        ]

    def test_signal_text_sent_with_start_flag_is_not_stored(self, assistant):
        self._turn(assistant, "Sokratischen Tutor starten", start_socratic_v2=True)
        assert assistant.is_learning_mode_active(self.THREAD)
        assert not any(m.content == "Sokratischen Tutor starten" for m in self._history(assistant))

    def test_stop_flag_twice_is_idempotent(self, assistant):
        self._turn(assistant, "", start_socratic_v2=True)
        self._turn(assistant, "", stop_socratic_v2=True)
        again = self._turn(assistant, "", stop_socratic_v2=True, ui_language="en")
        assert again.content == localize("exit", "English")
        assert not assistant.is_learning_mode_active(self.THREAD)
        assert not any(_role(m) == "user" for m in self._history(assistant))

    def test_typed_keyword_exit_stays_visible(self, assistant):
        self._turn(assistant, "", start_socratic_v2=True)
        self._turn(assistant, "Stopp")
        history = self._history(assistant)
        assert (_role(history[-2]), history[-2].content) == ("user", "Stopp")
        assert not assistant.is_learning_mode_active(self.THREAD)
