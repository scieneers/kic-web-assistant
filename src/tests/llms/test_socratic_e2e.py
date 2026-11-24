"""
E2E test: full socratic dialogue cycle through the real KICampusAssistant graph.

Mocks only the I/O boundaries (LLM.chat, retrieve/rerank) so that the actual
routing logic executes for real:
  - contextualize_and_route's explicit start_socratic entry trigger
  - contract -> diagnose -> core state machine, driven by the real
    BoundedMemorySaver checkpointer across multiple .chat_with_course() calls
  - core's internal branching (CONTINUE / HINT / auto-EXPLAIN after 2 hints)
  - exit handling and full socratic-state reset

LLM.chat is patched once at the class level with a dispatcher that inspects
system_prompt to decide which canned response to return, since every
socratic node/helper instantiates its own LLM() and there is no single
call site to patch individually.
"""

from unittest.mock import MagicMock, patch

from src.llm.assistant import KICampusAssistant
from src.llm.objects.LLMs import Models
from src.api.models.serializable_text_node import SerializableTextNode


def _response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    return msg


def _make_fake_chat(modes=("MODE: CONTINUE", "MODE: HINT", "MODE: HINT")):
    """Dispatches by system_prompt content; drives the mode sequence for
    the stuckness classifier (evaluate_user_response) across turns."""
    mode_sequence = iter(modes)

    def fake_chat(**kwargs) -> MagicMock:
        prompt = kwargs.get("system_prompt") or ""

        if "diagnostic baseline assessment" in prompt:
            return _response(
                "LEARNING_OBJECTIVE: Verstehe Gradient Descent\n"
                "DIAGNOSTIC_QUESTION: Was weißt du bereits über Optimierungsverfahren?"
            )
        if "Contextualizer for a Socratic learning dialogue system" in prompt:
            return _response("Gradient Descent Lernrate Optimierung")
        if "Socratic learning assessor" in prompt:
            return _response(next(mode_sequence))
        if "skilled Socratic tutor" in prompt:
            return _response("Was passiert, wenn die Lernrate zu groß gewählt wird?")
        if "providing graduated hints" in prompt:
            return _response("Guter Ansatz! Denk an die Schrittgröße beim Parameter-Update.")
        if "providing an explanation only after" in prompt:
            return _response(
                "Gradient Descent aktualisiert die Parameter entgegen der Richtung des Gradienten, "
                "gesteuert durch die Lernrate."
            )
        raise AssertionError(f"fake_chat: no dispatch rule for system_prompt {prompt[:80]!r}")

    return fake_chat


def _chunk() -> SerializableTextNode:
    return SerializableTextNode(text="Gradient Descent ist ein Optimierungsverfahren.", metadata={})


class TestSocraticFullCycleE2E:
    def test_full_cycle_contract_to_explain_to_exit(self):
        with patch("src.llm.graphs.socratic.retrieve_chunks", return_value={"retrieved": [_chunk()]}), \
             patch("src.llm.graphs.socratic.rerank_chunks", return_value={"reranked": [_chunk()]}), \
             patch("src.llm.objects.LLMs.LLM.chat", side_effect=_make_fake_chat()):

            assistant = KICampusAssistant(enable_socratic=True)
            config = None

            def state_after(thread_id: str) -> dict:
                nonlocal config
                config = {"configurable": {"thread_id": thread_id}}
                return assistant.graph.get_state(config).values

            # Turn 1: entry via explicit request-level flag, no LLM call expected.
            msg1, thread_id = assistant.chat_with_course(
                query="unterstütze mich beim lernen",
                model=Models.AZURE_FALLBACK,
                course_id=42,
                start_socratic=True,
            )
            state1 = state_after(thread_id)
            assert "Lernmodus" in msg1.content
            assert state1["socratic_mode"] == "diagnose"

            # Turn 2: names the topic -> diagnose node runs its one LLM call.
            msg2, _ = assistant.chat_with_course(
                query="Ich möchte Gradient Descent verstehen",
                model=Models.AZURE_FALLBACK,
                course_id=42,
                thread_id=thread_id,
            )
            state2 = state_after(thread_id)
            assert msg2.content == "Was weißt du bereits über Optimierungsverfahren?"
            assert state2["socratic_mode"] == "core"
            assert state2["learning_objective"] == "Verstehe Gradient Descent"

            # Turn 3: good-faith answer -> classifier returns CONTINUE.
            msg3, _ = assistant.chat_with_course(
                query="Man passt die Parameter iterativ an, glaube ich.",
                model=Models.AZURE_FALLBACK,
                course_id=42,
                thread_id=thread_id,
            )
            state3 = state_after(thread_id)
            assert msg3.content == "Was passiert, wenn die Lernrate zu groß gewählt wird?"
            assert state3["socratic_mode"] == "core"
            assert state3["attempt_count"] == 1
            assert state3["number_given_hints"] == 0

            # Turns 4 & 5: stuck answers -> classifier returns HINT twice.
            msg4, _ = assistant.chat_with_course(
                query="Ich weiß nicht.",
                model=Models.AZURE_FALLBACK,
                course_id=42,
                thread_id=thread_id,
            )
            state4 = state_after(thread_id)
            assert "Schrittgröße" in msg4.content
            assert state4["number_given_hints"] == 1
            assert state4["socratic_mode"] == "core"

            msg5, _ = assistant.chat_with_course(
                query="Ich weiß es immer noch nicht.",
                model=Models.AZURE_FALLBACK,
                course_id=42,
                thread_id=thread_id,
            )
            state5 = state_after(thread_id)
            assert "Schrittgröße" in msg5.content
            assert state5["number_given_hints"] == 2
            assert state5["socratic_mode"] == "core"

            # Turn 6: third stuck turn -> auto-escalates to EXPLAIN without a
            # further classifier call (number_given_hints >= 2 shortcut),
            # then resets socratic state so a new topic can start.
            msg6, _ = assistant.chat_with_course(
                query="Ich habe wirklich keine Ahnung.",
                model=Models.AZURE_FALLBACK,
                course_id=42,
                thread_id=thread_id,
            )
            state6 = state_after(thread_id)
            assert "Gradient Descent aktualisiert die Parameter" in msg6.content
            assert "Lernmodus" in msg6.content  # closing text appended by socratic_explain
            assert state6["socratic_mode"] == "diagnose"
            assert state6["number_given_hints"] == 0
            assert state6["learning_objective"] is None

            # Turn 7: exit keyword -> fixed message, no LLM call, full reset.
            msg7, _ = assistant.chat_with_course(
                query="exit",
                model=Models.AZURE_FALLBACK,
                course_id=42,
                thread_id=thread_id,
            )
            state7 = state_after(thread_id)
            assert "Lernmodus verlassen" in msg7.content
            assert state7["socratic_mode"] is None

    def test_reflect_path_on_demonstrated_mastery(self):
        """Mastery answer -> REFLECT branch -> real generate_reflection_text().

        Regression test: generate_reflection_text() previously had no
        parameters while socratic_core called it with learning_objective=...,
        raising a TypeError the moment a student actually reached mastery.
        Unit tests never caught it because they patch generate_reflection_text
        out entirely; this test deliberately leaves it unmocked.
        """
        with patch("src.llm.graphs.socratic.retrieve_chunks", return_value={"retrieved": [_chunk()]}), \
             patch("src.llm.graphs.socratic.rerank_chunks", return_value={"reranked": [_chunk()]}), \
             patch("src.llm.objects.LLMs.LLM.chat", side_effect=_make_fake_chat(modes=["MODE: REFLECT"])):

            assistant = KICampusAssistant(enable_socratic=True)

            def state_after(thread_id: str) -> dict:
                config = {"configurable": {"thread_id": thread_id}}
                return assistant.graph.get_state(config).values

            _, thread_id = assistant.chat_with_course(
                query="unterstütze mich beim lernen",
                model=Models.AZURE_FALLBACK,
                course_id=42,
                start_socratic=True,
            )
            assistant.chat_with_course(
                query="Ich möchte Gradient Descent verstehen",
                model=Models.AZURE_FALLBACK,
                course_id=42,
                thread_id=thread_id,
            )
            msg3, _ = assistant.chat_with_course(
                query="Gradient Descent aktualisiert Parameter entgegen dem Gradienten, "
                "gesteuert durch die Lernrate, bis das Minimum der Verlustfunktion erreicht ist.",
                model=Models.AZURE_FALLBACK,
                course_id=42,
                thread_id=thread_id,
            )
            state3 = state_after(thread_id)
            assert "Ausgezeichnet" in msg3.content
            assert "Verstehe Gradient Descent" in msg3.content
            assert state3["socratic_mode"] == "diagnose"
            assert state3["attempt_count"] == 0
            assert state3["number_given_hints"] == 0
            assert state3["learning_objective"] is None
