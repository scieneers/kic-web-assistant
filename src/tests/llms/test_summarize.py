"""Unit tests for the "summarize" scenario: full-scope retrieval (no reranking),
content-gap/EmptyModule handling, and router disambiguation between
summarizing the material vs. summarizing the conversation."""

from unittest.mock import MagicMock, patch

from llama_index.core.llms import MessageRole
from llama_index.core.schema import TextNode

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.api.models.serializable_text_node import SerializableTextNode
from src.llm.objects.LLMs import Models


def make_message(role: str, content: str) -> SerializableChatMessage:
    return SerializableChatMessage(role=role, content=content)


# ---------------------------------------------------------------------------
# KiCampusRetriever.retrieve_all — full-scope fetch, ordered by chunk position
# ---------------------------------------------------------------------------

class TestRetrieveAll:
    def _retriever(self, rows: list[dict]):
        from src.llm.objects.retriever import KiCampusRetriever

        instance = object.__new__(KiCampusRetriever)
        instance.index_name = "idx"
        instance.vector_db = MagicMock()
        instance.vector_db.fetch_all.return_value = rows
        return instance

    def _row(self, source_doc_key: str, chunk_index: int, text: str) -> dict:
        import json
        return {
            "id": f"{source_doc_key}__chunk{chunk_index}",
            "text": text,
            "metadata_json": json.dumps({"source_doc_key": source_doc_key, "chunk_index": chunk_index}),
        }

    def test_orders_by_source_doc_key_then_chunk_index(self):
        rows = [
            self._row("moodle:1:2", 1, "b-second"),
            self._row("moodle:1:1", 0, "a-first"),
            self._row("moodle:1:2", 0, "b-first"),
        ]
        retriever = self._retriever(rows)
        nodes = retriever.retrieve_all(course_id=1, module_id=[1, 2])
        assert [n.text for n in nodes] == ["a-first", "b-first", "b-second"]

    def test_missing_chunk_index_defaults_to_zero(self):
        rows = [{"id": "x", "text": "no-position", "metadata_json": "{}"}]
        retriever = self._retriever(rows)
        nodes = retriever.retrieve_all(course_id=1)
        assert [n.text for n in nodes] == ["no-position"]

    def test_builds_filter_and_forwards_to_fetch_all(self):
        retriever = self._retriever([])
        retriever.retrieve_all(course_id=79, module_id=[1, 33])
        odata_filter = retriever.vector_db.fetch_all.call_args.args[0]
        assert "(module_id eq 1 or module_id eq 33)" in odata_filter
        assert "course_id eq 79" in odata_filter


# ---------------------------------------------------------------------------
# retrieve_full_scope node — bypasses reranking by setting `reranked` directly
# ---------------------------------------------------------------------------

class TestRetrieveFullScopeNode:
    def _state(self, course_id=79, module_id=None) -> dict:
        return {
            "runtime_config": {"model": Models.AZURE_FALLBACK, "course_id": course_id, "module_id": module_id},
        }

    def test_sets_retrieved_and_reranked_identically(self):
        from src.llm.tools import retrieve as retrieve_module

        node = SerializableTextNode(text="content", metadata={})
        mock_retriever = MagicMock()
        mock_retriever.retrieve_all.return_value = [node]

        with patch.object(retrieve_module, "get_retriever", return_value=mock_retriever):
            result = retrieve_module.retrieve_full_scope(self._state())

        assert result["retrieved"] == [node]
        assert result["reranked"] == [node]
        assert result["retrieval_semantic_ranked"] is False

    def test_forwards_course_and_module_id(self):
        from src.llm.tools import retrieve as retrieve_module

        mock_retriever = MagicMock()
        mock_retriever.retrieve_all.return_value = []

        with patch.object(retrieve_module, "get_retriever", return_value=mock_retriever):
            retrieve_module.retrieve_full_scope(self._state(course_id=79, module_id=[1, 2]))

        mock_retriever.retrieve_all.assert_called_once_with(course_id=79, module_id=[1, 2])


# ---------------------------------------------------------------------------
# SummaryAnswerer — no reranker/min-score pipeline, own no-content handling
# ---------------------------------------------------------------------------

class TestSummaryAnswerer:
    def _answerer(self):
        from src.llm.objects.summary_answerer import SummaryAnswerer
        return SummaryAnswerer()

    def test_no_sources_returns_scope_specific_fallback(self):
        answerer = self._answerer()
        result = answerer.summarize(
            chat_history=[],
            sources=[],
            model=Models.AZURE_FALLBACK,
            language="German",
            scope_name="diesem Modul",
        )
        assert "diesem Modul" in result.content
        assert result.role == MessageRole.ASSISTANT

    def test_all_empty_module_returns_fixed_fallback_without_llm_call(self):
        from src.llm.objects.question_answerer import NO_CONTENT_IN_MODULE

        answerer = self._answerer()
        sources = [TextNode(text="", metadata={"type": "EmptyModule", "fullname": "KI-Grundlagen"})]

        with patch.object(answerer.llm, "chat") as mock_chat:
            result = answerer.summarize(
                chat_history=[],
                sources=sources,
                model=Models.AZURE_FALLBACK,
                language="German",
                scope_name="diesem Modul",
            )

        mock_chat.assert_not_called()
        assert result.content == NO_CONTENT_IN_MODULE.format(module_name="KI-Grundlagen")

    def test_unsupported_content_type_names_the_reason(self):
        from src.llm.objects.question_answerer import NO_CONTENT_UNSUPPORTED_TYPE

        answerer = self._answerer()
        sources = [
            TextNode(
                text="",
                metadata={
                    "type": "EmptyModule",
                    "fullname": "Lernziel-Check II",
                    "unsupported_label": "natives Moodle-Quiz",
                    "url": "https://moodle.ki-campus.org/mod/quiz/view.php?id=1406",
                },
            )
        ]

        with patch.object(answerer.llm, "chat") as mock_chat:
            result = answerer.summarize(
                chat_history=[],
                sources=sources,
                model=Models.AZURE_FALLBACK,
                language="German",
                scope_name="diesem Modul",
            )

        mock_chat.assert_not_called()
        assert result.content == NO_CONTENT_UNSUPPORTED_TYPE.format(
            module_name="Lernziel-Check II",
            label="natives Moodle-Quiz",
            url="https://moodle.ki-campus.org/mod/quiz/view.php?id=1406",
        )

    def test_normal_path_calls_llm_and_returns_content(self):
        answerer = self._answerer()
        sources = [TextNode(text="Modulinhalt zu neuronalen Netzen", metadata={"type": "module"})]
        mock_response = MagicMock()
        mock_response.content = "Zusammenfassung des Moduls [doc1]"

        with patch.object(answerer.llm, "chat", return_value=mock_response) as mock_chat:
            result = answerer.summarize(
                chat_history=[make_message(MessageRole.USER, "vorherige Frage")],
                sources=sources,
                model=Models.AZURE_FALLBACK,
                language="German",
                scope_name="diesem Modul",
            )

        assert result.content == "Zusammenfassung des Moduls [doc1]"
        call_kwargs = mock_chat.call_args.kwargs
        assert "Modulinhalt zu neuronalen Netzen" in call_kwargs["query"]
        assert "German" in call_kwargs["system_prompt"]

    def test_focus_hint_included_in_prompt(self):
        answerer = self._answerer()
        sources = [TextNode(text="Inhalt", metadata={})]
        mock_response = MagicMock()
        mock_response.content = "Zusammenfassung"

        with patch.object(answerer.llm, "chat", return_value=mock_response) as mock_chat:
            answerer.summarize(
                chat_history=[],
                sources=sources,
                model=Models.AZURE_FALLBACK,
                language="German",
                scope_name="diesem Modul",
                focus_hint="nur den zweiten Teil",
            )

        assert "nur den zweiten Teil" in mock_chat.call_args.kwargs["query"]


# ---------------------------------------------------------------------------
# generate_summary node
# ---------------------------------------------------------------------------

class TestGenerateSummaryNode:
    def _state(self, reranked, course_id=79, module_id=None) -> dict:
        return {
            "chat_history": [],
            "detected_language": "German",
            "reranked": reranked,
            "runtime_config": {"model": Models.AZURE_FALLBACK, "course_id": course_id, "module_id": module_id},
            "contextualized_query": None,
        }

    def test_fan_in_guard_defers_when_reranked_missing(self):
        from src.llm.tools.summarize import generate_summary

        result = generate_summary({"reranked": None})
        assert result == {}

    def test_delegates_to_summary_answerer(self):
        from src.llm.tools import summarize as summarize_module

        mock_answerer = MagicMock()
        mock_answerer.summarize.return_value = SerializableChatMessage(
            role=MessageRole.ASSISTANT, content="Zusammenfassung"
        )
        node = SerializableTextNode(text="Inhalt", metadata={})

        with patch.object(summarize_module, "get_summary_answerer", return_value=mock_answerer):
            result = summarize_module.generate_summary(self._state([node], module_id=42))

        assert result == {"answer": "Zusammenfassung"}
        call_kwargs = mock_answerer.summarize.call_args.kwargs
        assert call_kwargs["scope_name"] == "diesem Modul"


class TestScopeName:
    def test_single_module(self):
        from src.llm.tools.summarize import _scope_name
        assert _scope_name(course_id=1, module_id=42) == "diesem Modul"

    def test_multiple_modules(self):
        from src.llm.tools.summarize import _scope_name
        assert _scope_name(course_id=1, module_id=[1, 2]) == "diesen Modulen"

    def test_course_only(self):
        from src.llm.tools.summarize import _scope_name
        assert _scope_name(course_id=1, module_id=None) == "diesem Kurs"

    def test_neither(self):
        from src.llm.tools.summarize import _scope_name
        assert _scope_name(course_id=None, module_id=None) == "diesem Bereich"


# ---------------------------------------------------------------------------
# Router: "summarize" scope guard + history-based disambiguation signal
# ---------------------------------------------------------------------------

class TestSummarizeRouting:
    def _state(self, course_id=None, module_id=None, chat_history=None) -> dict:
        return {
            "user_query": "Fasse das zusammen",
            "chat_history": chat_history or [],
            "runtime_config": {
                "model": Models.AZURE_FALLBACK,
                "course_id": course_id,
                "module_id": module_id,
                "start_socratic": False,
            },
            "system_config": {"enable_socratic": False},
        }

    def test_summarize_without_scope_downgrades_to_no_vectordb(self):
        from src.llm.tools import contextualize as contextualize_module

        mock_contextualizer = MagicMock()
        mock_contextualizer.classify_scenario.return_value = "summarize"
        mock_contextualizer.contextualize.return_value = "Fasse das zusammen"

        with patch.object(contextualize_module, "get_contextualizer", return_value=mock_contextualizer):
            result = contextualize_module.contextualize_and_route(self._state())

        assert result["mode"] == "no_vectordb"

    def test_summarize_with_module_scope_is_kept(self):
        from src.llm.tools import contextualize as contextualize_module

        mock_contextualizer = MagicMock()
        mock_contextualizer.classify_scenario.return_value = "summarize"
        mock_contextualizer.contextualize.return_value = "Fasse das Modul zusammen"

        with patch.object(contextualize_module, "get_contextualizer", return_value=mock_contextualizer):
            result = contextualize_module.contextualize_and_route(self._state(course_id=79, module_id=42))

        assert result["mode"] == "summarize"

    def test_has_prior_history_flag_passed_to_classifier(self):
        from src.llm.tools import contextualize as contextualize_module

        mock_contextualizer = MagicMock()
        mock_contextualizer.classify_scenario.return_value = "no_vectordb"

        history = [make_message(MessageRole.USER, "erste Frage")]
        with patch.object(contextualize_module, "get_contextualizer", return_value=mock_contextualizer):
            contextualize_module.contextualize_and_route(self._state(chat_history=history))

        assert mock_contextualizer.classify_scenario.call_args.kwargs["has_prior_history"] is True

    def test_no_prior_history_flag_false_on_first_message(self):
        from src.llm.tools import contextualize as contextualize_module

        mock_contextualizer = MagicMock()
        mock_contextualizer.classify_scenario.return_value = "summarize"
        mock_contextualizer.contextualize.return_value = "Fasse zusammen"

        with patch.object(contextualize_module, "get_contextualizer", return_value=mock_contextualizer):
            contextualize_module.contextualize_and_route(self._state(course_id=79, chat_history=[]))

        assert mock_contextualizer.classify_scenario.call_args.kwargs["has_prior_history"] is False


class TestClassifyScenarioHistoryContext:
    def test_router_prompt_formats_without_error(self):
        """router_prompt.txt must contain exactly the {history_context} placeholder
        (no stray braces that would break .format())."""
        from src.llm.objects.contextualizer import Contextualizer

        with patch("src.llm.objects.contextualizer.LLM"):
            contextualizer = Contextualizer()

        mock_response = MagicMock()
        mock_response.content = "summarize"
        with patch.object(contextualizer.llm, "chat", return_value=mock_response) as mock_chat:
            contextualizer.classify_scenario(query="Fasse zusammen", model=Models.AZURE_FALLBACK, has_prior_history=True)

        system_prompt = mock_chat.call_args.kwargs["system_prompt"]
        assert "a prior conversation already exists" in system_prompt

    def test_first_message_context_note(self):
        from src.llm.objects.contextualizer import Contextualizer

        with patch("src.llm.objects.contextualizer.LLM"):
            contextualizer = Contextualizer()

        mock_response = MagicMock()
        mock_response.content = "summarize"
        with patch.object(contextualizer.llm, "chat", return_value=mock_response) as mock_chat:
            contextualizer.classify_scenario(query="Fasse zusammen", model=Models.AZURE_FALLBACK, has_prior_history=False)

        system_prompt = mock_chat.call_args.kwargs["system_prompt"]
        assert "first message in this session" in system_prompt
