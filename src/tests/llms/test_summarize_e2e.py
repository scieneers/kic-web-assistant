"""
E2E test: the summarize scenario runs through the real compiled summarize
subgraph (retrieve_full_scope -> detect_language -> summarize_answer_node ->
citation_node). Mocks only the I/O boundaries (retriever + summary answerer)
so that generate_summary and LangGraph fan-in/fan-out plumbing execute for
real — mirrors test_empty_module_e2e.py's approach for simple_hop.

Note: get_summary_answerer() is a process-wide singleton (like
get_question_answerer()), so tests patch it directly at the factory-function
boundary rather than constructing a real SummaryAnswerer with a mocked LLM
class — the latter would leak a stale mock into later tests via the cached
singleton.
"""

from unittest.mock import MagicMock, patch

from llama_index.core.llms import MessageRole

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.api.models.serializable_text_node import SerializableTextNode
from src.llm.graphs.summarize import build_summarize_graph
from src.llm.objects.LLMs import Models
from src.llm.objects.question_answerer import NO_CONTENT_IN_MODULE


def _base_state(module_id=None) -> dict:
    return {
        "user_query": "Fasse dieses Modul zusammen",
        "chat_history": [],
        "contextualized_query": "Fasse dieses Modul zusammen",
        "runtime_config": {
            "model": Models.AZURE_FALLBACK,
            "course_id": 79,
            "module_id": module_id,
            "thread_id": "test-summarize-e2e",
        },
        "system_config": {
            "rerank_top_n": 5,
            "retrieve_top_n": 10,
            "enable_socratic": False,
            "reranker_type": "llm",
            "min_reranker_score": 0.0,
        },
    }


def _answer_message(content: str) -> SerializableChatMessage:
    return SerializableChatMessage(role=MessageRole.ASSISTANT, content=content)


class TestSummarizeE2E:
    def test_no_reranker_involved(self):
        """The reranker singleton must never be touched by the summarize path."""
        node = SerializableTextNode(text="Modulinhalt", metadata={"type": "module"})
        mock_answerer = MagicMock()
        mock_answerer.summarize.return_value = _answer_message("Zusammenfassung [doc1]")

        with patch("src.llm.tools.retrieve.get_retriever") as mock_get_retriever, \
             patch("src.llm.tools.language.detect_language", return_value={"detected_language": "German"}), \
             patch("src.llm.tools.rerank.get_reranker") as mock_get_reranker, \
             patch("src.llm.tools.summarize.get_summary_answerer", return_value=mock_answerer):
            mock_get_retriever.return_value.retrieve_all.return_value = [node]

            graph = build_summarize_graph()
            result = graph.invoke(_base_state())

        mock_get_reranker.assert_not_called()
        assert result["answer"] == "Zusammenfassung [doc1]"
        assert result["reranked"] == [node]

    def test_empty_module_marker_short_circuits_without_llm_call(self):
        """EmptyModule handling in SummaryAnswerer.summarize runs for real here
        (only the retriever is mocked) — LLM() is side-effect-free to construct,
        and .chat() must never be reached for this branch."""
        from src.llm.objects.LLMs import LLM

        node = SerializableTextNode(text="", metadata={"type": "EmptyModule", "fullname": "KI-Grundlagen"})

        with patch("src.llm.tools.retrieve.get_retriever") as mock_get_retriever, \
             patch("src.llm.tools.language.detect_language", return_value={"detected_language": "German"}), \
             patch.object(LLM, "chat") as mock_chat:
            mock_get_retriever.return_value.retrieve_all.return_value = [node]

            graph = build_summarize_graph()
            result = graph.invoke(_base_state(module_id=42))

        mock_chat.assert_not_called()
        assert result["answer"] == NO_CONTENT_IN_MODULE.format(module_name="KI-Grundlagen")

    def test_no_sources_returns_scope_fallback_without_llm_call(self):
        from src.llm.objects.LLMs import LLM

        with patch("src.llm.tools.retrieve.get_retriever") as mock_get_retriever, \
             patch("src.llm.tools.language.detect_language", return_value={"detected_language": "German"}), \
             patch.object(LLM, "chat") as mock_chat:
            mock_get_retriever.return_value.retrieve_all.return_value = []

            graph = build_summarize_graph()
            result = graph.invoke(_base_state(module_id=42))

        mock_chat.assert_not_called()
        assert "diesem Modul" in result["answer"]

    def test_citation_node_produces_markdown(self):
        node = SerializableTextNode(text="Modulinhalt", metadata={"url": "https://example.org", "title": "Modul X"})
        mock_answerer = MagicMock()
        mock_answerer.summarize.return_value = _answer_message("Der Kurs behandelt X [doc1].")

        with patch("src.llm.tools.retrieve.get_retriever") as mock_get_retriever, \
             patch("src.llm.tools.language.detect_language", return_value={"detected_language": "German"}), \
             patch("src.llm.tools.summarize.get_summary_answerer", return_value=mock_answerer):
            mock_get_retriever.return_value.retrieve_all.return_value = [node]

            graph = build_summarize_graph()
            result = graph.invoke(_base_state())

        assert result["citations_markdown"] is not None
        assert "example.org" in result["citations_markdown"]
