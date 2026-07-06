"""
E2E test: EmptyModule marker runs through the real compiled simple_hop graph.

Mocks only the I/O boundaries (retrieve + language detection) so that
rerank_chunks and generate_answer execute for real. This verifies:
  - single-node rerank bypass passes the EmptyModule node through unchanged
  - LangGraph state plumbing (reranked key written by rerank_node, read by answer_node)
  - parallel fan-in (rerank_node + detect_language_node both complete before answer_node)

The graph must be built inside each test's patch context so that the local
function references captured by graph.add_node() resolve to the mocks.
"""

from unittest.mock import patch

from src.api.models.serializable_text_node import SerializableTextNode
from src.llm.graphs.simple_hop import build_simple_hop_graph
from src.llm.objects.LLMs import Models
from src.llm.objects.question_answerer import NO_CONTENT_IN_MODULE


def _base_state() -> dict:
    return {
        "user_query": "Was ist in diesem Modul?",
        "chat_history": [],
        "contextualized_query": "Was ist in diesem Modul?",
        "runtime_config": {
            "model": Models.AZURE_FALLBACK,
            "course_id": 42,
            "module_id": None,
            "thread_id": "test-e2e",
        },
        "system_config": {
            "rerank_top_n": 5,
            "retrieve_top_n": 10,
            "enable_socratic": False,
            "reranker_type": "llm",
            "min_reranker_score": 0.0,
        },
    }


def _empty_node(fullname: str = "KI-Grundlagen") -> SerializableTextNode:
    return SerializableTextNode(text="", metadata={"type": "EmptyModule", "fullname": fullname})


class TestEmptyModuleE2E:
    def test_fixed_message_returned_for_empty_module(self):
        with patch("src.llm.tools.retrieve.retrieve_chunks", return_value={"retrieved": [_empty_node("KI-Grundlagen")]}), \
             patch("src.llm.tools.language.detect_language", return_value={"detected_language": "German"}):
            graph = build_simple_hop_graph()
            result = graph.invoke(_base_state())

        assert result["answer"] == NO_CONTENT_IN_MODULE.format(module_name="KI-Grundlagen")

    def test_module_name_in_answer(self):
        with patch("src.llm.tools.retrieve.retrieve_chunks", return_value={"retrieved": [_empty_node("Einführung in NLP")]}), \
             patch("src.llm.tools.language.detect_language", return_value={"detected_language": "German"}):
            graph = build_simple_hop_graph()
            result = graph.invoke(_base_state())

        assert "Einführung in NLP" in result["answer"]

    def test_no_llm_call_for_empty_module(self):
        """generate_answer must short-circuit without touching QuestionAnswerer."""
        with patch("src.llm.tools.retrieve.retrieve_chunks", return_value={"retrieved": [_empty_node()]}), \
             patch("src.llm.tools.language.detect_language", return_value={"detected_language": "German"}), \
             patch("src.llm.tools.answer.get_question_answerer") as mock_qa:
            graph = build_simple_hop_graph()
            graph.invoke(_base_state())

        mock_qa.assert_not_called()
