"""Tests for the rerank_chunks graph node and PassthroughReranker."""

from unittest.mock import MagicMock, patch

import pytest

from src.api.models.serializable_text_node import SerializableTextNode
from src.llm.objects.LLMs import Models
from src.llm.objects.rerankers.base import RerankResult
from src.llm.objects.rerankers.passthrough_reranker import PassthroughReranker
from src.llm.tools.rerank import rerank_chunks


def _node(text: str, score: float = 1.0) -> SerializableTextNode:
    return SerializableTextNode(text=text, metadata={}, score=score)


def _state(retrieved: list, query: str = "Was ist KI?", top_n: int = 3, reranker_type: str = "llm") -> dict:
    return {
        "user_query": query,
        "contextualized_query": query,
        "retrieved": retrieved,
        "runtime_config": {"model": Models.AZURE_FALLBACK},
        "system_config": {
            "rerank_top_n": top_n,
            "reranker_type": reranker_type,
            "min_reranker_score": 0.0,
        },
    }


# ---------------------------------------------------------------------------
# PassthroughReranker
# ---------------------------------------------------------------------------

class TestPassthroughReranker:
    def test_returns_top_n_in_original_order(self):
        nodes = [_node(f"text {i}") for i in range(5)]
        reranker = PassthroughReranker(top_n=3)
        result = reranker.rerank("query", nodes)
        assert len(result.nodes) == 3
        assert result.nodes[0].text == "text 0"

    def test_returns_all_when_fewer_than_top_n(self):
        nodes = [_node("only one")]
        reranker = PassthroughReranker(top_n=5)
        result = reranker.rerank("query", nodes)
        assert len(result.nodes) == 1

    def test_result_has_latency(self):
        reranker = PassthroughReranker(top_n=2)
        result = reranker.rerank("q", [_node("a"), _node("b")])
        assert result.latency_ms >= 0

    def test_zero_cost(self):
        reranker = PassthroughReranker(top_n=2)
        result = reranker.rerank("q", [_node("a")])
        assert result.estimated_cost_eur == 0.0

    def test_name(self):
        assert "rerank" in PassthroughReranker(top_n=1).name.lower()


# ---------------------------------------------------------------------------
# rerank_chunks node
# ---------------------------------------------------------------------------

class TestRerankChunksNode:
    def _mock_reranker(self, returned_nodes: list) -> MagicMock:
        mock = MagicMock()
        mock.rerank.return_value = RerankResult(nodes=returned_nodes, latency_ms=10.0)
        return mock

    def test_returns_reranked_nodes(self):
        nodes = [_node(f"doc {i}") for i in range(5)]
        reranked = nodes[:3]
        with patch("src.llm.tools.rerank.get_reranker", return_value=self._mock_reranker(reranked)):
            result = rerank_chunks(_state(nodes))
        assert result["reranked"] == reranked

    def test_empty_retrieved_returns_empty(self):
        result = rerank_chunks(_state([]))
        assert result["reranked"] == []

    def test_single_chunk_skips_reranking(self):
        nodes = [_node("solo")]
        result = rerank_chunks(_state(nodes))
        assert result["reranked"] == nodes

    def test_no_query_returns_retrieved_as_is(self):
        nodes = [_node("a"), _node("b")]
        state = _state(nodes, query="")
        state["contextualized_query"] = ""
        state["user_query"] = ""
        result = rerank_chunks(state)
        assert result["reranked"] == nodes

    def test_zero_results_from_reranker_returns_empty(self):
        """When reranker returns 0 nodes the no-content fallback must be triggered."""
        nodes = [_node(f"doc {i}") for i in range(5)]
        with patch("src.llm.tools.rerank.get_reranker", return_value=self._mock_reranker([])):
            result = rerank_chunks(_state(nodes))
        assert result["reranked"] == []

    def test_unknown_reranker_type_raises(self):
        nodes = [_node("a"), _node("b"), _node("c")]
        with pytest.raises(ValueError, match="Unknown reranker_type"):
            rerank_chunks(_state(nodes, reranker_type="nonexistent"))
