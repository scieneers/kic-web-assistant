"""Tests for the rerank_chunks graph node and PassthroughReranker."""

from unittest.mock import MagicMock, patch

import pytest

from src.api.models.serializable_text_node import SerializableTextNode
from src.llm.objects.LLMs import Models
from src.llm.objects.rerankers.base import BaseReranker, RerankResult
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


# ---------------------------------------------------------------------------
# min_score filtering + score normalization (BaseReranker contract)
# ---------------------------------------------------------------------------

class _DummyReranker(BaseReranker):
    """Identity-scale reranker exercising the shared base-class logic."""

    def rerank(self, query, nodes, model=None):
        kept, dropped = self.apply_min_score(nodes)
        return RerankResult(nodes=kept[: self.top_n], latency_ms=0.0, metadata={"dropped_below_min_score": dropped})

    @property
    def name(self):
        return "Dummy"


class TestMinScoreFiltering:
    def test_disabled_by_default(self):
        nodes = [_node("a", 0.1), _node("b", 0.9), SerializableTextNode(text="c", metadata={}, score=None)]
        kept, dropped = _DummyReranker(top_n=5).apply_min_score(nodes)
        assert kept == nodes
        assert dropped == 0

    def test_drops_below_threshold(self):
        nodes = [_node("a", 0.9), _node("b", 0.3), _node("c", 0.7)]
        kept, dropped = _DummyReranker(top_n=5, min_score=0.5).apply_min_score(nodes)
        assert [n.text for n in kept] == ["a", "c"]
        assert dropped == 1

    def test_keeps_scores_at_threshold(self):
        nodes = [_node("a", 0.5)]
        kept, dropped = _DummyReranker(top_n=5, min_score=0.5).apply_min_score(nodes)
        assert len(kept) == 1
        assert dropped == 0

    def test_unscored_nodes_dropped_when_filter_active(self):
        nodes = [SerializableTextNode(text="no score", metadata={}, score=None), _node("ok", 0.8)]
        kept, dropped = _DummyReranker(top_n=5, min_score=0.5).apply_min_score(nodes)
        assert [n.text for n in kept] == ["ok"]
        assert dropped == 1

    def test_all_below_threshold_returns_empty_for_no_answer(self):
        """All hits weak → empty result → answer node produces the no-answer fallback."""
        nodes = [_node("a", 0.1), _node("b", 0.2)]
        result = _DummyReranker(top_n=5, min_score=0.5).rerank("q", nodes)
        assert result.nodes == []
        assert result.metadata["dropped_below_min_score"] == 2

    def test_rerank_node_passes_min_score_drop_through_to_empty_result(self):
        nodes = [_node(f"doc {i}", 0.1) for i in range(3)]
        state = _state(nodes)
        state["system_config"]["min_reranker_score"] = 0.5
        dummy = _DummyReranker(top_n=3, min_score=0.5)
        with patch("src.llm.tools.rerank.get_reranker", return_value=dummy):
            result = rerank_chunks(state)
        assert result["reranked"] == []


class TestScoreNormalization:
    """Each backend maps its native score scale to the shared 0-1 min_score scale."""

    def test_llm_reranker_uses_1_to_10_scale(self):
        from src.llm.objects.rerankers.llm_reranker import LLMReranker
        instance = object.__new__(LLMReranker)  # skip heavy __init__
        assert instance.normalize_score(7.0) == pytest.approx(0.7)

    def test_azure_semantic_uses_0_to_4_scale(self):
        from src.llm.objects.rerankers.azure_semantic_reranker import AzureSemanticReranker
        instance = object.__new__(AzureSemanticReranker)
        assert instance.normalize_score(2.0) == pytest.approx(0.5)

    def test_bge_scores_are_already_0_to_1(self):
        from src.llm.objects.rerankers.bge_reranker import BGEReranker
        instance = object.__new__(BGEReranker)
        assert instance.normalize_score(0.8) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# BGEReranker (CrossEncoder mocked — no model download)
# ---------------------------------------------------------------------------

class TestBGEReranker:
    def _make(self, scores: list[float], top_n: int = 2, min_score: float = 0.0):
        from src.llm.objects.rerankers import bge_reranker as module
        with patch("sentence_transformers.CrossEncoder") as cross_encoder_cls:
            cross_encoder_cls.return_value.predict.return_value = scores
            reranker = module.BGEReranker(top_n=top_n, min_score=min_score)
        return reranker, cross_encoder_cls

    def test_sigmoid_activation_is_forced(self):
        _, cross_encoder_cls = self._make([0.5])
        assert "default_activation_function" in cross_encoder_cls.call_args.kwargs

    def test_sorts_by_score_and_slices_top_n(self):
        reranker, _ = self._make([0.9, 0.2, 0.6], top_n=2)
        result = reranker.rerank("q", [_node("a"), _node("b"), _node("c")])
        assert [n.text for n in result.nodes] == ["a", "c"]
        assert [n.score for n in result.nodes] == [0.9, 0.6]

    def test_min_score_drops_weak_chunks(self):
        reranker, _ = self._make([0.9, 0.2, 0.6], top_n=3, min_score=0.7)
        result = reranker.rerank("q", [_node("a"), _node("b"), _node("c")])
        assert [n.text for n in result.nodes] == ["a"]
        assert result.metadata["dropped_below_min_score"] == 2

    def test_input_nodes_stay_unmodified(self):
        """Scores are written to copies — the shared graph state must not mutate."""
        reranker, _ = self._make([0.9])
        original = _node("a", score=0.123)
        reranker.rerank("q", [original])
        assert original.score == 0.123


# ---------------------------------------------------------------------------
# LLM reranker cost estimate (model-aware: GWDG-hosted models are free)
# ---------------------------------------------------------------------------

class TestLLMRerankerCostEstimate:
    def test_gwdg_models_cost_nothing(self):
        from src.llm.objects.rerankers.llm_reranker import _estimate_cost
        nodes = [_node("x" * 1000) for _ in range(10)]
        assert _estimate_cost("query", nodes, batch_size=5, model=Models.GEMMA4_31B) == 0.0
        assert _estimate_cost("query", nodes, batch_size=5, model=Models.LLAMA3) == 0.0

    def test_azure_models_have_positive_cost(self):
        from src.llm.objects.rerankers.llm_reranker import _estimate_cost
        nodes = [_node("x" * 1000) for _ in range(10)]
        azure = _estimate_cost("query", nodes, batch_size=5, model=Models.AZURE_FALLBACK)
        mini = _estimate_cost("query", nodes, batch_size=5, model=Models.MINI)
        assert azure > 0
        assert 0 < mini < azure


# ---------------------------------------------------------------------------
# get_reranker singleton cache re-applies configuration
# ---------------------------------------------------------------------------

class TestGetRerankerConfigure:
    def test_cached_instance_gets_fresh_top_n_and_min_score(self):
        from src.llm.tools.rerank import _reranker_instances, get_reranker
        dummy = _DummyReranker(top_n=5, min_score=0.0)
        _reranker_instances["llm"] = dummy
        try:
            returned = get_reranker("llm", top_n=2, min_score=0.4)
            assert returned is dummy
            assert dummy.top_n == 2
            assert dummy.min_score == 0.4
        finally:
            _reranker_instances.pop("llm", None)
