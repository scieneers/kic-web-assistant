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

    def rerank(self, query, nodes, model=None, *, course_id=None, module_id=None, preranked=False):
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


# ---------------------------------------------------------------------------
# AzureSemanticReranker: request scope is explicit, never guessed from chunks
# ---------------------------------------------------------------------------

def _azure_instance(top_n: int = 3, min_score: float = 0.0, search_results: list | None = None):
    """Build an AzureSemanticReranker without hitting Azure (skips __init__)."""
    from src.llm.objects.rerankers.azure_semantic_reranker import AzureSemanticReranker

    instance = object.__new__(AzureSemanticReranker)
    instance.top_n = top_n
    instance.min_score = min_score
    instance._llm = MagicMock()
    instance._llm.get_embedder.return_value.get_query_embedding.return_value = [0.1, 0.2]
    instance._vector_db = MagicMock()
    instance._vector_db.hybrid_search.return_value = search_results or []
    return instance


def _node_with_scope_metadata(text: str, course_id: int, module_id: int | None = None) -> SerializableTextNode:
    """Chunk whose metadata carries a course/module it BELONGS TO or DESCRIBES —
    which must never be mistaken for the request scope."""
    metadata = {"course_id": course_id}
    if module_id is not None:
        metadata["module_id"] = module_id
    return SerializableTextNode(text=text, metadata=metadata, score=1.0)


class TestAzureSemanticScope:
    def test_unscoped_request_searches_drupal_despite_chunk_course_id(self):
        """Course-discovery case: top chunk is a Drupal course page carrying the
        course_id it describes — the re-search must still use the Drupal scope."""
        reranker = _azure_instance()
        nodes = [_node_with_scope_metadata("Kursseite zu ML", course_id=387)]
        reranker.rerank("Welche Kurse gibt es zu ML?", nodes, course_id=None, module_id=None)

        odata_filter = reranker._vector_db.hybrid_search.call_args.kwargs["odata_filter"]
        assert "source eq 'Drupal'" in odata_filter
        assert "course_id" not in odata_filter

    def test_course_scoped_request_is_not_narrowed_to_chunk_module(self):
        """Course-level case: first chunk is a module chunk — its module_id must
        not narrow the re-search to a single module."""
        reranker = _azure_instance()
        nodes = [_node_with_scope_metadata("Modulinhalt", course_id=63, module_id=2511)]
        reranker.rerank("Worum geht es in diesem Kurs?", nodes, course_id=63, module_id=None)

        odata_filter = reranker._vector_db.hybrid_search.call_args.kwargs["odata_filter"]
        assert "course_id eq 63" in odata_filter
        assert "module_id" not in odata_filter

    def test_module_scoped_request_filters_on_module(self):
        reranker = _azure_instance()
        nodes = [_node_with_scope_metadata("Modulinhalt", course_id=63, module_id=2511)]
        reranker.rerank("Worum geht es in diesem Modul?", nodes, course_id=63, module_id=2511)

        odata_filter = reranker._vector_db.hybrid_search.call_args.kwargs["odata_filter"]
        assert "course_id eq 63" in odata_filter
        assert "module_id eq 2511" in odata_filter

    def test_research_uses_semantic_query(self):
        reranker = _azure_instance()
        reranker.rerank("query", [_node("a")], course_id=None, module_id=None)
        assert reranker._vector_db.hybrid_search.call_args.kwargs["use_semantic"] is True


class TestAzureSemanticPreranked:
    """Integrated mode: retrieval already ranked semantically — no second search."""

    def test_no_search_call(self):
        reranker = _azure_instance()
        reranker.rerank("query", [_node("a"), _node("b")], preranked=True)
        reranker._vector_db.hybrid_search.assert_not_called()
        reranker._llm.get_embedder.assert_not_called()

    def test_sorts_and_cuts_to_top_n(self):
        reranker = _azure_instance(top_n=2)
        nodes = [_node("low", 1.0), _node("high", 3.5), _node("mid", 2.0)]
        result = reranker.rerank("query", nodes, preranked=True)
        assert [n.text for n in result.nodes] == ["high", "mid"]

    def test_min_score_applies_on_0_to_4_scale(self):
        reranker = _azure_instance(top_n=5, min_score=0.5)  # 0.5 normalized == 2.0 raw
        nodes = [_node("strong", 3.2), _node("weak", 1.2)]
        result = reranker.rerank("query", nodes, preranked=True)
        assert [n.text for n in result.nodes] == ["strong"]
        assert result.metadata["dropped_below_min_score"] == 1

    def test_cost_attributes_semantic_surcharge(self):
        reranker = _azure_instance()
        result = reranker.rerank("query", [_node("a")], preranked=True)
        assert result.estimated_cost_eur > 0
        assert result.metadata["integrated"] is True


# ---------------------------------------------------------------------------
# rerank_chunks node forwards request scope + preranked flag
# ---------------------------------------------------------------------------

class TestRerankNodeScopeForwarding:
    def test_scope_and_preranked_are_forwarded(self):
        nodes = [_node(f"doc {i}") for i in range(3)]
        state = _state(nodes)
        state["runtime_config"] = {"model": Models.AZURE_FALLBACK, "course_id": 63, "module_id": 2511}
        state["retrieval_semantic_ranked"] = True
        mock = MagicMock()
        mock.rerank.return_value = RerankResult(nodes=nodes[:2], latency_ms=1.0)
        with patch("src.llm.tools.rerank.get_reranker", return_value=mock):
            rerank_chunks(state)
        kwargs = mock.rerank.call_args.kwargs
        assert kwargs["course_id"] == 63
        assert kwargs["module_id"] == 2511
        assert kwargs["preranked"] is True

    def test_defaults_without_scope_and_flag(self):
        nodes = [_node("a"), _node("b")]
        state = _state(nodes)
        mock = MagicMock()
        mock.rerank.return_value = RerankResult(nodes=nodes, latency_ms=1.0)
        with patch("src.llm.tools.rerank.get_reranker", return_value=mock):
            rerank_chunks(state)
        kwargs = mock.rerank.call_args.kwargs
        assert kwargs["course_id"] is None
        assert kwargs["module_id"] is None
        assert kwargs["preranked"] is False

    def test_single_chunk_is_not_shortcut_in_integrated_mode(self):
        """Preranked nodes already carry their score — min_score must still apply
        even for a single chunk (weak lone hit → no-answer instead of weak answer)."""
        node = _node("weak solo", score=1.0)
        state = _state([node])
        state["retrieval_semantic_ranked"] = True
        mock = MagicMock()
        mock.rerank.return_value = RerankResult(nodes=[], latency_ms=0.1, metadata={"dropped_below_min_score": 1})
        with patch("src.llm.tools.rerank.get_reranker", return_value=mock):
            result = rerank_chunks(state)
        mock.rerank.assert_called_once()
        assert result["reranked"] == []

    def test_single_chunk_shortcut_still_active_without_preranked(self):
        node = _node("solo")
        with patch("src.llm.tools.rerank.get_reranker") as get_mock:
            result = rerank_chunks(_state([node]))
        get_mock.assert_not_called()
        assert result["reranked"] == [node]


# ---------------------------------------------------------------------------
# retrieve_chunks node: integrated semantic retrieval for the azure backend
# ---------------------------------------------------------------------------

class TestRetrieveNodeSemantic:
    def _retrieve_state(self, reranker_type: str) -> dict:
        return {
            "user_query": "Was ist KI?",
            "contextualized_query": "Was ist KI?",
            "runtime_config": {"model": Models.AZURE_FALLBACK, "course_id": None, "module_id": None},
            "system_config": {"retrieve_top_n": 10, "reranker_type": reranker_type},
        }

    def _run(self, reranker_type: str):
        from src.llm.tools import retrieve as retrieve_module
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_retriever.use_semantic = reranker_type == "azure_semantic"
        with patch.object(retrieve_module, "get_retriever", return_value=mock_retriever) as get_mock:
            result = retrieve_module.retrieve_chunks(self._retrieve_state(reranker_type))
        return result, get_mock

    def test_azure_semantic_enables_integrated_semantic_retrieval(self):
        result, get_mock = self._run("azure_semantic")
        assert get_mock.call_args.kwargs["use_semantic"] is True
        assert result["retrieval_semantic_ranked"] is True

    def test_other_backends_keep_plain_retrieval(self):
        result, get_mock = self._run("llm")
        assert get_mock.call_args.kwargs["use_semantic"] is False
        assert result["retrieval_semantic_ranked"] is False


# ---------------------------------------------------------------------------
# hybrid_search: semantic ranking gets a full candidate window
# ---------------------------------------------------------------------------

class TestHybridSearchSemanticWindow:
    def _search(self, use_semantic: bool, top: int = 5):
        from src.vectordb.azure_search import VectorDBAzureSearch
        instance = object.__new__(VectorDBAzureSearch)
        client = MagicMock()
        client.search.return_value = iter([])
        instance._client = lambda name=None: client
        instance.hybrid_search(
            query_text="q",
            query_vector=[0.1],
            index_name="idx",
            top=top,
            use_semantic=use_semantic,
        )
        return client.search.call_args.kwargs

    def test_semantic_feeds_full_rerank_window(self):
        kwargs = self._search(use_semantic=True, top=5)
        assert kwargs["vector_queries"][0].k_nearest_neighbors == 50
        assert kwargs["query_type"] == "semantic"

    def test_plain_search_keeps_candidate_factor(self):
        kwargs = self._search(use_semantic=False, top=5)
        assert kwargs["vector_queries"][0].k_nearest_neighbors == 15
        assert "query_type" not in kwargs


# ---------------------------------------------------------------------------
# Retriever result mapping: semantic reranker_score wins over RRF score
# ---------------------------------------------------------------------------

class TestRetrieverToNode:
    def test_prefers_reranker_score(self):
        from src.llm.objects.retriever import KiCampusRetriever
        node = KiCampusRetriever._to_node(
            {"id": "1", "text": "t", "@search.score": 0.03, "@search.reranker_score": 2.7}
        )
        assert node.score == 2.7

    def test_falls_back_to_rrf_score(self):
        from src.llm.objects.retriever import KiCampusRetriever
        node = KiCampusRetriever._to_node({"id": "1", "text": "t", "@search.score": 0.03})
        assert node.score == 0.03
