"""Unit tests for evaluation/metrics.py — NDCG ideal-pool handling."""

import pytest

from evaluation.metrics import aggregate, compute_all, mrr, ndcg_at_k, precision_at_k, recall_at_k


class TestNdcgIdealPool:
    def test_without_ideal_pool_uses_ranked_list_itself(self):
        # Perfectly ordered list scores 1.0 against its own ideal.
        assert ndcg_at_k([2, 1, 0], k=3) == pytest.approx(1.0)

    def test_ideal_pool_penalizes_dropped_relevant_chunks(self):
        # Ranker returned only mediocre chunks although the pool held a highly
        # relevant one — without the pool ideal this would score 1.0.
        without_pool = ndcg_at_k([1, 1], k=5)
        with_pool = ndcg_at_k([1, 1], k=5, ideal_relevances=[2, 1, 1, 0])
        assert without_pool == pytest.approx(1.0)
        assert with_pool < 1.0

    def test_perfect_selection_from_pool_scores_one(self):
        assert ndcg_at_k([2, 1], k=2, ideal_relevances=[2, 1, 0, 0]) == pytest.approx(1.0)

    def test_result_is_clamped_to_one(self):
        # Ranked list better than the (incomplete) ideal pool must not exceed 1.
        assert ndcg_at_k([2, 2], k=2, ideal_relevances=[1, 0]) == 1.0

    def test_no_relevant_documents_returns_zero(self):
        assert ndcg_at_k([0, 0], k=2, ideal_relevances=[0, 0, 0]) == 0.0


class TestBasicMetrics:
    def test_mrr_first_relevant_at_rank_two(self):
        assert mrr([0, 2, 1]) == pytest.approx(0.5)

    def test_precision_counts_relevant_fraction(self):
        assert precision_at_k([2, 0, 1, 0], k=4) == pytest.approx(0.5)

    def test_compute_all_passes_ideal_pool_through(self):
        result = compute_all([1, 1], k=5, ideal_relevances=[2, 1, 1])
        assert result["ndcg@5"] < 1.0
        assert result["mrr"] == 1.0


class TestRecall:
    def test_counts_found_fraction_of_all_relevant(self):
        # 2 of 3 relevant docs in the union pool made it into the top-k.
        assert recall_at_k([2, 0, 1], k=3, total_relevant=3) == pytest.approx(2 / 3)

    def test_undefined_without_relevant_docs(self):
        # No-answer queries: nothing relevant exists — recall is undefined, not 0.
        assert recall_at_k([0, 0], k=2, total_relevant=0) is None

    def test_compute_all_includes_recall_only_when_pool_size_given(self):
        with_recall = compute_all([1, 0], k=2, total_relevant=2)
        without_recall = compute_all([1, 0], k=2)
        assert with_recall["recall@2"] == pytest.approx(0.5)
        assert "recall@2" not in without_recall

    def test_aggregate_skips_undefined_recall(self):
        # One query with defined recall, one no-answer query (None) — the
        # average must only cover the defined one instead of crashing or
        # counting None as 0.
        merged = aggregate([{"recall@5": 0.8, "mrr": 1.0}, {"recall@5": None, "mrr": 0.0}])
        assert merged["recall@5"] == pytest.approx(0.8)
        assert merged["mrr"] == pytest.approx(0.5)
