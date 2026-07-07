"""
Retrieval evaluation metrics: NDCG@k, MRR, Precision@k.

Relevance scale (graded):
  0 = not relevant
  1 = partially relevant
  2 = highly relevant

All functions accept a list of relevance scores in ranked order
(index 0 = top result) and return a float in [0, 1].
"""

import math
from typing import List


def dcg_at_k(relevances: List[int], k: int) -> float:
    """Discounted Cumulative Gain at position k."""
    return sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(relevances[:k])
    )


def ndcg_at_k(relevances: List[int], k: int, ideal_relevances: List[int] | None = None) -> float:
    """Normalized DCG at k. Returns 0 if no relevant documents exist.

    ideal_relevances: relevance labels of the FULL candidate pool the ranker
    could have chosen from. Without it, the ideal is computed from the ranked
    list itself — which overrates rankers that return few (but ordered) chunks
    while better candidates existed in the pool.
    """
    ideal_pool = ideal_relevances if ideal_relevances is not None else relevances
    ideal_dcg = dcg_at_k(sorted(ideal_pool, reverse=True), k)
    if ideal_dcg == 0:
        return 0.0
    return min(1.0, dcg_at_k(relevances, k) / ideal_dcg)


def mrr(relevances: List[int]) -> float:
    """Mean Reciprocal Rank — reciprocal of the rank of the first relevant doc."""
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(relevances: List[int], k: int) -> float:
    """Fraction of top-k results that are relevant (relevance > 0)."""
    if k == 0:
        return 0.0
    return sum(1 for r in relevances[:k] if r > 0) / k


def compute_all(relevances: List[int], k: int = 5, ideal_relevances: List[int] | None = None) -> dict:
    """Compute all metrics for a single ranked list.

    ideal_relevances: labels of the full candidate pool (see ndcg_at_k).
    """
    return {
        f"ndcg@{k}": ndcg_at_k(relevances, k, ideal_relevances=ideal_relevances),
        "mrr": mrr(relevances),
        f"precision@{k}": precision_at_k(relevances, k),
    }


def aggregate(per_query_metrics: List[dict]) -> dict:
    """Average metrics across all queries."""
    if not per_query_metrics:
        return {}
    keys = per_query_metrics[0].keys()
    return {
        key: sum(m[key] for m in per_query_metrics) / len(per_query_metrics)
        for key in keys
    }
