"""Regression test: reranker backends against the REAL index / REAL models.

Closes a gap the reranker analysis flagged explicitly: test_reranker_node.py
only exercises Passthrough + mocks, and evaluation/benchmark.py compares
backends for a human decision (`python -m evaluation.benchmark`) but is not a
pytest gate. This module runs a small, fixed slice of the ground-truth
dataset through each real backend and fails if NDCG@5 drops below a floor.

Scope:
  - BGE: real cross-encoder model (no mocks), no index access.
  - Azure Semantic: real backend AND a real round-trip to Azure AI Search —
    the "gegen einen echten Index" part of the gap.
  - LLM reranker is deliberately NOT covered here: it makes paid,
    rate-limited GWDG calls per query (see benchmark.py's --cooldown) which
    would make this test slow/flaky/costly on every run. Its error handling
    is covered by TestRerankerErrorHandling (mocked) in test_reranker_node.py;
    its quality is compared manually via evaluation/benchmark.py before a
    reranker decision.

Run with: pytest src/tests/llms/test_reranker_regression.py -v -m integration
Skip in fast CI: pytest -m "not integration"

Skips automatically if the ground-truth dataset (gitignored, generated via
`python -m evaluation.dataset.create_dataset`) is not present, and — for the
Azure Semantic test — if AZURE_SEARCH_ENDPOINT is not configured.
"""

import pytest

from evaluation import metrics as m
from evaluation.benchmark import DEFAULT_DATASET, _load_dataset, _nodes_from_chunks
from src.env import env

pytestmark = pytest.mark.integration

# Small, fixed slice: fast enough to run regularly, large enough to be a signal.
_SLICE_SIZE = 15
_TOP_N = 5

# Regression floors, not targets: measured 0.894 (BGE) / 0.832 (Azure
# Semantic) on this exact slice on 2026-07-09; floors sit ~15% below that to
# absorb normal variance (model/library updates, dataset re-generation) while
# still catching a real regression (e.g. a broken score mapping or a min_score
# default that silently drops everything). If a real evaluation/benchmark.py
# run shows a genuine, durable improvement, raise the floor to match — don't
# lower it just to make a real regression pass.
_MIN_NDCG_AT_5 = {
    "bge": 0.75,
    "azure_semantic": 0.70,
}


def _dataset_slice() -> list[dict]:
    if not DEFAULT_DATASET.exists():
        pytest.skip(
            f"Ground-truth dataset not found: {DEFAULT_DATASET} "
            "(run: python -m evaluation.dataset.create_dataset)"
        )
    return _load_dataset(DEFAULT_DATASET)[:_SLICE_SIZE]


def _mean_ndcg(reranker, records: list[dict]) -> float:
    scores = []
    for record in records:
        nodes = _nodes_from_chunks(record["retrieved_chunks"])
        label_map = {c["id"]: c["relevance"] for c in record["retrieved_chunks"]}
        ideal = [c["relevance"] for c in record["retrieved_chunks"]]

        result = reranker.rerank(
            query=record["query"],
            nodes=nodes,
            course_id=record.get("course_id"),
            module_id=record.get("module_id"),
        )

        relevances = [label_map.get(n.id_, 0) for n in result.nodes]
        scores.append(m.ndcg_at_k(relevances, _TOP_N, ideal_relevances=ideal))
    return sum(scores) / len(scores)


class TestBGERerankerAgainstRealModel:
    """No mocks: downloads/loads the real cross-encoder and scores real chunks."""

    def test_ndcg_does_not_regress(self):
        records = _dataset_slice()
        from src.llm.objects.rerankers.bge_reranker import BGEReranker

        reranker = BGEReranker(top_n=_TOP_N)
        ndcg = _mean_ndcg(reranker, records)

        floor = _MIN_NDCG_AT_5["bge"]
        assert ndcg >= floor, (
            f"BGE NDCG@{_TOP_N} dropped to {ndcg:.3f} (floor {floor}) "
            f"on the fixed {_SLICE_SIZE}-query slice"
        )


class TestAzureSemanticRerankerAgainstRealIndex:
    """Re-searches the real Azure AI Search index with query_type=semantic —
    the one path in the reranker suite that exercises a real backend end to
    end against a real index, not just mocked responses."""

    def test_ndcg_does_not_regress(self):
        if env.AZURE_SEARCH_ENDPOINT == "UNSET":
            pytest.skip("AZURE_SEARCH_ENDPOINT not configured")
        records = _dataset_slice()
        from src.llm.objects.rerankers.azure_semantic_reranker import AzureSemanticReranker

        reranker = AzureSemanticReranker(top_n=_TOP_N)
        ndcg = _mean_ndcg(reranker, records)

        floor = _MIN_NDCG_AT_5["azure_semantic"]
        assert ndcg >= floor, (
            f"Azure Semantic NDCG@{_TOP_N} dropped to {ndcg:.3f} (floor {floor}) "
            f"on the fixed {_SLICE_SIZE}-query slice"
        )
