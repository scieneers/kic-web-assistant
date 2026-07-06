"""
Reranker benchmark — compares all available rerankers on the ground truth dataset.

Usage:
  python -m evaluation.benchmark

  Optional flags:
    --dataset PATH    Path to queries.jsonl (default: dataset/queries.jsonl)
    --top-n N         Number of results after reranking (default: 5)
    --runs N          Runs per query for stable latency measurements (default: 3)
    --output PATH     Save JSON results to file

Creates the dataset first if it does not exist:
  python -m evaluation.dataset.create_dataset
"""

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import List

from src.api.models.serializable_text_node import SerializableTextNode
from src.llm.objects.LLMs import Models
from src.llm.objects.rerankers.base import BaseReranker, RerankResult
from src.llm.objects.rerankers.llm_reranker import LLMReranker
from evaluation import metrics as m

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path(__file__).parent / "dataset" / "queries.jsonl"


def _load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run first: python -m evaluation.dataset.create_dataset"
        )
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _nodes_from_record(record: dict) -> List[SerializableTextNode]:
    return [
        SerializableTextNode(
            text=c["text"],
            metadata=c["metadata"],
            score=c.get("score"),
            id_=c.get("id"),
        )
        for c in record["retrieved_chunks"]
    ]


def _ground_truth(record: dict) -> List[int]:
    """Relevance labels in original retrieval order."""
    return [c["relevance"] for c in record["retrieved_chunks"]]


def _reranked_relevances(
    original_chunks: list[dict],
    reranked_nodes: List[SerializableTextNode],
) -> List[int]:
    """Map reranked node order back to relevance labels from ground truth."""
    id_to_relevance = {c["id"]: c["relevance"] for c in original_chunks}
    result = []
    for node in reranked_nodes:
        result.append(id_to_relevance.get(node.id_, 0))
    return result


def _build_rerankers(top_n: int) -> list[BaseReranker]:
    """Instantiate all available rerankers. Stubs are skipped with a warning."""
    from src.llm.objects.rerankers.passthrough_reranker import PassthroughReranker
    candidates = [
        ("no_rerank", PassthroughReranker),
        ("llm", LLMReranker),
    ]
    from src.llm.objects.rerankers.azure_semantic_reranker import AzureSemanticReranker
    candidates.append(("azure_semantic", AzureSemanticReranker))
    from src.llm.objects.rerankers.bge_reranker import BGEReranker, BGE_LARGE_MODEL, BGE_SMALL_MODEL
    candidates.append(("bge_large", lambda top_n: BGEReranker(top_n=top_n, model_name=BGE_LARGE_MODEL)))
    candidates.append(("bge_small", lambda top_n: BGEReranker(top_n=top_n, model_name=BGE_SMALL_MODEL)))

    active = []
    for name, factory in candidates:
        try:
            active.append(factory(top_n))
        except NotImplementedError as e:
            logger.warning("Skipping %s: %s", name, e)
    return active


def run_benchmark(
    dataset_path: Path = DEFAULT_DATASET,
    top_n: int = 5,
    runs: int = 3,
    model: Models = Models.AZURE_FALLBACK,
) -> list[dict]:
    records = _load_dataset(dataset_path)
    rerankers = _build_rerankers(top_n)

    if not rerankers:
        raise RuntimeError("No rerankers available to benchmark.")

    languages = sorted(set(r.get("language", "de") for r in records))
    multilingual = len(languages) > 1

    print(f"\nBenchmarking {len(rerankers)} reranker(s) on {len(records)} queries ({runs} runs each)")
    if multilingual:
        lang_counts = {lang: sum(1 for r in records if r.get("language", "de") == lang) for lang in languages}
        print(f"Languages: {', '.join(f'{lang}={n}' for lang, n in lang_counts.items())}")
    print()

    results = []

    for reranker in rerankers:
        per_query_metrics = []
        per_lang_metrics: dict[str, list] = {lang: [] for lang in languages}
        all_latencies = []
        all_costs = []
        skip = False

        for record in records:
            if skip:
                break
            nodes = _nodes_from_record(record)
            query = record["query"]
            lang = record.get("language", "de")
            query_latencies = []

            for run in range(runs):
                try:
                    result: RerankResult = reranker.rerank(query=query, nodes=nodes, model=model)
                except Exception as e:
                    print(f"  ⚠ {reranker.name} failed on '{query[:60]}': {e}")
                    skip = True
                    break
                query_latencies.append(result.latency_ms)
                if run == 0:
                    relevances = _reranked_relevances(record["retrieved_chunks"], result.nodes)
                    query_metrics = m.compute_all(relevances, k=top_n)
                    per_query_metrics.append(query_metrics)
                    per_lang_metrics[lang].append(query_metrics)
                    all_costs.append(result.estimated_cost_eur)

            all_latencies.extend(query_latencies)

        if skip or not per_query_metrics:
            print(f"  → {reranker.name} skipped (not enough data)\n")
            continue

        aggregated = m.aggregate(per_query_metrics)
        result_entry = {
            "reranker": reranker.name,
            **aggregated,
            "latency_p50_ms": round(statistics.median(all_latencies), 1),
            "latency_p95_ms": round(sorted(all_latencies)[int(len(all_latencies) * 0.95)], 1),
            "cost_per_1k_queries_eur": round(sum(all_costs) / len(all_costs) * 1000, 4),
            "by_language": {
                lang: m.aggregate(metrics)
                for lang, metrics in per_lang_metrics.items()
                if metrics
            },
        }
        results.append(result_entry)

    _print_table(results, top_n, label="Overall")
    if multilingual:
        for lang in languages:
            _print_table(results, top_n, label=f"Language: {lang}", language=lang)
    return results


def _print_table(results: list[dict], k: int, label: str = "Overall", language: str | None = None) -> None:
    headers = [
        "Reranker",
        f"NDCG@{k}",
        "MRR",
        f"P@{k}",
        "Latenz p50",
        "Latenz p95",
        "Kosten/1k €",
    ]
    col_w = [28, 10, 10, 10, 12, 12, 13]

    print(f"--- {label} ---")
    sep = "  ".join("-" * w for w in col_w)
    header = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(header)
    print(sep)

    for r in results:
        if language:
            lang_metrics = r.get("by_language", {}).get(language)
            if not lang_metrics:
                continue
            ndcg = lang_metrics.get(f"ndcg@{k}", 0)
            mrr = lang_metrics.get("mrr", 0)
            prec = lang_metrics.get(f"precision@{k}", 0)
        else:
            ndcg = r[f"ndcg@{k}"]
            mrr = r["mrr"]
            prec = r[f"precision@{k}"]

        row = [
            r["reranker"],
            f"{ndcg:.3f}",
            f"{mrr:.3f}",
            f"{prec:.3f}",
            f"{r['latency_p50_ms']}ms",
            f"{r['latency_p95_ms']}ms",
            f"{r['cost_per_1k_queries_eur']:.4f}",
        ]
        print("  ".join(v.ljust(w) for v, w in zip(row, col_w)))
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Reranker benchmark")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None, help="Save results as JSON")
    args = parser.parse_args()

    results = run_benchmark(dataset_path=args.dataset, top_n=args.top_n, runs=args.runs)

    if args.output:
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"Results saved to {args.output}")
