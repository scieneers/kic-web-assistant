"""
Reranker benchmark — compares all available rerankers on the ground truth dataset.

Fairness design:
  - The REQUEST scope (course_id/module_id) comes from the dataset record and is
    passed explicitly to index-querying backends (Azure Semantic). It is never
    guessed from chunk metadata — Drupal course pages carry the course_id of the
    course they *describe*, which is not the scope the user asked in.
  - Per query, ALL systems run first; every returned-but-unlabeled chunk is
    LLM-judged once (disk-cached); then metrics for ALL systems are computed
    against the SAME union candidate pool (pool labels ∪ labels of every chunk
    any system returned) — TREC-style pooling, so the NDCG/recall denominators
    are identical across systems.
  - Azure Semantic represents the INTEGRATED production setup (semantic ranking
    inside the retrieval call). Its latency column shows the full semantic
    search round-trip; the run additionally times a plain (non-semantic) search
    per query so the report can show the marginal semantic overhead — in
    production the semantic call REPLACES retrieval, it does not add to it.

Usage:
  python -m evaluation.benchmark

  Optional flags:
    --dataset PATH          Path to queries.jsonl (default: dataset/queries.jsonl)
    --top-n N               Number of results after reranking (default: 5)
    --runs N                Runs per query for stable latency measurements (default: 3)
    --pool-sizes 10,30      Candidate-pool sizes to compare (default: full pool).
                            Requires a dataset created with --n-chunks >= max pool size.
                            Note: Azure Semantic re-searches the index — pool size
                            does not constrain it (that is the integrated setup).
    --no-judge-unlabeled    Skip on-the-fly judging of out-of-pool chunks
                            (they then count as relevance 0 and are excluded
                            from the union ideal — affected systems are underrated).
    --no-latency-baseline   Skip the plain-search timing used to compute Azure
                            Semantic's marginal latency.
    --output PATH           Save JSON results to file

Creates the dataset first if it does not exist:
  python -m evaluation.dataset.create_dataset
"""

import argparse
import hashlib
import json
import logging
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from src.api.models.serializable_text_node import SerializableTextNode
from src.env import env
from src.llm.objects.LLMs import LLM, Models
from src.llm.objects.rerankers.base import BaseReranker, RerankResult
from src.llm.objects.rerankers.llm_reranker import LLMReranker
from evaluation import metrics as m
from evaluation.dataset.create_dataset import judge_relevance

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path(__file__).parent / "dataset" / "queries.jsonl"
DEFAULT_JUDGE_CACHE = Path(__file__).parent / "dataset" / "judge_cache.jsonl"


class UnlabeledChunkJudge:
    """LLM-judges chunks that a reranker returned but that carry no ground-truth
    label (Azure Semantic re-searches the whole index and can surface documents
    outside the labeled pool — without judging they would unfairly count as 0).

    Judgments are cached on disk (judge_cache.jsonl) so repeated benchmark runs
    do not re-judge the same (query, chunk) pairs. On LLM failure the judge
    disables itself and unlabeled chunks fall back to relevance 0 (with a warning).
    """

    def __init__(self, cache_path: Path = DEFAULT_JUDGE_CACHE, model: Models = Models.AZURE_FALLBACK, enabled: bool = True):
        self.enabled = enabled
        self.cache_path = cache_path
        self.model = model
        self.judged_count = 0
        self.cache_hits = 0
        self._llm: LLM | None = None
        self._cache: dict[str, int] = {}
        if enabled and cache_path.exists():
            with cache_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        self._cache[entry["key"]] = entry["relevance"]

    @staticmethod
    def _key(query: str, node: SerializableTextNode) -> str:
        chunk_id = node.id_ or hashlib.sha1((node.text or "").encode()).hexdigest()[:16]
        return hashlib.sha1(f"{query}|{chunk_id}".encode()).hexdigest()

    def relevance(self, query: str, node: SerializableTextNode) -> Optional[int]:
        """Return a cached or freshly judged relevance, or None if judging is off/failed."""
        if not self.enabled:
            return None
        key = self._key(query, node)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        try:
            if self._llm is None:
                self._llm = LLM()
            rel = judge_relevance(self._llm, query, node.text or "", self.model)
        except Exception as e:
            logger.warning("On-the-fly judging failed (%s) — unlabeled chunks count as 0 from now on", e)
            self.enabled = False
            return None
        self._cache[key] = rel
        self.judged_count += 1
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "query": query, "chunk_id": node.id_, "relevance": rel}, ensure_ascii=False) + "\n")
        return rel


def _load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run first: python -m evaluation.dataset.create_dataset"
        )
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _nodes_from_chunks(chunks: list[dict]) -> List[SerializableTextNode]:
    return [
        SerializableTextNode(
            text=c["text"],
            metadata=c["metadata"],
            score=c.get("score"),
            id_=c.get("id"),
        )
        for c in chunks
    ]


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


def _plain_search_latency_ms(azure_reranker, query: str, course_id, module_id) -> float:
    """Time one plain (non-semantic) hybrid search with the same scope.

    Used to compute Azure Semantic's MARGINAL latency: in the integrated setup
    the semantic call replaces this plain retrieval call, so the honest latency
    cost of the semantic ranker is (semantic search − plain search), not the
    full round-trip. Embedding happens outside the timer — same convention as
    AzureSemanticReranker.
    """
    from src.llm.objects.retriever import _build_odata_filter

    embedding = azure_reranker._llm.get_embedder().get_query_embedding(query)
    odata_filter = _build_odata_filter(course_id, module_id)
    t0 = time.perf_counter()
    azure_reranker._vector_db.hybrid_search(
        query_text=query,
        query_vector=embedding,
        index_name=env.AZURE_SEARCH_INDEX,
        odata_filter=odata_filter,
        top=azure_reranker.top_n,
    )
    return (time.perf_counter() - t0) * 1000


def run_benchmark(
    dataset_path: Path = DEFAULT_DATASET,
    top_n: int = 5,
    runs: int = 3,
    model: Models = Models.AZURE_FALLBACK,
    judge_model: Models = Models.AZURE_FALLBACK,
    pool_sizes: list[int] | None = None,
    judge_unlabeled: bool = True,
    latency_baseline: bool = True,
) -> list[dict]:
    """model drives the LLM RERANKER (set it to the production model for a
    production-near comparison); judge_model drives the on-the-fly labeling and
    must stay consistent with the dataset-creation judge — do not couple the
    two, or a weaker reranker model would also degrade the ground truth."""
    records = _load_dataset(dataset_path)
    rerankers = _build_rerankers(top_n)

    if not rerankers:
        raise RuntimeError("No rerankers available to benchmark.")

    judge = UnlabeledChunkJudge(model=judge_model, enabled=judge_unlabeled)
    pools: list[int | None] = list(pool_sizes) if pool_sizes else [None]

    languages = sorted(set(r.get("language", "de") for r in records))
    kinds = sorted(set(r.get("kind", "general") for r in records))
    multilingual = len(languages) > 1
    max_pool = max(len(r["retrieved_chunks"]) for r in records)
    azure_rerankers = [r for r in rerankers if r.name == "Azure Semantic"]

    print(f"\nBenchmarking {len(rerankers)} reranker(s) on {len(records)} queries ({runs} runs each)")
    print(f"Dataset pool: up to {max_pool} labeled chunks per query")
    print("Scoring: per query all systems run first, out-of-pool chunks are judged, "
          "then all systems are scored against the same union candidate pool.")
    if pool_sizes:
        oversized = [p for p in pool_sizes if p > max_pool]
        if oversized:
            print(f"⚠ pool sizes {oversized} exceed the dataset pool ({max_pool}) — "
                  f"recreate the dataset with --n-chunks >= {max(oversized)} for a real comparison")
        print("Note: Azure Semantic re-searches the whole index — the pool size does not "
              "constrain it (this measures the integrated setup).")
    if multilingual:
        lang_counts = {lang: sum(1 for r in records if r.get("language", "de") == lang) for lang in languages}
        print(f"Languages: {', '.join(f'{lang}={n}' for lang, n in lang_counts.items())}")
    if len(kinds) > 1:
        print(f"Query kinds: {', '.join(kinds)}")
    print()

    # Per (pool_size, reranker name) accumulation
    def _empty_stats():
        return {
            "per_query": [],
            "by_language": defaultdict(list),
            "by_kind": defaultdict(list),
            "latencies": [],
            "costs": [],
            "unlabeled_as_zero": 0,
            "errors": 0,
        }

    stats: dict[tuple, dict] = {}
    plain_search_latencies: list[float] = []
    semantic_search_latencies: list[float] = []

    for pool_index, pool_size in enumerate(pools):
        pool_label = f"pool={pool_size}" if pool_size else "pool=full"

        for record in tqdm(records, desc=f"Queries ({pool_label})", unit="query"):
            query = record["query"]
            language = record.get("language", "de")
            kind = record.get("kind", "general")
            course_id = record.get("course_id")
            module_id = record.get("module_id")

            pool_chunks = record["retrieved_chunks"][:pool_size] if pool_size else record["retrieved_chunks"]
            nodes = _nodes_from_chunks(pool_chunks)
            pool_ids = {c["id"] for c in pool_chunks}
            pool_labels = [c["relevance"] for c in pool_chunks]
            # Full-record labels: valid judgments even for chunks sliced out of
            # the experiment pool (a re-searching system may still return them).
            label_map = {c["id"]: c["relevance"] for c in record["retrieved_chunks"]}

            # --- Phase 1: run all systems on this query -------------------
            returned: dict[str, RerankResult] = {}
            for reranker in rerankers:
                key = (pool_size, reranker.name)
                stats.setdefault(key, _empty_stats())
                for run in range(runs):
                    try:
                        result = reranker.rerank(
                            query=query,
                            nodes=nodes,
                            model=model,
                            course_id=course_id,
                            module_id=module_id,
                        )
                    except Exception as e:
                        logger.warning("%s failed on %r: %s", reranker.name, query[:60], e)
                        stats[key]["errors"] += 1
                        break
                    stats[key]["latencies"].append(result.latency_ms)
                    if run == 0:
                        returned[reranker.name] = result
                        stats[key]["costs"].append(result.estimated_cost_eur)

            # Marginal-latency baseline for Azure Semantic (once per query,
            # first pool iteration only — scope-identical plain search).
            if latency_baseline and azure_rerankers and pool_index == 0 and "Azure Semantic" in returned:
                try:
                    plain_search_latencies.append(
                        _plain_search_latency_ms(azure_rerankers[0], query, course_id, module_id)
                    )
                    semantic_search_latencies.append(returned["Azure Semantic"].latency_ms)
                except Exception as e:
                    logger.warning("Plain-search latency baseline failed on %r: %s", query[:60], e)

            # --- Phase 2: build the union candidate pool ------------------
            # Union = experiment pool ∪ every chunk any system returned.
            # Out-of-pool chunks get their record label if available (sliced
            # pools), otherwise an on-the-fly LLM judgment. Unjudgeable chunks
            # are excluded from the ideal and score 0 for the system that
            # returned them (tracked in unlabeled_as_zero).
            union_extra: dict[str, int] = {}
            for result in returned.values():
                for node in result.nodes:
                    nid = node.id_
                    if nid in pool_ids or nid in union_extra:
                        continue
                    if nid in label_map:
                        union_extra[nid] = label_map[nid]
                        continue
                    rel = judge.relevance(query, node)
                    if rel is not None:
                        union_extra[nid] = rel
            union_pool = pool_labels + list(union_extra.values())
            total_relevant = sum(1 for r in union_pool if r > 0)

            # --- Phase 3: score every system against the union pool -------
            for name, result in returned.items():
                key = (pool_size, name)
                relevances: List[int] = []
                for node in result.nodes:
                    rel = label_map.get(node.id_)
                    if rel is None:
                        rel = union_extra.get(node.id_)
                    if rel is None:
                        rel = 0
                        stats[key]["unlabeled_as_zero"] += 1
                    relevances.append(rel)
                query_metrics = m.compute_all(
                    relevances, k=top_n, ideal_relevances=union_pool, total_relevant=total_relevant
                )
                stats[key]["per_query"].append(query_metrics)
                stats[key]["by_language"][language].append(query_metrics)
                stats[key]["by_kind"][kind].append(query_metrics)

    # --- Aggregate ------------------------------------------------------
    results = []
    for pool_size in pools:
        for reranker in rerankers:
            key = (pool_size, reranker.name)
            s = stats.get(key)
            if not s or not s["per_query"]:
                print(f"  → {reranker.name} (pool={pool_size or max_pool}) skipped (no data)")
                continue
            aggregated = m.aggregate(s["per_query"])
            latencies = sorted(s["latencies"])
            results.append({
                "reranker": reranker.name,
                "pool_size": pool_size or max_pool,
                **aggregated,
                "latency_p50_ms": round(statistics.median(latencies), 1),
                "latency_p95_ms": round(latencies[int(len(latencies) * 0.95)], 1),
                "cost_per_1k_queries_eur": round(sum(s["costs"]) / len(s["costs"]) * 1000, 4) if s["costs"] else 0.0,
                "unlabeled_as_zero": s["unlabeled_as_zero"],
                "errors": s["errors"],
                "by_language": {lang: m.aggregate(v) for lang, v in s["by_language"].items() if v},
                "by_kind": {kind: m.aggregate(v) for kind, v in s["by_kind"].items() if v},
            })

    # --- Report -----------------------------------------------------------
    for pool_size in pools:
        p = pool_size or max_pool
        pool_results = [r for r in results if r["pool_size"] == p]
        suffix = f" — Pool={p}" if len(pools) > 1 else ""
        _print_table(pool_results, top_n, label=f"Overall{suffix}")
        if multilingual:
            for lang in languages:
                _print_table(pool_results, top_n, label=f"Language: {lang}{suffix}", language=lang)
        if len(kinds) > 1:
            _print_kind_matrix(pool_results, top_n, kinds, label=f"NDCG@{top_n} nach Query-Kind{suffix}")

    if plain_search_latencies and semantic_search_latencies:
        plain_p50 = statistics.median(plain_search_latencies)
        semantic_p50 = statistics.median(semantic_search_latencies)
        print("Azure Semantic — Latenz-Einordnung (integriertes Setup):")
        print(f"  volle semantische Suche p50: {semantic_p50:.0f} ms")
        print(f"  plain Hybrid-Suche p50 (gleicher Scope): {plain_p50:.0f} ms")
        print(f"  → marginale Semantic-Latenz ≈ {semantic_p50 - plain_p50:.0f} ms "
              "(der semantische Call ERSETZT das Retrieval, er kommt nicht dazu)\n")

    if judge.judged_count or judge.cache_hits:
        print(f"On-the-fly judging: {judge.judged_count} new, {judge.cache_hits} from cache ({judge.cache_path})")
    zero_fallbacks = sum(r.get("unlabeled_as_zero", 0) for r in results)
    if zero_fallbacks:
        print(f"⚠ {zero_fallbacks} out-of-pool chunks counted as relevance 0 (judging disabled/failed) — "
              "affected rerankers are underrated")
    total_errors = sum(r.get("errors", 0) for r in results)
    if total_errors:
        print(f"⚠ {total_errors} rerank call(s) failed — affected (reranker, query) pairs were skipped")
    return results


def _print_table(results: list[dict], k: int, label: str = "Overall", language: str | None = None) -> None:
    headers = [
        "Reranker",
        f"NDCG@{k}",
        f"Recall@{k}",
        "MRR",
        f"P@{k}",
        "Latenz p50",
        "Latenz p95",
        "Kosten/1k €",
    ]
    col_w = [28, 9, 10, 8, 8, 12, 12, 12]

    print(f"--- {label} ---")
    sep = "  ".join("-" * w for w in col_w)
    header = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(header)
    print(sep)

    for r in results:
        if language:
            metrics = r.get("by_language", {}).get(language)
            if not metrics:
                continue
        else:
            metrics = r

        row = [
            r["reranker"],
            f"{metrics.get(f'ndcg@{k}', 0):.3f}",
            f"{metrics.get(f'recall@{k}', 0):.3f}",
            f"{metrics.get('mrr', 0):.3f}",
            f"{metrics.get(f'precision@{k}', 0):.3f}",
            f"{r['latency_p50_ms']}ms",
            f"{r['latency_p95_ms']}ms",
            f"{r['cost_per_1k_queries_eur']:.4f}",
        ]
        print("  ".join(v.ljust(w) for v, w in zip(row, col_w)))
    print()


def _print_kind_matrix(results: list[dict], k: int, kinds: list[str], label: str) -> None:
    """Compact matrix: rows = rerankers, columns = query kinds, cells = NDCG@k.

    Makes scope-sensitive regressions visible at a glance (e.g. a re-searching
    backend degrading on course-discovery queries but not on module-level ones).
    """
    name_w = 28
    col_w = max(10, max((len(kind) for kind in kinds), default=10) + 1)
    print(f"--- {label} ---")
    print("Reranker".ljust(name_w) + "".join(kind.ljust(col_w) for kind in kinds))
    print("-" * (name_w + col_w * len(kinds)))
    for r in results:
        cells = []
        for kind in kinds:
            kind_metrics = r.get("by_kind", {}).get(kind)
            cells.append(f"{kind_metrics.get(f'ndcg@{k}', 0):.3f}" if kind_metrics else "—")
        print(r["reranker"].ljust(name_w) + "".join(c.ljust(col_w) for c in cells))
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Reranker benchmark")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--pool-sizes", type=str, default=None,
        help="Comma-separated candidate-pool sizes to compare, e.g. '10,30'. "
             "Dataset must contain at least that many chunks per query (create with --n-chunks).",
    )
    parser.add_argument(
        "--no-judge-unlabeled", action="store_true",
        help="Do not LLM-judge out-of-pool chunks (they count as relevance 0).",
    )
    parser.add_argument(
        "--no-latency-baseline", action="store_true",
        help="Skip the plain-search timing used for Azure Semantic's marginal latency.",
    )
    parser.add_argument("--model", type=str, default="Azure-Fallback", choices=[m_.value for m_ in Models],
                        help="LLM model for the LLM RERANKER — set to the production model "
                             "(e.g. Gemma4) for a production-near comparison")
    parser.add_argument("--judge-model", type=str, default="Azure-Fallback", choices=[m_.value for m_ in Models],
                        help="LLM model for on-the-fly judging of unlabeled chunks. Keep "
                             "consistent with the dataset-creation judge (default: Azure-Fallback)")
    parser.add_argument("--output", type=Path, default=None, help="Save results as JSON")
    args = parser.parse_args()

    pool_sizes = [int(p) for p in args.pool_sizes.split(",")] if args.pool_sizes else None

    results = run_benchmark(
        dataset_path=args.dataset,
        top_n=args.top_n,
        runs=args.runs,
        model=Models(args.model),
        judge_model=Models(args.judge_model),
        pool_sizes=pool_sizes,
        judge_unlabeled=not args.no_judge_unlabeled,
        latency_baseline=not args.no_latency_baseline,
    )

    if args.output:
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"Results saved to {args.output}")
