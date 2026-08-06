"""
Ground truth dataset creator for reranker evaluation.

Workflow:
  1. Collect queries: built-in SAMPLE_QUERIES (curated core), --queries-file,
     or --generate (curated core + index-generated level questions in one run).
  2. For each query, retrieve the top --n-chunks candidate chunks via the real
     retriever (with the query's course_id/module_id scope).
  3. For each (query, chunk) pair, ask an LLM to rate relevance: 0 / 1 / 2.
  4. Save results to evaluation/dataset/queries.jsonl (flushed per record).

Recommended one-shot (after an ingest):
  python -m evaluation.dataset.create_dataset --generate --regenerate

Crashed / partially failed? Resume without re-judging what is already labeled
(the generated questions are cached in generated_queries.json, so the resumed
run labels the same question set):
  python -m evaluation.dataset.create_dataset --generate --append

  Optional flags:
    --output PATH        Override output file path
    --limit N            Only process first N queries (useful for testing)
    --model MODEL        LLM model for judging (default: Azure-Fallback)
    --n-chunks N         Candidate pool size per query (default: 30) — benchmark
                         pool-size experiments need at least the largest pool
    --queries-file PATH  Use generated queries (see generate_questions.py)
                         instead of the built-in SAMPLE_QUERIES
    --generate           SAMPLE_QUERIES + index-generated questions in one run
    --regenerate         With --generate: force fresh question generation
    --append             Only add queries not yet in the dataset (resume mode)
"""

import argparse
import json
import logging
import time
from pathlib import Path

from tqdm import tqdm

from src.llm.objects.LLMs import LLM, Models
from src.llm.objects.retriever import KiCampusRetriever

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path(__file__).parent / "queries.jsonl"

# Kuratierter Kern (~36 Fragen): nur Frage-Typen, die sich NICHT sinnvoll aus
# dem Index generieren lassen (Plattform/Orga, Kurzsuchen, Umgangssprache,
# No-Answer-Kalibrierung) plus wenige klassische Konzept-/Vergleichsfragen als
# stabile Vergleichsbasis über Dataset-Versionen hinweg.
#
# Kurs-/Modul-Level-Fragen mit echten IDs gehören bewusst NICHT hierher — sie
# veralten mit jedem Re-Ingest. Sie kommen aus generate_questions.py, das sie
# mit aktuellen course_id/module_id direkt aus dem Index erzeugt:
#   python -m evaluation.dataset.generate_questions
#   python -m evaluation.dataset.create_dataset
#   python -m evaluation.dataset.create_dataset --queries-file evaluation/dataset/generated_queries.json --append
#
# Each entry:
#   query              : the user query (required)
#   language           : question language (default de)
#   kind               : category for the benchmark's per-kind breakdown
#   expected_no_answer : True for no-answer / min_score calibration queries
SAMPLE_QUERIES: list[dict] = [
    # --- Kursentdeckung (Drupal, unscoped) ---
    {"query": "Welche Kurse gibt es zum Thema maschinelles Lernen?", "language": "de", "kind": "discovery"},
    {"query": "Welche Kurse gibt es zum Thema Ethik?", "language": "de", "kind": "discovery"},
    {"query": "gibts hier auch was für totale anfänger ohne mathe?", "language": "de", "kind": "discovery"},
    {"query": "What courses are available on machine learning?", "language": "en", "kind": "discovery"},

    # --- Konzepterklärung (stabile Basis über Dataset-Versionen) ---
    {"query": "Erkläre mir was ein neuronales Netz ist.", "language": "de", "kind": "concept"},
    {"query": "Was ist ein Large Language Model?", "language": "de", "kind": "concept"},
    {"query": "Was versteht man unter Bias in KI-Systemen?", "language": "de", "kind": "concept"},
    {"query": "What is overfitting and how do you prevent it?", "language": "en", "kind": "concept"},

    # --- Technische Vertiefung ---
    {"query": "Wie funktioniert Backpropagation beim Deep Learning Training?", "language": "de", "kind": "technical"},
    {"query": "Wie funktioniert der Attention-Mechanismus in Transformern?", "language": "de", "kind": "technical"},
    {"query": "Was ist Retrieval-Augmented Generation?", "language": "de", "kind": "technical"},

    # --- Vergleichs-/Multi-Konzept-Fragen (Ex-multi_hop-Fragetyp) ---
    {"query": "Was ist der Unterschied zwischen KI, maschinellem Lernen und Deep Learning?", "language": "de", "kind": "comparison"},
    {"query": "Wie hängen Bias, Varianz und Overfitting zusammen?", "language": "de", "kind": "comparison"},
    {"query": "How do CNNs and RNNs differ and when would I use each?", "language": "en", "kind": "comparison"},

    # --- Kurze/vage Suchanfragen (realistisches Nutzerverhalten) ---
    {"query": "KI Ethik", "language": "de", "kind": "short"},
    {"query": "zertifikat", "language": "de", "kind": "short"},
    {"query": "prompt engineering", "language": "de", "kind": "short"},

    # --- Umgangssprachlich / mit Tippfehlern ---
    {"query": "was is n transformer modell?", "language": "de", "kind": "colloquial"},
    {"query": "kannst du mir was über machine lerning erzählen", "language": "de", "kind": "colloquial"},
    {"query": "i dont get gradient descent at all", "language": "en", "kind": "colloquial"},

    # --- Plattform-/Orga-Fragen (Drupal) ---
    {"query": "Wie bekomme ich ein Zertifikat auf dem KI-Campus?", "language": "de", "kind": "platform"},
    {"query": "Ist der KI-Campus kostenlos?", "language": "de", "kind": "platform"},
    {"query": "Wer steht hinter dem KI-Campus und wer finanziert ihn?", "language": "de", "kind": "platform"},
    {"query": "Is KI-Campus free to use?", "language": "en", "kind": "platform"},

    # --- Domänenspezifisch ---
    {"query": "Wie wird KI in der Medizin eingesetzt?", "language": "de", "kind": "domain"},
    {"query": "Welche KI-Kompetenzen brauchen Lehrkräfte an Schulen?", "language": "de", "kind": "domain"},

    # --- Weitere Sprachen (bewusst unscoped → unabhängig von Index-IDs;
    #     Inhalte sind DE/EN → misst Cross-Lingual-Retrieval) ---
    {"query": "¿Qué es el aprendizaje automático?", "language": "es", "kind": "concept"},
    {"query": "Comment fonctionne un réseau de neurones ?", "language": "fr", "kind": "concept"},
    {"query": "Che cos'è il deep learning?", "language": "it", "kind": "concept"},
    {"query": "Yapay zeka etiği neden önemlidir?", "language": "tr", "kind": "concept"},

    # --- No-Answer-Kalibrierung: plausibel, aber (sehr wahrscheinlich) nicht
    # durch die Wissensbasis gedeckt → Ground Truth für MIN_RERANKER_SCORE ---
    {"query": "Wie viel kostet ein ChatGPT-Plus-Abo?", "language": "de", "kind": "negative", "expected_no_answer": True},
    {"query": "Welche KI-Aktien soll ich kaufen?", "language": "de", "kind": "negative", "expected_no_answer": True},
    {"query": "Wie installiere ich CUDA-Treiber unter Windows 11?", "language": "de", "kind": "negative", "expected_no_answer": True},
    {"query": "Was hat der Bundestag letzte Woche zur KI-Regulierung beschlossen?", "language": "de", "kind": "negative", "expected_no_answer": True},
    {"query": "How much does a ChatGPT Plus subscription cost?", "language": "en", "kind": "negative", "expected_no_answer": True},
    {"query": "Which AI stocks should I buy right now?", "language": "en", "kind": "negative", "expected_no_answer": True},
]

JUDGE_PROMPT = """Du bist ein Experte für Information Retrieval Evaluation.

Bewerte die Relevanz des folgenden Dokuments für die gegebene Suchanfrage.

Suchanfrage: {query}

Dokument:
{chunk_text}

Bewertungsskala:
2 = Hoch relevant: Das Dokument beantwortet die Suchanfrage direkt und vollständig.
1 = Teilweise relevant: Das Dokument enthält verwandte Informationen, beantwortet die Anfrage aber nur teilweise.
0 = Nicht relevant: Das Dokument enthält keine nützlichen Informationen für die Suchanfrage.

Antworte ausschließlich mit einer einzigen Zahl: 0, 1 oder 2."""


def judge_relevance(llm: LLM, query: str, chunk_text: str, model: Models) -> int:
    """Ask the LLM to rate chunk relevance for a query. Returns 0, 1, or 2."""
    prompt = JUDGE_PROMPT.format(query=query, chunk_text=chunk_text[:2000])
    response = llm.chat(query=prompt, chat_history=[], model=model, system_prompt="")
    content = (response.content or "").strip()
    if content in ("0", "1", "2"):
        return int(content)
    # Fallback: try to extract first digit
    for char in content:
        if char in ("0", "1", "2"):
            return int(char)
    logger.warning("Could not parse judge response %r — defaulting to 0", content)
    return 0


def load_queries_file(path: Path) -> list[dict]:
    """Load queries from a JSON file (list of dicts with at least a 'query' key),
    e.g. produced by evaluation/dataset/generate_questions.py."""
    with path.open(encoding="utf-8") as f:
        queries = json.load(f)
    if not isinstance(queries, list) or not all(isinstance(q, dict) and "query" in q for q in queries):
        raise ValueError(f"{path} must contain a JSON list of objects with a 'query' key")
    return queries


def _normalized(query: str) -> str:
    return query.strip().lower()


def _label_query(retriever: KiCampusRetriever, llm: LLM, entry: dict, model: Models) -> list[dict] | None:
    """Retrieve + judge one query. Returns chunk dicts, or None when retrieval
    found nothing (legitimate empty result, not an error)."""
    nodes = retriever.retrieve(
        query=entry["query"],
        course_id=entry.get("course_id"),
        module_id=entry.get("module_id"),
    )
    if not nodes:
        return None
    chunks = []
    for node in tqdm(nodes, desc="  judging chunks", leave=False, unit="chunk"):
        relevance = judge_relevance(llm, entry["query"], node.text, model)
        chunks.append({
            "id": node.id_,
            "text": node.text,
            "metadata": node.metadata,
            "score": node.score,
            "relevance": relevance,
        })
        time.sleep(0.2)  # avoid rate limits
    return chunks


def create_dataset(
    output_path: Path = DEFAULT_OUTPUT,
    limit: int | None = None,
    model: Models = Models.AZURE_FALLBACK,
    retrieve_top_n: int = 30,
    append: bool = False,
    queries: list[dict] | None = None,
) -> None:
    """Label queries into output_path (JSONL, one record per query).

    Reliability contract: records are written and flushed one by one; a failing
    query is retried once and then skipped (listed in the final summary) — a
    single flaky LLM/API call never kills the run. Re-running with append=True
    resumes: already-labeled queries are skipped, failed/missing ones are
    re-attempted. Without append the file is rebuilt from scratch.
    """
    llm = LLM()
    retriever = KiCampusRetriever(use_hybrid=True, n_chunks=retrieve_top_n)
    existing_queries: set[str] = set()
    existing_records = 0
    if append and output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_queries.add(_normalized(json.loads(line)["query"]))
                    existing_records += 1
    elif not append and output_path.exists():
        print(f"Overwriting existing {output_path} (use --append to resume/extend instead).")

    source_queries = queries if queries is not None else SAMPLE_QUERIES
    # In-run dedup (curated core + generated set may overlap), order-preserving.
    seen: set[str] = set()
    deduped: list[dict] = []
    for q in source_queries:
        key = _normalized(q["query"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(q)
    query_dicts = deduped[:limit] if limit else deduped
    if append:
        query_dicts = [q for q in query_dicts if _normalized(q["query"]) not in existing_queries]
        if not query_dicts:
            print("No new queries to add.")
            return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    empty: list[str] = []
    failed: list[str] = []
    with output_path.open("a" if append else "w", encoding="utf-8") as f:
        query_bar = tqdm(enumerate(query_dicts), total=len(query_dicts), unit="query")
        for i, entry in query_bar:
            query = entry["query"]
            course_id = entry.get("course_id")
            module_id = entry.get("module_id")
            language = entry.get("language", "de")
            scope = f"course_id={course_id}" if course_id else ("Drupal" if not module_id else f"module_id={module_id}")

            query_bar.set_description(f"[{language}/{scope}] {query[:50]}")
            logger.info("[%d/%d] Retrieving (%s, %s): %s", i + 1, len(query_dicts), language, scope, query)

            # One retry, then skip — a flaky call must not kill a long run.
            chunks: list[dict] | None = None
            query_failed = False
            for attempt in (1, 2):
                try:
                    chunks = _label_query(retriever, llm, entry, model)
                    break
                except Exception as e:
                    if attempt == 1:
                        logger.warning("Query failed (%s) — retrying in 5s: %s", e, query[:60])
                        time.sleep(5)
                    else:
                        logger.error("Query failed twice — skipping (rerun with --append to retry): %s", query[:60])
                        failed.append(query)
                        query_failed = True
            if query_failed:
                continue
            if chunks is None:
                logger.warning("No chunks retrieved for query: %s", query)
                empty.append(query)
                continue

            record = {
                # Numbered by records actually written so --append never
                # produces colliding ids after skipped queries.
                "query_id": f"q{existing_records + written + 1:03d}",
                "query": query,
                "language": language,
                "course_id": course_id,
                "module_id": module_id,
                # Provenance from generated query sets (kind, expected_no_answer
                # for min_score calibration, source_doc_keys for grounding).
                **{k: entry[k] for k in ("kind", "expected_no_answer", "source_doc_keys") if k in entry},
                "retrieved_chunks": chunks,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # crash-safe: every finished record is durable
            written += 1
            logger.info("  → %d chunks, relevance distribution: %s",
                len(chunks),
                {r: sum(1 for c in chunks if c["relevance"] == r) for r in (0, 1, 2)},
            )

    print(f"Dataset saved to {output_path} ({existing_records + written} queries total, {written} added)")
    if empty:
        print(f"⚠ {len(empty)} query/queries returned no chunks (not written): {empty}")
    if failed:
        print(f"⚠ {len(failed)} query/queries FAILED (rerun the same command with --append to retry): {failed}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Create reranker evaluation dataset")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="Only process first N queries")
    parser.add_argument("--model", type=str, default="Azure-Fallback", choices=[m.value for m in Models])
    parser.add_argument("--append", action="store_true", help="Only add queries not yet in the dataset (preserves existing labels)")
    parser.add_argument(
        "--n-chunks", type=int, default=30,
        help="Candidate pool size: chunks retrieved and judged per query (default: 30). "
             "Benchmark pool-size experiments (--pool-sizes) need at least the largest pool here.",
    )
    parser.add_argument(
        "--queries-file", type=Path, default=None,
        help="JSON file with queries (list of {query, language, course_id?, kind?, ...}), "
             "e.g. from evaluation/dataset/generate_questions.py. Default: built-in SAMPLE_QUERIES.",
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="One-shot mode: curated core (SAMPLE_QUERIES) + index-generated level "
             "questions (drupal/course/module/comparison/negative) in a single run. "
             "Reuses evaluation/dataset/generated_queries.json when it exists so a "
             "resumed run labels the SAME questions; --regenerate forces fresh generation.",
    )
    parser.add_argument(
        "--regenerate", action="store_true",
        help="With --generate: regenerate the questions even if generated_queries.json "
             "exists. Do this once after every re-ingest (ids/content changed).",
    )
    args = parser.parse_args()

    if args.generate and args.queries_file:
        parser.error("--generate and --queries-file are mutually exclusive "
                     "(--generate already combines SAMPLE_QUERIES with the generated set)")

    if args.generate:
        from src.env import env
        from evaluation.dataset.generate_questions import (
            DEFAULT_OUTPUT as GENERATED_QUERIES_PATH,
            generate_questions,
        )
        if args.regenerate or not GENERATED_QUERIES_PATH.exists():
            print("Generating level questions from the index …")
            generated = generate_questions(index_name=env.AZURE_SEARCH_INDEX)
        else:
            print(f"Reusing existing {GENERATED_QUERIES_PATH} — pass --regenerate after a re-ingest.")
            generated = load_queries_file(GENERATED_QUERIES_PATH)
        queries = SAMPLE_QUERIES + generated
    elif args.queries_file:
        queries = load_queries_file(args.queries_file)
    else:
        queries = None

    create_dataset(
        output_path=args.output,
        limit=args.limit,
        model=Models(args.model),
        append=args.append,
        retrieve_top_n=args.n_chunks,
        queries=queries,
    )
