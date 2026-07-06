"""
Ground truth dataset creator for reranker evaluation.

Workflow:
  1. For each query in SAMPLE_QUERIES, retrieve top-10 chunks via the real retriever.
  2. For each (query, chunk) pair, ask an LLM to rate relevance: 0 / 1 / 2.
  3. Save results to evaluation/dataset/queries.jsonl

Usage:
  python -m evaluation.dataset.create_dataset

  Optional flags:
    --output PATH     Override output file path
    --limit N         Only process first N queries (useful for testing)
    --model MODEL     LLM model for judging (default: Azure-Fallback)
"""

import argparse
import json
import logging
import time
from pathlib import Path

from src.llm.objects.LLMs import LLM, Models
from src.llm.objects.retriever import KiCampusRetriever

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path(__file__).parent / "queries.jsonl"

# Realistic KI-Campus queries drawn from Bruno collection and actual usage patterns.
#
# Each entry is a dict with:
#   query     : the user query (required)
#   course_id : Moodle course ID — retrieves course-specific content (optional)
#   module_id : Moodle module ID — retrieves module-specific content (optional)
#
# Without course_id/module_id the retriever restricts to Drupal content only.
# For course-specific queries, fill in the actual Moodle course_id to get
# realistic results — otherwise the retriever returns Drupal articles about
# the course rather than the course content itself.
#
# Find course IDs via the frontend course tree or the Azure Search index.
SAMPLE_QUERIES: list[dict] = [
    # --- Kursentdeckung: Drupal-Content, kein course_id nötig ---
    {"query": "Welche Kurse gibt es zum Thema maschinelles Lernen?", "language": "de"},
    {"query": "Welche Kurse gibt es zum Thema Ethik?", "language": "de"},
    {"query": "Welche Kurse gibt es auf KI-Campus zum Thema Datenschutz und KI?", "language": "de"},
    {"query": "Wie unterscheiden sich die Kurse 'Die Welt der Daten' und 'KI und Arbeitswelt'?", "language": "de"},

    # --- Kursspezifisch: echter Moodle-Content via course_id ---
    {"query": "Was lerne ich in diesem Kurs?", "course_id": 387, "language": "de"},           # Einführung in die Künstliche Intelligenz
    {"query": "Was sind die Lernziele dieses Kurses?", "course_id": 435, "language": "de"},   # MPG: Die Welt der Daten
    {"query": "Was ist der Inhalt dieses Kurses?", "course_id": 344, "language": "de"},       # KI und Arbeitswelt
    {"query": "Wie steht es um KI in der Arbeitswelt?", "course_id": 344, "language": "de"},
    {"query": "Was ist AI Literacy?", "course_id": 344, "language": "de"},
    {"query": "Erkläre mir die fünf Säulen der KI-Ethik.", "course_id": 331, "language": "de"},  # Die fünf Säulen der KI-Ethik
    {"query": "Was versteht man unter erklärbarem maschinellen Lernen?", "course_id": 342, "language": "de"},  # Erklärbares ML für Ingenieurwissenschaften
    {"query": "Wie funktionieren autonome KI-Agenten?", "course_id": 406, "language": "de"},  # Von autonomen KI-Agenten zu Multiagentensystemen
    {"query": "Was ist Deep Learning?", "course_id": 369, "language": "de"},                  # Foundations of Deep Learning

    # --- Konzepterklärung (Einsteigerniveau) ---
    {"query": "Erkläre mir was ein neuronales Netz ist.", "language": "de"},
    {"query": "Was ist maschinelles Lernen?", "language": "de"},
    {"query": "Was ist ein Large Language Model?", "language": "de"},
    {"query": "Was versteht man unter Bias in KI-Systemen?", "language": "de"},

    # --- Technische Vertiefung (aus Bruno Collection) ---
    {"query": "Wie funktioniert Backpropagation beim Deep Learning Training?", "language": "de"},
    {"query": "Wie funktioniert das Attention-Mechanismus in Transformern?", "language": "de"},
    {"query": "Was ist der Unterschied zwischen Supervised und Unsupervised Learning?", "language": "de"},
    {"query": "Was ist Overfitting und wie verhindert man es?", "language": "de"},
    {"query": "Was ist der Unterschied zwischen Precision und Recall bei Klassifikationsmodellen?", "language": "de"},

    # --- Anwendung und Transfer ---
    {"query": "Wie kann KI im Unterricht eingesetzt werden?", "language": "de"},
    {"query": "Was sind die Auswirkungen von KI auf die Arbeitswelt?", "language": "de"},
    {"query": "Welche Rechte haben Beschäftigte beim Einsatz von KI am Arbeitsplatz?", "language": "de"},
    {"query": "Wie kann ich als Lehrkraft KI-Tools sinnvoll nutzen?", "language": "de"},

    # --- Lernhilfe / Assignment-Kontext (aus Bruno Collection) ---
    {"query": "Ich brauche Hilfe mit meinem Machine Learning Assignment zum Thema Overfitting.", "language": "de"},
    {"query": "Ich verstehe den Unterschied zwischen Varianz und Bias nicht.", "language": "de"},

    # --- Ethik und Gesellschaft ---
    {"query": "Was sind die ethischen Risiken von Gesichtserkennung?", "language": "de"},
    {"query": "Was bedeutet algorithmische Diskriminierung?", "language": "de"},
    {"query": "Was ist Explainable AI und warum ist es wichtig?", "language": "de"},

    # --- Berufsfelder und Orientierung ---
    {"query": "Welche Berufsfelder entstehen durch KI und Data Science?", "language": "de"},
    {"query": "Was muss ich können um Data Scientist zu werden?", "language": "de"},

    # --- English queries (multilingual test) ---
    # Course discovery
    {"query": "What courses are available on machine learning?", "language": "en"},
    {"query": "What courses are there about AI ethics?", "language": "en"},
    {"query": "What courses cover data privacy and AI?", "language": "en"},

    # Course-specific
    {"query": "What will I learn in this course?", "course_id": 387, "language": "en"},
    {"query": "What are the learning objectives of this course?", "course_id": 435, "language": "en"},
    {"query": "What is the impact of AI on the workplace?", "course_id": 344, "language": "en"},
    {"query": "What is deep learning?", "course_id": 369, "language": "en"},

    # Concept explanations
    {"query": "Explain what a neural network is.", "language": "en"},
    {"query": "What is machine learning?", "language": "en"},
    {"query": "What is a large language model?", "language": "en"},
    {"query": "What is bias in AI systems?", "language": "en"},

    # Technical depth
    {"query": "How does backpropagation work in deep learning?", "language": "en"},
    {"query": "How does the attention mechanism in transformers work?", "language": "en"},
    {"query": "What is the difference between supervised and unsupervised learning?", "language": "en"},
    {"query": "What is overfitting and how do you prevent it?", "language": "en"},

    # Ethics and application
    {"query": "What are the ethical risks of facial recognition?", "language": "en"},
    {"query": "What is explainable AI and why does it matter?", "language": "en"},
    {"query": "How can AI be used in education?", "language": "en"},
    {"query": "What career fields are emerging from AI and data science?", "language": "en"},
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


def create_dataset(
    output_path: Path = DEFAULT_OUTPUT,
    limit: int | None = None,
    model: Models = Models.AZURE_FALLBACK,
    retrieve_top_n: int = 10,
    append: bool = False,
) -> None:
    llm = LLM()
    retriever = KiCampusRetriever(use_hybrid=True, n_chunks=retrieve_top_n)
    existing_queries: set[str] = set()
    existing_count = 0
    if append and output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_queries.add(json.loads(line)["query"])
        existing_count = len(existing_queries)

    query_dicts = SAMPLE_QUERIES[:limit] if limit else SAMPLE_QUERIES
    if append:
        query_dicts = [q for q in query_dicts if q["query"] not in existing_queries]
        if not query_dicts:
            print("No new queries to add.")
            return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a" if append else "w", encoding="utf-8") as f:
        for i, entry in enumerate(query_dicts):
            query = entry["query"]
            course_id = entry.get("course_id")
            module_id = entry.get("module_id")
            language = entry.get("language", "de")
            scope = f"course_id={course_id}" if course_id else ("Drupal" if not module_id else f"module_id={module_id}")

            logger.info("[%d/%d] Retrieving (%s, %s): %s", i + 1, len(query_dicts), language, scope, query)

            nodes = retriever.retrieve(query=query, course_id=course_id, module_id=module_id)
            if not nodes:
                logger.warning("No chunks retrieved for query: %s", query)
                continue

            chunks = []
            for node in nodes:
                relevance = judge_relevance(llm, query, node.text, model)
                chunks.append({
                    "id": node.id_,
                    "text": node.text,
                    "metadata": node.metadata,
                    "score": node.score,
                    "relevance": relevance,
                })
                time.sleep(0.2)  # avoid rate limits

            record = {
                "query_id": f"q{existing_count + i + 1:03d}",
                "query": query,
                "language": language,
                "course_id": course_id,
                "module_id": module_id,
                "retrieved_chunks": chunks,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("  → %d chunks, relevance distribution: %s",
                len(chunks),
                {r: sum(1 for c in chunks if c["relevance"] == r) for r in (0, 1, 2)},
            )

    total = existing_count + len(query_dicts)
    print(f"Dataset saved to {output_path} ({total} queries total, {len(query_dicts)} added)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Create reranker evaluation dataset")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="Only process first N queries")
    parser.add_argument("--model", type=str, default="Azure-Fallback", choices=[m.value for m in Models])
    parser.add_argument("--append", action="store_true", help="Only add queries not yet in the dataset (preserves existing labels)")
    args = parser.parse_args()

    create_dataset(
        output_path=args.output,
        limit=args.limit,
        model=Models(args.model),
        append=args.append,
    )
