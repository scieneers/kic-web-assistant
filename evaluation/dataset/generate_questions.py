"""
Content-grounded query generator for reranker/no-answer evaluation.

Instead of hand-crafting queries (and hard-coding course/module ids that go
stale with every re-ingest), this script samples real chunks from the Azure
Search index and lets an LLM generate realistic learner questions — WITH the
current, real ids straight from the index. Question kinds mirror the three
production scopes plus two special types:

  - drupal_level : unscoped question a ki-campus.org visitor would ask,
                   grounded in a Drupal chunk (prod: no course_id/module_id)
  - course_level : question a learner asks INSIDE a course (embedded chatbot),
                   grounded in a Moodle chunk, scoped with its course_id
  - module_level : question asked inside a specific module, scoped with
                   course_id + module_id (the embedded-chatbot default)
  - comparison   : requires content from two different Drupal documents (the
                   query type that used to be served by the removed multi_hop path)
  - negative     : plausible questions the content does NOT cover — ground
                   truth for no-answer / min_score calibration
                   (expected_no_answer=true)

Output is a JSON list consumed by create_dataset.py via --queries-file.

Usage:
  python -m evaluation.dataset.generate_questions

  Optional flags:
    --index NAME       Search index to sample from (default: env AZURE_SEARCH_INDEX)
    --n-drupal N       Drupal-level questions (default: 8)
    --n-course N       Course-level questions (default: 8)
    --n-module N       Module-level questions (default: 8)
    --n-comparison N   Comparison questions (default: 4)
    --n-negative N     Negative/no-answer questions (default: 4)
    --languages de,en  Languages, round-robin over generated questions (default: de,en)
    --model MODEL      LLM for generation (default: Azure-Fallback)
    --seed N           Random seed for reproducible sampling (default: 42)
    --output PATH      Output file (default: evaluation/dataset/generated_queries.json)

Then create the labeled dataset from it (typically appended to the curated core):
  python -m evaluation.dataset.create_dataset
  python -m evaluation.dataset.create_dataset --queries-file evaluation/dataset/generated_queries.json --append
"""

import argparse
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

from src.env import env
from src.llm.objects.LLMs import LLM, Models
from src.vectordb.azure_search import VectorDBAzureSearch

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path(__file__).parent / "generated_queries.json"

# Chunks shorter than this are unlikely to support a substantive question.
_MIN_CHUNK_CHARS = 300
# How many chunks to fetch per source before local sampling.
_FETCH_PER_SOURCE = 400

_LANG_NAME = {"de": "Deutsch", "en": "English", "es": "Español", "fr": "Français", "it": "Italiano", "tr": "Türkçe"}

# Per-scope persona context injected into the grounded prompt so the generated
# question matches how users actually ask at that level in production.
_LEVEL_CONTEXT = {
    "drupal_level": (
        "Der Fragende ist Besucher der Website ki-campus.org und in keinen Kurs "
        "eingeschrieben. Die Frage darf sich NICHT auf einen bestimmten Kurs "
        "beziehen (kein „dieser Kurs“)."
    ),
    "course_level": (
        "Der Fragende ist in den Kurs „{course}“ eingeschrieben und nutzt den "
        "Chatbot INNERHALB des Kurses. Die Frage darf sich implizit auf den Kurs "
        "beziehen (z. B. „in diesem Kurs“), muss aber nicht."
    ),
    "module_level": (
        "Der Fragende arbeitet gerade an einem konkreten Lernmodul im Kurs "
        "„{course}“ und nutzt den Chatbot direkt dort. Die Frage bezieht sich "
        "auf den Inhalt genau dieses Moduls (z. B. „in diesem Modul“ oder direkt "
        "inhaltlich)."
    ),
}

GROUNDED_PROMPT = """Du erstellst Evaluationsfragen für den KI-Campus-Chatbot (Lernplattform für Künstliche Intelligenz).

Situation: {level_context}

Hier ist ein Ausschnitt aus den Lerninhalten:

Titel: {title}
Inhalt:
{text}

Formuliere EINE realistische Frage, die dieser Lernende dem Chatbot stellen könnte und die sich mit diesem Inhalt beantworten lässt.

Regeln:
- Sprache der Frage: {language}
- Natürliche Lernenden-Sprache, kein Zitat aus dem Text
- Die Frage darf den Titel/Kursnamen NICHT wörtlich enthalten (Lernende kennen ihn oft nicht)
- Keine Meta-Fragen über das Dokument ("Was steht in diesem Text?")

Antworte ausschließlich mit der Frage, ohne Anführungszeichen."""

COMPARISON_PROMPT = """Du erstellst Evaluationsfragen für den KI-Campus-Chatbot (Lernplattform für Künstliche Intelligenz).

Hier sind Ausschnitte aus ZWEI verschiedenen Lerninhalten:

--- Inhalt A: {title_a} ---
{text_a}

--- Inhalt B: {title_b} ---
{text_b}

Formuliere EINE realistische Vergleichs- oder Verbindungsfrage, die ein Lernender stellen könnte und für deren Antwort man BEIDE Inhalte braucht (z. B. Unterschied, Gemeinsamkeit, Zusammenhang).

Regeln:
- Sprache der Frage: {language}
- Natürliche Lernenden-Sprache
- Keine Meta-Fragen über die Dokumente

Antworte ausschließlich mit der Frage, ohne Anführungszeichen."""

NEGATIVE_PROMPT = """Du erstellst Evaluationsfragen für den KI-Campus-Chatbot (Lernplattform für Künstliche Intelligenz).

Hier sind typische Beispiele für die Lerninhalte der Plattform:

{examples}

Formuliere EINE Frage, die ein Nutzer dem Chatbot plausibel stellen könnte, die aber mit den Lerninhalten der Plattform sehr wahrscheinlich NICHT beantwortbar ist. Beispiele für solche Lücken: sehr spezifische Produkt-/Preisfragen, tagesaktuelle Ereignisse, organisatorische Fragen außerhalb der Kurse, sehr spezielle Nischenthemen ohne Kursbezug.

Regeln:
- Sprache der Frage: {language}
- Die Frage muss thematisch in die Nähe von KI/Lernen passen (kein absichtlicher Unsinn)
- Antworte ausschließlich mit der Frage, ohne Anführungszeichen."""


def _clean_question(raw: str) -> str | None:
    """Normalize LLM output to a single plausible question, or None if unusable."""
    q = (raw or "").strip().strip('"').strip("'").strip()
    q = q.splitlines()[0].strip() if q else ""
    if not (10 <= len(q) <= 300):
        return None
    return q


def _sample_chunks(index_name: str, rng: random.Random) -> dict[str, list[dict]]:
    """Fetch and sample substantive chunks per source, spread across documents.

    Returns {source: [chunk_dict, ...]} with fields id, text, title, source,
    type, course_id, module_id, url, source_doc_key.
    """
    db = VectorDBAzureSearch()
    client = db._client(index_name)  # evaluation tooling — direct client access is deliberate
    select = ["id", "text", "title", "fullname", "source", "type", "course_id", "module_id", "url", "source_doc_key"]
    by_source: dict[str, list[dict]] = {}

    for source in ("Drupal", "Moodle"):
        results = client.search(
            search_text="*",
            filter=f"source eq '{source}' and type ne 'EmptyModule'",
            select=select,
            top=_FETCH_PER_SOURCE,
        )
        chunks = [dict(r) for r in results]
        chunks = [c for c in chunks if len(c.get("text") or "") >= _MIN_CHUNK_CHARS]

        # Spread across documents: one chunk per source_doc_key/url first,
        # so the sample is not dominated by one long course.
        by_doc: dict[str, list[dict]] = defaultdict(list)
        for c in chunks:
            by_doc[c.get("source_doc_key") or c.get("url") or c["id"]].append(c)
        picked = [rng.choice(doc_chunks) for doc_chunks in by_doc.values()]
        rng.shuffle(picked)
        by_source[source] = picked
        logger.info("Sampled %d document chunks from %s (of %d fetched)", len(picked), source, len(chunks))

    return by_source


def _spread(chunks: list[dict], key_fn: Callable[[dict], object], rng: random.Random) -> list[dict]:
    """One chunk per key (e.g. per course, per module), shuffled — so a level's
    questions spread across real courses/modules instead of clustering."""
    groups: dict[object, list[dict]] = defaultdict(list)
    for c in chunks:
        key = key_fn(c)
        if key is not None:
            groups[key].append(c)
    picked = [rng.choice(group) for group in groups.values()]
    rng.shuffle(picked)
    return picked


def _generate(llm: LLM, model: Models, prompt: str) -> str | None:
    response = llm.chat(query=prompt, chat_history=[], model=model, system_prompt="")
    return _clean_question(response.content)


def generate_questions(
    index_name: str,
    n_drupal: int = 8,
    n_course: int = 8,
    n_module: int = 8,
    n_comparison: int = 4,
    n_negative: int = 4,
    languages: list[str] | None = None,
    model: Models = Models.AZURE_FALLBACK,
    seed: int = 42,
    output_path: Path = DEFAULT_OUTPUT,
) -> list[dict]:
    languages = languages or ["de", "en"]
    rng = random.Random(seed)
    llm = LLM()
    by_source = _sample_chunks(index_name, rng)
    drupal_pool = by_source.get("Drupal", [])
    moodle_pool = by_source.get("Moodle", [])
    if not drupal_pool and not moodle_pool:
        raise RuntimeError(f"No usable chunks sampled from index {index_name!r}")

    # Level pools with real, current ids from the index. Course-level questions
    # spread across courses, module-level across (course, module) pairs.
    course_pool = _spread(
        [c for c in moodle_pool if c.get("course_id") is not None],
        lambda c: c.get("course_id"), rng,
    )
    module_pool = _spread(
        [c for c in moodle_pool if c.get("course_id") is not None and c.get("module_id") is not None],
        lambda c: (c.get("course_id"), c.get("module_id")), rng,
    )

    queries: list[dict] = []
    seen: set[str] = set()

    def _add(question: str | None, *, kind: str, language: str, course_id=None, module_id=None, source_doc_keys=None, expected_no_answer=False):
        if not question or question.lower() in seen:
            return False
        seen.add(question.lower())
        entry: dict = {"query": question, "language": language, "kind": kind}
        if course_id is not None:
            entry["course_id"] = course_id
        if module_id is not None:
            entry["module_id"] = module_id
        if source_doc_keys:
            entry["source_doc_keys"] = source_doc_keys
        if expected_no_answer:
            entry["expected_no_answer"] = True
        queries.append(entry)
        return True

    def _grounded_round(pool: list[dict], n: int, kind: str, scope_fn: Callable[[dict], tuple]) -> None:
        """Generate up to n grounded questions of one scope kind from pool."""
        count = 0
        i = 0
        for chunk in pool:
            if count >= n:
                break
            lang = languages[i % len(languages)]
            i += 1
            course_name = chunk.get("fullname") or chunk.get("title") or ""
            question = _generate(llm, model, GROUNDED_PROMPT.format(
                level_context=_LEVEL_CONTEXT[kind].format(course=course_name),
                title=chunk.get("title") or course_name,
                text=(chunk.get("text") or "")[:2500],
                language=_LANG_NAME.get(lang, lang),
            ))
            course_id, module_id = scope_fn(chunk)
            added = _add(question, kind=kind, language=lang, course_id=course_id, module_id=module_id,
                         source_doc_keys=[chunk.get("source_doc_key")])
            if added:
                count += 1
                logger.info("[%s/%s] %s", kind, lang, question)
            time.sleep(0.2)
        if count < n:
            logger.warning("Only generated %d/%d %s questions (pool exhausted)", count, n, kind)

    # --- grounded per production scope ---
    _grounded_round(drupal_pool, n_drupal, "drupal_level", lambda c: (None, None))
    _grounded_round(course_pool, n_course, "course_level", lambda c: (c.get("course_id"), None))
    _grounded_round(module_pool, n_module, "module_level", lambda c: (c.get("course_id"), c.get("module_id")))

    # --- comparison (two chunks from different documents) ---
    # Comparison queries are never scoped with course_id/module_id, and an
    # unscoped query only ever searches Drupal content in production — so
    # candidates must be Drupal chunks, otherwise the question would be
    # grounded in content an unscoped query can never retrieve.
    comparison_pool = [c for c in drupal_pool if c.get("source_doc_key")]
    if len({c.get("source_doc_key") for c in comparison_pool}) < 2:
        logger.warning(
            "Skipping comparison questions: need chunks from >= 2 distinct Drupal documents, got %d",
            len(comparison_pool),
        )
        n_comparison = 0
    i = 0
    attempts = 0
    while sum(1 for q in queries if q["kind"] == "comparison") < n_comparison and attempts < n_comparison * 3:
        attempts += 1
        a, b = rng.sample(comparison_pool, 2)
        if a.get("source_doc_key") == b.get("source_doc_key"):
            continue
        lang = languages[i % len(languages)]
        i += 1
        question = _generate(llm, model, COMPARISON_PROMPT.format(
            title_a=a.get("title") or a.get("fullname") or "",
            text_a=(a.get("text") or "")[:1500],
            title_b=b.get("title") or b.get("fullname") or "",
            text_b=(b.get("text") or "")[:1500],
            language=_LANG_NAME.get(lang, lang),
        ))
        added = _add(question, kind="comparison", language=lang,
                     source_doc_keys=[a.get("source_doc_key"), b.get("source_doc_key")])
        if added:
            logger.info("[comparison/%s] %s", lang, question)
        time.sleep(0.2)

    # --- negative / no-answer ---
    example_pool = drupal_pool + moodle_pool
    i = 0
    attempts = 0
    while sum(1 for q in queries if q["kind"] == "negative") < n_negative and attempts < n_negative * 3:
        attempts += 1
        examples = "\n\n".join(
            f"- {c.get('title') or c.get('fullname') or ''}: {(c.get('text') or '')[:300]}"
            for c in rng.sample(example_pool, min(3, len(example_pool)))
        )
        lang = languages[i % len(languages)]
        i += 1
        question = _generate(llm, model, NEGATIVE_PROMPT.format(examples=examples, language=_LANG_NAME.get(lang, lang)))
        added = _add(question, kind="negative", language=lang, expected_no_answer=True)
        if added:
            logger.info("[negative/%s] %s", lang, question)
        time.sleep(0.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(queries, indent=2, ensure_ascii=False), encoding="utf-8")
    counts = {kind: sum(1 for q in queries if q["kind"] == kind)
              for kind in ("drupal_level", "course_level", "module_level", "comparison", "negative")}
    print(f"Generated {len(queries)} queries → {output_path} ({counts})")
    print("Next: python -m evaluation.dataset.create_dataset --queries-file", output_path, "--append")
    return queries


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate content-grounded evaluation queries from the search index")
    parser.add_argument("--index", type=str, default=None, help="Search index name (default: env AZURE_SEARCH_INDEX)")
    parser.add_argument("--n-drupal", type=int, default=8, help="Drupal-level (unscoped) questions")
    parser.add_argument("--n-course", type=int, default=8, help="Course-level questions (course_id from index)")
    parser.add_argument("--n-module", type=int, default=8, help="Module-level questions (course_id+module_id from index)")
    parser.add_argument("--n-comparison", type=int, default=4)
    parser.add_argument("--n-negative", type=int, default=4)
    parser.add_argument("--languages", type=str, default="de,en", help="Comma-separated, round-robin (default: de,en)")
    parser.add_argument("--model", type=str, default="Azure-Fallback", choices=[m.value for m in Models])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    generate_questions(
        index_name=args.index or env.AZURE_SEARCH_INDEX,
        n_drupal=args.n_drupal,
        n_course=args.n_course,
        n_module=args.n_module,
        n_comparison=args.n_comparison,
        n_negative=args.n_negative,
        languages=[lang.strip() for lang in args.languages.split(",") if lang.strip()],
        model=Models(args.model),
        seed=args.seed,
        output_path=args.output,
    )
