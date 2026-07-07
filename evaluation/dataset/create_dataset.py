"""
Ground truth dataset creator for reranker evaluation.

Workflow:
  1. For each query (built-in SAMPLE_QUERIES or --queries-file), retrieve the
     top --n-chunks candidate chunks via the real retriever.
  2. For each (query, chunk) pair, ask an LLM to rate relevance: 0 / 1 / 2.
  3. Save results to evaluation/dataset/queries.jsonl

Usage:
  python -m evaluation.dataset.create_dataset

  Optional flags:
    --output PATH        Override output file path
    --limit N            Only process first N queries (useful for testing)
    --model MODEL        LLM model for judging (default: Azure-Fallback)
    --n-chunks N         Candidate pool size per query (default: 30) — benchmark
                         pool-size experiments need at least the largest pool
    --queries-file PATH  Use generated queries (see generate_questions.py)
                         instead of the built-in SAMPLE_QUERIES
    --append             Only add queries not yet in the dataset
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

    # --- Kurs-Level: reale course_ids aus dem Index kic-content (Stand 2026-07) ---
    # So fragt der eingebettete Chatbot-Nutzer: Er ist IM Kurs und sagt "dieser Kurs".
    {"query": "Ich bin Ärztin — was bringt mir dieser Kurs?", "course_id": 78, "language": "de", "kind": "course_level"},   # Dr. med. KI - Grundlagen für Ärztinnen und Ärzte
    {"query": "Wie kann KI beim Erreichen der Nachhaltigkeitsziele helfen?", "course_id": 63, "language": "de", "kind": "course_level"},  # KI und Ziele für nachhaltige Entwicklung
    {"query": "Welche Methoden zur Bias-Reduktion werden hier behandelt?", "course_id": 224, "language": "de", "kind": "course_level"},   # Methoden der Bias-Reduktion
    {"query": "Welche ethischen Anwendungsfälle bespricht dieser Kurs?", "course_id": 257, "language": "de", "kind": "course_level"},     # KI und Ethik IV: Anwendungen
    {"query": "Worum geht es in diesem Kurs?", "course_id": 321, "language": "de", "kind": "course_level"},                               # experimenta: KI-Basics
    {"query": "Wie funktionieren neuronale Netze laut diesem Kurs?", "course_id": 301, "language": "de", "kind": "course_level"},         # experimenta: KNN & Deep Learning
    {"query": "Give me an overview of what this course covers.", "course_id": 50, "language": "en", "kind": "course_level"},              # Reinforcement Learning (EN)

    # --- Modul-Level: reale course_id+module_id-Paare aus kic-content (Stand 2026-07) ---
    # Der häufigste Embedded-Fall: Nutzer steht in einem konkreten Modul.
    {"query": "Wie ist ein künstliches neuronales Netz aufgebaut?", "course_id": 56, "module_id": 1558, "language": "de", "kind": "module_level"},        # "Aufbau eines KNN"
    {"query": "Was unterscheidet ein biologisches von einem künstlichen neuronalen Netz?", "course_id": 56, "module_id": 1559, "language": "de", "kind": "module_level"},  # "Biologisches vs. Künstliches NN"
    {"query": "Wie hat sich die KI vom Perzeptron bis heute entwickelt?", "course_id": 56, "module_id": 1550, "language": "de", "kind": "module_level"},   # "Vom Perzeptron bis Pepper"
    {"query": "Worum geht es in diesem Modul?", "course_id": 63, "module_id": 2511, "language": "de", "kind": "module_level"},             # generische Modul-Frage
    {"query": "Wie könnte KI in Zukunft für die Gesundheit eingesetzt werden?", "course_id": 63, "module_id": 2513, "language": "de", "kind": "module_level"},
    {"query": "Fasse mir den Inhalt dieses Moduls kurz zusammen.", "course_id": 56, "module_id": 1558, "language": "de", "kind": "module_level"},          # Zusammenfassungs-Anfrage
    {"query": "Warum ist Datenqualität so wichtig?", "course_id": 60, "module_id": 4339, "language": "de", "kind": "module_level"},        # "Video: Daten – Datenqualität"
    {"query": "Was bedeutet Datenschutz und Selbstbestimmung am Arbeitsplatz?", "course_id": 38, "module_id": 2148, "language": "de", "kind": "module_level"},
    {"query": "What is Monte Carlo evaluation in reinforcement learning?", "course_id": 50, "module_id": 2684, "language": "en", "kind": "module_level"},
    {"query": "Explain model-free control in simple terms.", "course_id": 50, "module_id": 2689, "language": "en", "kind": "module_level"},

    # --- Modul-Level in weiteren Sprachen (ES/FR/IT/TR — wie Language Detector
    # und Bruno-Collections). Bewusst DIESELBEN Module wie die deutschen Fragen
    # oben: Der Sprach-Slice im Benchmark isoliert so den reinen Spracheffekt
    # bei identischer Ground Truth (Inhalte sind DE/EN → Cross-Lingual-Retrieval). ---
    {"query": "¿Cómo está estructurada una red neuronal artificial?", "course_id": 56, "module_id": 1558, "language": "es", "kind": "module_level"},
    {"query": "¿De qué trata este módulo?", "course_id": 63, "module_id": 2511, "language": "es", "kind": "module_level"},
    {"query": "¿Por qué es tan importante la calidad de los datos?", "course_id": 60, "module_id": 4339, "language": "es", "kind": "module_level"},
    {"query": "Comment un réseau de neurones artificiel est-il structuré ?", "course_id": 56, "module_id": 1558, "language": "fr", "kind": "module_level"},
    {"query": "Comment l'IA pourrait-elle être utilisée pour la santé à l'avenir ?", "course_id": 63, "module_id": 2513, "language": "fr", "kind": "module_level"},
    {"query": "Résume-moi le contenu de ce module.", "course_id": 38, "module_id": 2148, "language": "fr", "kind": "module_level"},
    {"query": "Qual è la differenza tra una rete neurale biologica e una artificiale?", "course_id": 56, "module_id": 1559, "language": "it", "kind": "module_level"},
    {"query": "Di cosa parla questo modulo?", "course_id": 60, "module_id": 4325, "language": "it", "kind": "module_level"},
    {"query": "Yapay sinir ağı nasıl yapılandırılmıştır?", "course_id": 56, "module_id": 1558, "language": "tr", "kind": "module_level"},
    {"query": "Bu modül ne hakkında?", "course_id": 63, "module_id": 2511, "language": "tr", "kind": "module_level"},
    {"query": "Veri kalitesi neden bu kadar önemli?", "course_id": 60, "module_id": 4339, "language": "tr", "kind": "module_level"},

    # --- Vergleichs-/Multi-Konzept-Fragen ---
    # Der Fragetyp, der früher über multi_hop lief und jetzt allein am
    # Zusammenspiel Hybrid-Retrieval + Reranker hängt — gezielt messen!
    {"query": "Was ist der Unterschied zwischen KI, maschinellem Lernen und Deep Learning?", "language": "de", "kind": "comparison"},
    {"query": "Was unterscheidet starke von schwacher KI?", "language": "de", "kind": "comparison"},
    {"query": "Worin unterscheiden sich CNNs und RNNs und wofür nutzt man sie jeweils?", "language": "de", "kind": "comparison"},
    {"query": "Was ist der Unterschied zwischen generativer KI und klassischen Klassifikationsmodellen?", "language": "de", "kind": "comparison"},
    {"query": "Wie hängen Bias, Varianz und Overfitting zusammen?", "language": "de", "kind": "comparison"},
    {"query": "Sollte ich für den Einstieg lieber einen Python-Kurs oder einen Kurs ohne Programmierung wählen?", "language": "de", "kind": "comparison"},

    # --- Kurze/vage Suchanfragen (realistisches Nutzerverhalten) ---
    {"query": "KI Ethik", "language": "de", "kind": "short"},
    {"query": "prompt engineering", "language": "de", "kind": "short"},
    {"query": "neuronale netze einfach erklärt", "language": "de", "kind": "short"},
    {"query": "zertifikat", "language": "de", "kind": "short"},
    {"query": "chatbot kurs", "language": "de", "kind": "short"},

    # --- Umgangssprachlich / mit Tippfehlern ---
    {"query": "was is n transformer modell?", "language": "de", "kind": "colloquial"},
    {"query": "kannst du mir was über machine lerning erzählen", "language": "de", "kind": "colloquial"},
    {"query": "ich check das mit dem gradient descent einfach nicht", "language": "de", "kind": "colloquial"},
    {"query": "gibts hier auch was für totale anfänger ohne mathe?", "language": "de", "kind": "colloquial"},

    # --- Plattform-/Orga-Fragen (Drupal-Content) ---
    {"query": "Wie bekomme ich ein Zertifikat auf dem KI-Campus?", "language": "de", "kind": "platform"},
    {"query": "Ist der KI-Campus kostenlos?", "language": "de", "kind": "platform"},
    {"query": "Wer steht hinter dem KI-Campus und wer finanziert ihn?", "language": "de", "kind": "platform"},
    {"query": "Kann ich Kurse auch ohne Anmeldung ausprobieren?", "language": "de", "kind": "platform"},
    {"query": "Gibt es Kurse mit ECTS-Punkten oder Micro-Degrees?", "language": "de", "kind": "platform"},

    # --- Domänenspezifisch (Medizin, Schule, Verwaltung) ---
    {"query": "Wie wird KI in der Medizin eingesetzt?", "language": "de", "kind": "domain"},
    {"query": "Was muss ich als Pflegekraft über KI wissen?", "language": "de", "kind": "domain"},
    {"query": "Welche KI-Kompetenzen brauchen Lehrkräfte an Schulen?", "language": "de", "kind": "domain"},
    {"query": "Gibt es Kurse zu KI in der öffentlichen Verwaltung?", "language": "de", "kind": "domain"},
    {"query": "Wie kann KI in der Diagnostik unterstützen?", "language": "de", "kind": "domain"},

    # --- Technische Vertiefung (Ergänzung) ---
    {"query": "Was ist Reinforcement Learning und wo wird es eingesetzt?", "language": "de"},
    {"query": "Was sind Halluzinationen bei Sprachmodellen und warum entstehen sie?", "language": "de"},
    {"query": "Was ist Retrieval-Augmented Generation?", "language": "de"},
    {"query": "Was bedeutet Fine-Tuning bei Sprachmodellen?", "language": "de"},
    {"query": "Wie funktioniert ein Entscheidungsbaum?", "language": "de"},
    {"query": "Was ist Feature Engineering und warum ist es wichtig?", "language": "de"},

    # --- No-Answer-Kalibrierung: plausibel, aber (sehr wahrscheinlich) nicht
    # durch die Wissensbasis gedeckt. Erwartung: alle Chunks niedrig bewertet →
    # Ground Truth für MIN_RERANKER_SCORE und die No-Answer-Logik. ---
    {"query": "Wie viel kostet ein ChatGPT-Plus-Abo?", "language": "de", "kind": "negative", "expected_no_answer": True},
    {"query": "Wann findet die nächste KI-Campus-Präsenzveranstaltung in Berlin statt?", "language": "de", "kind": "negative", "expected_no_answer": True},
    {"query": "Welche KI-Aktien soll ich kaufen?", "language": "de", "kind": "negative", "expected_no_answer": True},
    {"query": "Wie installiere ich CUDA-Treiber unter Windows 11?", "language": "de", "kind": "negative", "expected_no_answer": True},
    {"query": "Kannst du mir die Klausurlösungen für meinen Uni-Kurs geben?", "language": "de", "kind": "negative", "expected_no_answer": True},
    {"query": "Was hat der Bundestag letzte Woche zur KI-Regulierung beschlossen?", "language": "de", "kind": "negative", "expected_no_answer": True},

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

    # Comparison / multi-concept
    {"query": "What is the difference between AI, machine learning and deep learning?", "language": "en", "kind": "comparison"},
    {"query": "How do CNNs and RNNs differ and when would I use each?", "language": "en", "kind": "comparison"},
    {"query": "How are bias, variance and overfitting related?", "language": "en", "kind": "comparison"},

    # Short / vague
    {"query": "prompt engineering course", "language": "en", "kind": "short"},
    {"query": "certificate", "language": "en", "kind": "short"},

    # Colloquial / with typos
    {"query": "can u explain transformers like im five", "language": "en", "kind": "colloquial"},
    {"query": "i dont get gradient descent at all", "language": "en", "kind": "colloquial"},

    # Platform questions
    {"query": "How do I get a certificate on KI-Campus?", "language": "en", "kind": "platform"},
    {"query": "Is KI-Campus free to use?", "language": "en", "kind": "platform"},

    # Domain-specific
    {"query": "How is AI used in medical diagnostics?", "language": "en", "kind": "domain"},
    {"query": "What AI skills do teachers need?", "language": "en", "kind": "domain"},

    # Technical depth (additions)
    {"query": "What is reinforcement learning and where is it applied?", "language": "en"},
    {"query": "What are hallucinations in large language models?", "language": "en"},
    {"query": "What is retrieval-augmented generation?", "language": "en"},

    # No-answer calibration (expected to be uncovered by the knowledge base)
    {"query": "How much does a ChatGPT Plus subscription cost?", "language": "en", "kind": "negative", "expected_no_answer": True},
    {"query": "Which AI stocks should I buy right now?", "language": "en", "kind": "negative", "expected_no_answer": True},
    {"query": "What did the EU parliament decide about AI regulation last week?", "language": "en", "kind": "negative", "expected_no_answer": True},
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


def create_dataset(
    output_path: Path = DEFAULT_OUTPUT,
    limit: int | None = None,
    model: Models = Models.AZURE_FALLBACK,
    retrieve_top_n: int = 30,
    append: bool = False,
    queries: list[dict] | None = None,
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

    source_queries = queries if queries is not None else SAMPLE_QUERIES
    query_dicts = source_queries[:limit] if limit else source_queries
    if append:
        query_dicts = [q for q in query_dicts if q["query"] not in existing_queries]
        if not query_dicts:
            print("No new queries to add.")
            return

    output_path.parent.mkdir(parents=True, exist_ok=True)

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

            nodes = retriever.retrieve(query=query, course_id=course_id, module_id=module_id)
            if not nodes:
                logger.warning("No chunks retrieved for query: %s", query)
                continue

            chunks = []
            for node in tqdm(nodes, desc="  judging chunks", leave=False, unit="chunk"):
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
                # Provenance from generated query sets (kind: grounded/comparison/
                # negative, expected_no_answer for min_score calibration).
                **{k: entry[k] for k in ("kind", "expected_no_answer", "source_doc_keys") if k in entry},
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
    args = parser.parse_args()

    create_dataset(
        output_path=args.output,
        limit=args.limit,
        model=Models(args.model),
        append=args.append,
        retrieve_top_n=args.n_chunks,
        queries=load_queries_file(args.queries_file) if args.queries_file else None,
    )
