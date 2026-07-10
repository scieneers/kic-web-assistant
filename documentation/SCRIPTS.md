# Scripts & Tools

Alle Befehle werden aus dem Root-Verzeichnis des Repos ausgeführt.
Voraussetzung: `.env` mit den nötigen Credentials ist vorhanden.

---

## Backend starten

```bash
uv run uvicorn src.api.rest:app --reload
```

API dann erreichbar unter `http://localhost:8000`.  
Swagger-Doku: `http://localhost:8000/docs`

---

## Frontend starten (Streamlit)

```bash
uv run streamlit run src/frontend/frontend.py
```

---

## Interaktiver CLI-Chat

Direkt im Terminal mit dem Assistenten chatten — ohne Frontend, ohne HTTP.
Nützlich zum schnellen Testen von Modell, Reranker-Konfiguration oder course_id.

```bash
uv run python scripts/interactive_chat.py
```

**Flags:**

```bash
# Anderes Modell (Default: AZURE_FALLBACK)
uv run python scripts/interactive_chat.py --model GEMMA4_31B

# Auf einen Kurs einschränken
uv run python scripts/interactive_chat.py --course-id 387

# Auf ein Modul einschränken
uv run python scripts/interactive_chat.py --course-id 387 --module-id 12

# Mehr Chunks holen, weniger nach Reranking behalten
uv run python scripts/interactive_chat.py --n-chunks 15 --rerank-top-n 3

# Bestehende Chat-Session fortführen (thread_id aus Langfuse / vorherigem Run)
uv run python scripts/interactive_chat.py --thread-id abc-123
```

---

## Tests

### Unit- und Integrationstests

```bash
# Alle Unit-Tests (schnell, keine externen APIs)
uv run pytest src/tests/llms/ -v

# Nur Integrationstests (rufen echte LLM-API auf)
uv run pytest -m integration -v

# Integrationstests überspringen
uv run pytest -m "not integration" -v

# Einzelne Testdatei
uv run pytest src/tests/llms/test_no_answer_logic.py -v
uv run pytest src/tests/llms/test_router_classification_integration.py -v
uv run pytest src/tests/llms/test_language_detector_integration.py -v

# Reranker-Regressionstest gegen echtes Modell (BGE) + echten Index (Azure Semantic)
# Ergänzt test_reranker_node.py (nur Mocks) und benchmark.py (manueller Vergleich,
# kein Gate) um einen automatisierten NDCG@5-Mindestwert auf einem festen
# Datensatz-Ausschnitt. Setzt evaluation/dataset/queries.jsonl voraus (s. u.).
uv run pytest src/tests/llms/test_reranker_regression.py -v -m integration
```

---

## Reranker Evaluation

### 0. Fragen aus dem Index generieren (empfohlen für alle Level-Fragen)

Sampelt echte Chunks aus dem Search-Index und lässt ein LLM realistische
Lernenden-Fragen erzeugen — **mit aktuellen course_id/module_id direkt aus dem
Index**, damit nichts durch Re-Ingests veraltet. Kinds spiegeln die drei
Produktions-Scopes plus zwei Sondertypen:

- `drupal_level` — unscoped Besucherfrage (ki-campus.org)
- `course_level` — Frage im Kurs (course_id), über Kurse gestreut
- `module_level` — Frage im Modul (course_id+module_id), über Module gestreut
- `comparison` — braucht zwei Drupal-Dokumente (Ex-multi_hop-Fragetyp)
- `negative` — plausibel, aber nicht abgedeckt → Ground Truth für die
  No-Answer-/min_score-Kalibrierung

```bash
# Default: 8 drupal + 8 course + 8 module + 4 comparison + 4 negative
uv run python -m evaluation.dataset.generate_questions

# Anderer Index / andere Mengen
uv run python -m evaluation.dataset.generate_questions --index kic-content --n-course 12 --n-module 12

# Weitere Sprachen im Round-Robin
uv run python -m evaluation.dataset.generate_questions --languages de,en,es,fr
```

Ergebnis: `evaluation/dataset/generated_queries.json` → per `--queries-file` in Schritt 1 nutzen.

### 1. Dataset erstellen

Retrievet die Top-`--n-chunks` (Default 30) aus dem echten Index und lässt ein
LLM Relevanz-Labels vergeben (0/1/2). Das eingebaute Query-Set ist ein bewusst
kompakter kuratierter Kern (**36 Fragen**: Plattform-, Kurz-, Umgangssprache-,
Konzept-, Vergleichs-, Mehrsprachen- und No-Answer-Fragen, `kind`-Feld) — alle
Kurs-/Modul-Level-Fragen kommen per `--generate` aus dem Index (Schritt 0 läuft
dann automatisch mit). Typischer Gesamtumfang: 36 kuratiert + ~32 generiert
≈ 70 Queries. Ergebnis: `evaluation/dataset/queries.jsonl`

**Empfohlener One-Shot** (nach jedem Re-Ingest):

```bash
uv run python -m evaluation.dataset.create_dataset --generate --regenerate
```

Zuverlässigkeit: Jeder fertige Record wird sofort geschrieben und geflusht;
eine fehlgeschlagene Query wird einmal wiederholt und sonst übersprungen (am
Ende aufgelistet) — ein einzelner API-Fehler killt den Lauf nicht mehr.
**Abgebrochen/teilweise fehlgeschlagen?** Gleicher Befehl mit `--append` statt
`--regenerate` — bereits gelabelte Queries werden übersprungen, die generierten
Fragen kommen aus dem Cache (`generated_queries.json`), es wird also exakt
dasselbe Fragen-Set vervollständigt:

```bash
uv run python -m evaluation.dataset.create_dataset --generate --append
```

> Aufwand: ~70 Queries × 30 Chunks ≈ 2.100 Judge-Calls. Mit `--model Azure-Mini`
> deutlich günstiger; für einen Probelauf `--limit 5`.

Weitere Varianten:

```bash
# Nur der kuratierte Kern (ohne generierte Fragen)
uv run python -m evaluation.dataset.create_dataset

# Erstmal testen mit 5 Queries
uv run python -m evaluation.dataset.create_dataset --limit 5

# Günstigeres Mini-Modell für die Bewertung nutzen
uv run python -m evaluation.dataset.create_dataset --generate --model Azure-Mini

# Eigenes Fragen-File statt --generate (z. B. mit angepassten Mengen aus Schritt 0)
uv run python -m evaluation.dataset.create_dataset --queries-file evaluation/dataset/generated_queries.json --append

# Kleinerer Kandidaten-Pool (schneller/billiger, aber keine Pool-Experimente möglich)
uv run python -m evaluation.dataset.create_dataset --n-chunks 10
```

### 2. Benchmark ausführen

Vergleicht alle Reranker (Baseline, LLM, Azure Semantic, BGE large, BGE small)
auf dem Dataset. Gibt Tabellen mit NDCG@k, Recall@k, MRR, Precision@k, Latenz
und Kosten aus — gesamt, pro Sprache und als NDCG-Matrix pro Query-Kind.

Fairness-Mechaniken:

- **Echter Request-Scope**: `course_id`/`module_id` aus dem Dataset-Record
  werden explizit an index-suchende Backends (Azure Semantic) übergeben — der
  Scope wird nie aus Chunk-Metadaten geraten.
- **Union-Pool-Scoring (TREC-Pooling)**: Pro Query laufen erst ALLE Systeme;
  ungelabelte zurückgegebene Chunks werden per LLM nachgelabelt (Cache:
  `evaluation/dataset/judge_cache.jsonl`); danach werden alle Systeme gegen
  denselben Union-Kandidaten-Pool bewertet — NDCG-/Recall-Nenner sind identisch.
- **Marginale Semantic-Latenz**: Azure Semantic repräsentiert das integrierte
  Setup (Semantic-Ranking im Retrieval-Call selbst). Der Benchmark misst
  zusätzlich eine plain Hybrid-Suche pro Query und weist die Differenz aus —
  produktiv ersetzt der semantische Call das Retrieval, er kommt nicht dazu.

**Empfohlener Entscheidungs-Lauf** (produktionsnah, rate-limit-freundlich):

```bash
uv run python -m evaluation.benchmark --pool-sizes 10,30 --runs 1 --cooldown 2 \
  --model Gemma4 --judge-model Azure-Fallback --output results_fair.json
```

`--runs 1` reicht: Bei ~70 Queries gibt es pro Backend ohnehin 70 Latenz-Samples
für p50/p95 — 3 Runs verdreifachen nur die GWDG-Last. `--cooldown 2` legt 2 s
Pause hinter jeden LLM-Reranker-Call, damit das GWDG-Rate-Limit nicht in
429/120s-Retry-Schleifen läuft. Der LLM-Reranker läuft im Benchmark **strict**:
Ein endgültig fehlgeschlagener LLM-Call zählt als Fehler (Spalte `errors`),
statt still als Passthrough-Ergebnis in seine Metriken einzugehen (in
Produktion bleibt der Passthrough-Fallback aktiv). Langfuse-Tracing ist im
Benchmark deaktiviert.

Weitere Varianten:

```bash
# Standard-Benchmark (alle Reranker, voller Pool)
uv run python -m evaluation.benchmark

# Nur eine Teilmenge (schnelle Iteration — für die finale Entscheidung alle
# Systeme in EINEM Lauf vergleichen, das Union-Ideal gilt nur innerhalb eines Laufs)
uv run python -m evaluation.benchmark --rerankers no_rerank,azure_semantic,bge_large

# Ohne On-the-fly-Judging (out-of-pool Chunks zählen dann als 0)
uv run python -m evaluation.benchmark --no-judge-unlabeled

# Ohne Plain-Search-Baseline (spart Search-Calls, keine marginale Latenz-Angabe)
uv run python -m evaluation.benchmark --no-latency-baseline
```

**Beispielausgabe:**

```
Benchmarking 5 reranker(s) on 51 queries (3 runs each)
Languages: de=33, en=18

--- Overall ---
Reranker                      NDCG@5     Recall@5   MRR      P@5      Latenz p50  Latenz p95  Kosten/1k €
----------------------------  ---------  ---------- -------- -------- ----------- ----------- -----------
no_rerank (baseline)          0.412      0.365      0.531    0.380    0.1ms       0.2ms       0.0000
BGE (bge-reranker-v2-m3)      0.631      0.512      0.724    0.560    68.3ms      142.1ms     0.0000
BGE (bge-reranker-base)       0.598      0.488      0.689    0.531    31.2ms      67.4ms      0.0000
Azure Semantic                0.644      0.590      0.741    0.572    210.5ms     380.2ms     0.0900
LLM Reranker                  0.587      0.470      0.678    0.521    1840.2ms    3120.4ms    0.0480

Azure Semantic — Latenz-Einordnung (integriertes Setup):
  volle semantische Suche p50: 210 ms
  plain Hybrid-Suche p50 (gleicher Scope): 95 ms
  → marginale Semantic-Latenz ≈ 115 ms (der semantische Call ERSETZT das Retrieval, er kommt nicht dazu)
```

---

## Index-Management (scripts/)

### Index-Inhalt prüfen

```bash
# Dokument- und Token-Übersicht pro Source und Type
uv run python scripts/vectordb/check_index.py

# Eingebetteten Text pro Modul anzeigen (Extraktionsqualität prüfen)
uv run python scripts/vectordb/inspect_index_content.py --index kic-content
uv run python scripts/vectordb/inspect_index_content.py --index kic-content --course-ids 387 344
uv run python scripts/vectordb/inspect_index_content.py --index kic-content --short-only  # nur kurze/leere Chunks
```

### Zwei Indizes vergleichen

Zeigt Unterschiede zwischen zwei Indizes (Dokumentanzahl, Sources, fehlende Einträge).

```bash
# Default: aichat vs. kic-content
uv run python scripts/vectordb/compare_indexes.py

# Andere Indizes
uv run python scripts/vectordb/compare_indexes.py --a aichat --b kic-content
```

### Moodle-Abdeckung prüfen

Vergleicht Moodle-Module mit dem Index — zeigt welche Module fehlen oder leer sind.

```bash
# Alle Kurse im Index prüfen
uv run python scripts/loaders/audit_moodle_coverage.py --index kic-content

# Bestimmte Kurse prüfen
uv run python scripts/loaders/audit_moodle_coverage.py --index kic-content --course-ids 387 344 406
```

### Index-Schema aktualisieren

Wendet das aktuelle Schema auf einen bestehenden Index an (kein Datenverlust).

```bash
uv run python scripts/vectordb/update_index_schema.py
uv run python scripts/vectordb/update_index_schema.py --index my-other-index
```

### Index zurücksetzen

```bash
# Eine Source löschen (andere Sources bleiben)
uv run python scripts/vectordb/reset_index.py --source Moodle
uv run python scripts/vectordb/reset_index.py --source Moochup
uv run python scripts/vectordb/reset_index.py --source Drupal

# Kompletten Index löschen und neu anlegen (Vorsicht!)
uv run python scripts/vectordb/reset_index.py --all
```

### Test-Index befüllen

Kleinen Test-Index mit wenigen Kursen befüllen — ohne Produktions-Index anzufassen.

```bash
# Standard: 3 Moodle-Kurse in 'kic-content'
uv run python scripts/vectordb/ingest_test_index.py

# Mehr Kurse
uv run python scripts/vectordb/ingest_test_index.py --courses 5

# Index vorher zurücksetzen (testet Clean-Run)
uv run python scripts/vectordb/ingest_test_index.py --reset
```

---

## Lokaler Ingest (Hintergrund)

Vollständigen Ingest lokal starten — läuft im Hintergrund weiter wenn der Bildschirm gesperrt wird.
`caffeinate -i` verhindert, dass macOS in den Ruhemodus geht (Ruhemodus trennt SSL-Verbindungen zu Azure).

```bash
nohup caffeinate -i uv run python src/loaders/get_data.py > logs/ingest_local_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $!
```

Log live verfolgen:

```bash
tail -f logs/ingest_local_*.log
```

**Env-Variablen zum Steuern des Runs:**

```bash
# Nur bestimmte Sources ingesten
RUN_SOURCES=MOODLE,DRUPAL nohup caffeinate -i uv run python src/loaders/get_data.py > logs/ingest_local_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Moodle ab Kurs N starten (Resume nach Abbruch — überspringt Stale-Deletion!)
MOODLE_COURSE_OFFSET=150 nohup caffeinate -i uv run python src/loaders/get_data.py > logs/ingest_local_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Moodle auf N Kurse begrenzen (zum Testen)
MOODLE_COURSE_LIMIT=5 uv run python src/loaders/get_data.py
```

> **Hinweis Resume:** Ohne Offset einfach neu starten — bereits hochgeladene Dokumente werden per Content-Hash erkannt und übersprungen. Mit `MOODLE_COURSE_OFFSET` wird die Stale-Deletion automatisch deaktiviert, damit übersprungene Kurse nicht aus dem Index gelöscht werden.

---

## Azure Ingest (Produktion)

Startet / überwacht den Durable Function Orchestrator für den vollständigen Ingest.

```bash
# Ingest starten
uv run python scripts/vectordb/start_ingest.py

# Status prüfen
uv run python scripts/vectordb/check_ingest_status.py

# Ingest abbrechen
uv run python scripts/vectordb/terminate_ingest.py
```

Setzt `DATALOADER_FUNCTION_APP_URL` und `DATALOADER_FUNCTION_KEY` in der `.env` voraus.

---

## Drupal-Content manuell laden

Einzelne Drupal-PageTypes in den Index laden (lokal, ohne Azure Function).

```bash
# Im Script scripts/vectordb/ingest_drupal_courses.py PAGE_TYPE anpassen, dann:
uv run python scripts/vectordb/ingest_drupal_courses.py
```

PageType-Optionen: `COURSE`, `BLOGPOST`, `PAGE`, `ABOUT_US`, `SPEZIAL`
