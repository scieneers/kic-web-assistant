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
```

---

## Reranker Evaluation

### 0. (Optional) Fragen aus dem Index generieren

Statt nur handkuratierter Fragen: sampelt echte Chunks aus dem Search-Index und
lässt ein LLM realistische Lernenden-Fragen erzeugen — `grounded` (aus einem
Dokument beantwortbar), `comparison` (braucht zwei Dokumente; der Ex-multi_hop-Fragetyp)
und `negative` (plausibel, aber nicht abgedeckt → Ground Truth für die
No-Answer-/min_score-Kalibrierung).

```bash
# 30 grounded + 10 comparison + 10 negative aus dem Index kic-content
uv run python -m evaluation.dataset.generate_questions --index kic-content

# Kleiner Testlauf
uv run python -m evaluation.dataset.generate_questions --index kic-content --n-grounded 4 --n-comparison 2 --n-negative 2
```

Ergebnis: `evaluation/dataset/generated_queries.json` → per `--queries-file` in Schritt 1 nutzen.

### 1. Dataset erstellen

Retrievet die Top-`--n-chunks` (Default 30) aus dem echten Index und lässt ein
LLM Relevanz-Labels vergeben (0/1/2). Das eingebaute Query-Set umfasst 106 Fragen
(DE/EN) inkl. Vergleichs-, Kurz-, Umgangssprache-, Plattform-, Domänen- und
No-Answer-Fragen (`kind`-Feld). Ergebnis: `evaluation/dataset/queries.jsonl`

> Aufwand: 106 Queries × 30 Chunks ≈ 3.200 Judge-Calls. Mit `--model Azure-Mini`
> deutlich günstiger; für einen Probelauf `--limit 5`.

```bash
# Vollständiges Dataset neu erstellen
uv run python -m evaluation.dataset.create_dataset

# Erstmal testen mit 5 Queries
uv run python -m evaluation.dataset.create_dataset --limit 5

# Nur neue Queries hinzufügen, bestehende Labels behalten
uv run python -m evaluation.dataset.create_dataset --append

# Günstigeres Mini-Modell für die Bewertung nutzen
uv run python -m evaluation.dataset.create_dataset --model Azure-Mini

# Generierte Fragen statt der eingebauten nutzen (siehe Schritt 0)
uv run python -m evaluation.dataset.create_dataset --queries-file evaluation/dataset/generated_queries.json --append

# Kleinerer Kandidaten-Pool (schneller/billiger, aber keine Pool-Experimente möglich)
uv run python -m evaluation.dataset.create_dataset --n-chunks 10
```

### 2. Benchmark ausführen

Vergleicht alle Reranker (Baseline, LLM, Azure Semantic, BGE large, BGE small)
auf dem Dataset. Gibt eine Tabelle mit NDCG@k, MRR, Precision@k, Latenz und Kosten aus.

Fairness-Mechaniken: NDCG nutzt den vollen gelabelten Kandidaten-Pool als
Ideal (nicht nur die zurückgegebene Liste), und Chunks, die Azure Semantic
außerhalb des Pools findet, werden per LLM on-the-fly nachgelabelt
(Cache: `evaluation/dataset/judge_cache.jsonl`).

```bash
# Standard-Benchmark (top-5, 3 Runs pro Query für stabile Latenz)
uv run python -m evaluation.benchmark

# Pool-Größen-Experiment: Lohnt retrieve_top_n=30 statt 10?
# (Dataset muss mit --n-chunks >= 30 erstellt sein)
uv run python -m evaluation.benchmark --pool-sizes 10,30

# Ohne On-the-fly-Judging (out-of-pool Chunks zählen dann als 0)
uv run python -m evaluation.benchmark --no-judge-unlabeled

# Ergebnisse als JSON speichern
uv run python -m evaluation.benchmark --output results.json

# Schneller: nur 1 Run pro Query
uv run python -m evaluation.benchmark --runs 1
```

**Beispielausgabe:**

```
Benchmarking 5 reranker(s) on 51 queries (3 runs each)
Languages: de=33, en=18

--- Overall ---
Reranker                      NDCG@5     MRR        P@5        Latenz p50  Latenz p95  Kosten/1k €
----------------------------  ---------- ---------- ---------- ------------ ------------ -------------
no_rerank (baseline)          0.412      0.531      0.380      0.1ms        0.2ms        0.0000
BGE (bge-reranker-v2-m3)      0.631      0.724      0.560      68.3ms       142.1ms      0.0000
BGE (bge-reranker-base)       0.598      0.689      0.531      31.2ms       67.4ms       0.0000
azure_semantic                0.644      0.741      0.572      210.5ms      380.2ms      0.0120
llm                           0.587      0.678      0.521      1840.2ms     3120.4ms     0.0480
```

---

## Index-Management (scripts/)

### Index-Inhalt prüfen

```bash
# Dokument- und Token-Übersicht pro Source und Type
uv run python scripts/vectordb/check_index.py

# Eingebetteten Text pro Modul anzeigen (Extraktionsqualität prüfen)
uv run python scripts/vectordb/inspect_index_content.py --index aitestlaurien
uv run python scripts/vectordb/inspect_index_content.py --index aitestlaurien --course-ids 387 344
uv run python scripts/vectordb/inspect_index_content.py --index aitestlaurien --short-only  # nur kurze/leere Chunks
```

### Zwei Indizes vergleichen

Zeigt Unterschiede zwischen zwei Indizes (Dokumentanzahl, Sources, fehlende Einträge).

```bash
# Default: aichat vs. kic-content
uv run python scripts/vectordb/compare_indexes.py

# Andere Indizes
uv run python scripts/vectordb/compare_indexes.py --a aichat --b aitestlaurien
```

### Moodle-Abdeckung prüfen

Vergleicht Moodle-Module mit dem Index — zeigt welche Module fehlen oder leer sind.

```bash
# Alle Kurse im Index prüfen
uv run python scripts/loaders/audit_moodle_coverage.py --index aitestlaurien

# Bestimmte Kurse prüfen
uv run python scripts/loaders/audit_moodle_coverage.py --index aitestlaurien --course-ids 387 344 406
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
# Standard: 3 Moodle-Kurse in 'aitestlaurien'
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
