# KIC-Campus Web Assistant — Architektur & RAG-System

> KI-gestützter Chat-Assistent für ki-campus.org, der Nutzerfragen anhand von Kursinhalten und Website-Informationen beantwortet.

---

## Inhaltsverzeichnis

1. [Projektstruktur](#1-projektstruktur)
2. [High-Level Architektur](#2-high-level-architektur)
3. [RAG-Flow: Von der Frage zur Antwort](#3-rag-flow-von-der-frage-zur-antwort)
4. [Routing — Welcher Subgraph wird gewählt?](#4-routing--welcher-subgraph-wird-gewählt)
5. [Subgraphen im Detail](#5-subgraphen-im-detail)
6. [Vektordatenbank (Azure AI Search)](#6-vektordatenbank-azure-ai-search)
7. [Dateningestion](#7-dateningestion)
8. [API-Layer](#8-api-layer)
9. [Streaming-Architektur](#9-streaming-architektur)
10. [Konversationspersistenz](#10-konversationspersistenz)
11. [Observability (Langfuse)](#11-observability-langfuse)
12. [Frontend (Streamlit)](#12-frontend-streamlit)
13. [Deployment](#13-deployment)
14. [Konfiguration & Umgebungsvariablen](#14-konfiguration--umgebungsvariablen)
15. [Key-File-Referenz](#15-key-file-referenz)
16. [Bekannte Issues & Roadmap](#16-bekannte-issues--roadmap)

---

## 1. Projektstruktur

```
kic-web-assistant/
├── src/
│   ├── api/            # FastAPI REST Backend
│   ├── frontend/       # Streamlit UI
│   ├── llm/            # LLM-Orchestrierung (LangGraph)
│   │   ├── assistant.py          # Haupt-Orchestrator
│   │   ├── graphs/               # Subgraph-Definitionen
│   │   ├── objects/              # Retriever, Reranker, etc.
│   │   ├── state/                # GraphState-Modell
│   │   └── tools/                # Node-Implementierungen
│   ├── vectordb/       # Azure AI Search Integration
│   ├── loaders/        # Datenextraktion & -ingestion
│   ├── tests/
│   └── env.py          # Environment-Konfiguration
├── documentation/
├── rag_eval/           # RAG-Evaluierungsframework
├── docker-compose.yaml # Lokale Services (Langfuse + PostgreSQL)
├── pyproject.toml      # Python-Abhängigkeiten (uv)
└── Taskfile.yml        # Build/Deploy-Automatisierung
```

---

## 2. High-Level Architektur

```
┌─────────────────────────────────────────────────────────┐
│                      Nutzer / KI-Campus                  │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP (REST)
                     ▼
┌─────────────────────────────────────────────────────────┐
│               FastAPI REST API  (src/api/)               │
│  POST /api/chat   POST /api/chat/stream   /api/feedback  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         KICampusAssistant (src/llm/assistant.py)         │
│              LangGraph State Machine                     │
│                                                          │
│  ┌──────────────┐   ┌────────────────────────────────┐  │
│  │  contextualize│   │  Subgraph (je nach Routing):   │  │
│  │  _and_route  │──▶│  no_vectordb / simple_hop /    │  │
│  └──────────────┘   │  summarize / socratic /        │  │
│                     │  socratic_v2                   │  │
│                     └────────────────────────────────┘  │
└─────────┬───────────────────────┬───────────────────────┘
          │                       │
          ▼                       ▼
┌──────────────────┐   ┌─────────────────────────────────┐
│  Azure AI Search │   │  Azure OpenAI / GWDG API        │
│  (Vektordatenbank│   │  (LLM + Embeddings)             │
│   + BM25 Hybrid) │   └─────────────────────────────────┘
└──────────────────┘
          │
          ▼
┌──────────────────┐
│    Langfuse      │
│ (Observability)  │
└──────────────────┘
```

---

## 3. RAG-Flow: Von der Frage zur Antwort

```
1. Nutzer sendet Anfrage (REST oder Streamlit)
   │
2. ChatRequest validieren (course_id, module_id prüfen)
   │
3. KICampusAssistant lädt Checkpoint (thread_id)
   │   → Existierender thread_id: Konversationshistorie laden
   │   → Neuer thread_id: Frischen State anlegen
   │
4. contextualize_and_route
   │   → Szenario klassifizieren (LLM-basiert)
   │   → Query ggf. mit Konversationshistorie umformulieren
   │
5. Routing zu Subgraph ──────────────────────────────────┐
   │                                                      │
   ├─ no_vectordb ──────► Sprache erkennen              │
   │                       ► Direkte LLM-Antwort          │
   │                                                      │
   ├─ simple_hop ───────► Retrieval (Hybrid, Top-10)    │
   │                       ► Reranking (konfigurierbar, Top-5) │
   │                       ► Sprache erkennen (parallel)  │
   │                       ► Antwort generieren           │
   │                       ► Zitate parsen               │
   │                                                      │
   ├─ summarize ────────► Vollen Scope laden            │
   │                       (retrieve_all, kein Top-k)     │
   │                       ► Sprache erkennen (parallel)  │
   │                       ► Zusammenfassung generieren   │
   │                       ► Zitate parsen               │
   │                                                      │
   ├─ socratic (v1) ────► Lernziel ermitteln            │
   │                       ► Diagnose                      │
   │                       ► Hinweise / Reflektion         │
   │                                                      │
   └─ socratic_v2 ──────► opening / core / consolidation│
                           (Move-Policy, s. 5.5)           │
                                                          │
6. State persistieren (Checkpoint) ◄──────────────────────┘
   │
7. Antwort zurückgeben (sync oder Token-Streaming)
   │
8. Optional: Nutzerfeedback via /api/feedback → Langfuse
```

---

## 4. Routing — Welcher Subgraph wird gewählt?

Die Klassifizierung erfolgt **LLM-basiert** in `src/llm/tools/contextualize.py`.

| Szenario | Beschreibung | Beispiel |
|----------|--------------|---------|
| `no_vectordb` | Konversationelle Frage, kein Kontext nötig | "Was kann KI-Campus?" |
| `simple_hop` | Standard-RAG: Default für alle Wissensfragen (auch Multi-Konzept-/Vergleichsfragen) | "Worum geht es im Kurs X?" |
| `summarize` | Zusammenfassungs-Anfrage über den gewählten Scope; lädt den **vollen** Inhalt statt Top-k | "Fasse dieses Modul zusammen" |
| `socratic` | Geführtes Lernen v1 (Feature-Flag `ENABLE_SOCRATIC`; Start via Request-Flag `start_socratic` oder Trigger-Phrase, z. B. "unterstütze mich beim lernen") | Student arbeitet sich durch Stoff |
| `socratic_v2` | Lernmodus v2 (Feature-Flag `ENABLE_SOCRATIC_V2`; Start via `start_socratic_v2` oder "starte lernmodus v2"); braucht Kurs-/Modul-Scope | s. 5.5 |
| `exit_complete` | Sonderfall: aktive Socratic-Session beenden (v1 und v2) | "exit" / "beende den Lernmodus" |

> **Hinweis 1:** Der frühere `multi_hop`-Subgraph (Query-Dekomposition → paralleles Retrieval → Synthese) wurde im Zuge der Single-Hop-Umstellung entfernt. Alle Retrieval-Fragen — auch Vergleichsfragen — laufen über `simple_hop` (Hybrid-Retrieval + Reranking).
>
> **Hinweis 2:** `summarize` ohne gesetzten `course_id`/`module_id`-Scope wird zu `no_vectordb` heruntergestuft (`contextualize.py`). Bei aktiver Socratic-Session (v1 oder v2, im Checkpoint verfolgt) wird die Klassifikation übersprungen und direkt der jeweilige Subgraph fortgesetzt; `socratic_v2` hat Vorrang vor v1.

---

## 5. Subgraphen im Detail

### 5.1 simple_hop (`src/llm/graphs/simple_hop.py`)

```
START ─┬─► retrieve_chunks ──► rerank_chunks ─┐
       └─► detect_language ───────────────────┴─► generate_answer ─► parse_citations ─► END
```

- **Retrieval:** Azure AI Search Hybrid (BM25 + Vektor), Top-10
- **Reranking:** Backend per `RERANKER_TYPE` (Env) wählbar — `azure_semantic` (Default), `llm`, `bge`.
  Bei `azure_semantic` läuft das Reranking **integriert**: Azures Semantic Ranker rescort
  serverseitig innerhalb des Retrieval-Calls (Top-50-Fenster), `rerank_chunks` schneidet dann
  nur noch auf Top-5 und wendet `MIN_RERANKER_SCORE` an — kein zweiter Search-Roundtrip.
  Index-suchende Backends erhalten den Request-Scope (`course_id`/`module_id`) explizit
  aus der `runtime_config`, nie aus Chunk-Metadaten.
- **Sprache:** Parallel zum Retrieval, kein Latenz-Overhead
- **Antwort:** Mit Quellenangaben als `[docN]`-Marker, dann in Markdown-Links umgewandelt

### 5.2 no_vector_db (`src/llm/graphs/no_vector_db.py`)

```
START ─► detect_language ─► direct_answer ─► END
```

Kein Retrieval, direkter LLM-Call. Für allgemeine Fragen über KI-Campus.

### 5.3 socratic — Lernmodus v1 (`src/llm/graphs/socratic.py`)

Der Lernmodus aus dem Studierendenprojekt (koexistiert mit v2, s. 5.5). Drei Phasen: `contract` → `diagnose` → `core`

- Verfolgt `attempt_count` und `number_given_hints`
- Führt Studierende zur Antwort, statt sie direkt zu liefern
- Abbruch via Schlüsselwörter: "exit", "quit", "stop", "beende den lernmodus"

### 5.4 summarize (`src/llm/graphs/summarize.py`)

```
START ─┬─► retrieve_full_scope ─┐
       └─► detect_language ─────┴─► summarize_answer ─► parse_citations ─► END
```

- Lädt per `retriever.retrieve_all()` den **kompletten** Inhalt des gewählten Scopes (Kurs/Module) statt Top-k — Zusammenfassungen decken damit den ganzen Scope ab, kein Reranking.
- Antwortgenerierung über `SummaryAnswerer` (`src/llm/objects/summary_answerer.py`, Node in `src/llm/tools/summarize.py`).
- Nur mit gesetztem Kurs-/Modul-Scope erreichbar (sonst Downgrade zu `no_vectordb`, s. Abschnitt 4).

### 5.5 socratic_v2 — Lernmodus v2 (`src/llm/graphs/socratic_v2.py` + `src/llm/tools/socratic_v2_*.py`)

Neu konzipierter sokratischer Tutor (Konzept: `KONZEPT_SOKRATISCHER_LERNASSISTENT.md`, Vergleich zu v1: `VERGLEICH_SOKRATISCH_V1_V2.md`). Drei Phasen, im Checkpoint über `socratic_v2_phase` verfolgt:

```
opening ──► core (Move-Policy-Loop) ──► consolidation
```

- **opening** (`socratic_v2_opening.py`): lädt den Modulinhalt via `retrieve_all()`, extrahiert Lernziele/Schlüsselkonzepte per LLM, begrüßt mit Sessionziel. Bei `module_id`-Liste werden **alle gewählten Module gleichrangig** als gemeinsamer Inhaltspool behandelt.
- **core** (`socratic_v2_core.py`): eigener Retrieval+Rerank-Pfad, dann Move-Policy mit 10 Zügen (u. a. FRAGE, HINT, MICRO_EXPLAIN, QUIZ mit echten H5P-Quizfragen und deterministischer Auswertung, FEHLER_FINDEN, ENCOURAGE). Deterministische Leitplanken: 3 Fragen ohne Fortschritt → HINT, 2 Hints pro Konzept → MICRO_EXPLAIN.
- **consolidation**: Zusammenfassung + Feedback, danach State-Reset.
- Benötigt zwingend Kurs-/Modul-Scope; nutzt ein erweitertes History-Limit (`CHAT_HISTORY_LIMIT_SOCRATIC_V2`, Default 40, s. Abschnitt 10). Eigene State-Felder in `src/llm/state/models.py`, Routing-Konstanten in `src/llm/state/socratic_v2_routing.py`.

---

## 6. Vektordatenbank (Azure AI Search)

**Client:** `src/vectordb/azure_search.py`

### Index-Schema

| Feld | Typ | Funktion |
|------|-----|----------|
| `id` | String (Key) | Sanitized Node-ID |
| `text` | Searchable (DE-Analyzer) | Hauptinhalt |
| `fullname` | Searchable | Ressourcenname |
| `title` | Searchable | Seitentitel |
| `source` | Filterable | "Drupal" oder "Moodle" |
| `type` | Filterable | Inhaltstyp, s. Tabelle unten |
| `course_id` | Filterable (Int64) | Moodle-Kurs |
| `module_id` | Filterable (Int64) | Moodle-Modul |
| `url` | Filterable | Link zur Originalquelle |
| `is_important` | Filterable (bool) | Priorisierung |
| `date_created` | Filterable (String) | Erstellungsdatum |
| `modname` | Filterable/Facetable (String) | Moodle-Modultyp (z. B. `h5pactivity`, `page`) — aus `metadata_json` herausgezogen für typisiertes Retrieval |
| `h5p_content_type` | Filterable (String) | H5P-Inhaltstyp (z. B. `QuestionSet`) |
| `source_doc_key` | Filterable | Stabiler Dokument-Identifier für Change Detection |
| `content_hash` | String | Content-Fingerprint für Change Detection |
| `dense` | Vector (Dim. nach Embedding-Modell, aktuell 3072) | HNSW, Cosine Distance |
| `metadata_json` | String | Vollständige Metadaten |

Zusätzlich trägt der Index eine **Semantic Configuration** („default") — Voraussetzung für den `azure_semantic`-Reranker.

### Inhaltstypen (`type`-Feld)

Neben den klassischen Typen (`Resource`, `Module`, `Course`, …) existieren:

| Typ | Bedeutung |
|-----|-----------|
| `EmptyModule` | Modul ohne extrahierbaren Inhalt (nur Titel) — im normalen Retrieval ausgeschlossen, außer bei explizitem `module_id`-Filter (deterministischer Fallback-Hinweis) |
| `QuizItem` | Einzelne Quizfrage inkl. Lösung — **unbedingt aus dem normalen RAG ausgeschlossen** (`RETRIEVAL_EXCLUDED_TYPES` in `src/vectordb/doc_types.py`), damit keine Musterlösungen in Antworten gelangen; zugreifbar nur über den typisierten Pfad |
| `GlossaryEntry` | Glossareintrag (Begriff + Definition) |
| `Flashcard` | Lernkarte |
| `Transcript` | Video-Transkript-Abschnitt (mit Zeitstempel für `#t=`-Deep-Links) |
| `BookChapter` | Kapitel eines Moodle-Books |

**Typisierter Zugriffspfad:** `retriever.retrieve_items()` / `lookup_glossary()` (`src/llm/objects/retriever.py`) — einziger Weg zu `QuizItem`-Dokumenten, genutzt vom Lernmodus v2 (QUIZ-Move).

### Suchalgorithmus

```
Hybrid Search = BM25 (Keyword) + Dense Vector
                  └──────────────┘
                         │
              Reciprocal Rank Fusion (RRF)
                         │
                    Top-K Ergebnisse
```

- Embedding-Modell: Azure OpenAI (Dimension wird beim Ingest dynamisch erkannt, aktuell 3072)
- Optional: OData-Filter für `course_id` / `module_id`

---

## 7. Dateningestion

**Einstiegspunkt:** `src/loaders/get_data.py` (`Fetch_Data`-Klasse)

### Datenquellen

| Quelle | Datei | Inhalt |
|--------|-------|--------|
| Moodle | `moodle.py` | Kursinhalte via REST-API |
| Drupal | `drupal.py` | Website (Kurse, Blog, Seiten) |
| Moochup | `moochup.py` | Kurskatalog |
| YouTube | `youtube.py` | Video-Transkripte |
| Vimeo | `vimeo.py` | Video-Transkripte |
| PDF | `pdf.py` | Dokumentenextraktion + Chunking |
| Audio | `audio.py` | Transkripte via Vosk ASR |

### Ingestion-Pipeline

```
Quell-API / CMS
      │
      ▼
Extraktion (quellspezifische Loader)
      │
      ▼
Transformation (hierarchisches Chunking via helper.py)
      │
      ▼
Embedding (Batch-Verarbeitung, Azure OpenAI)
      │
      ▼
Upload (Batch-Upsert → Azure AI Search)
      │  Limits: max. 1.000 Docs/Batch, 14 MB
      ▼
Sanity Check (URL-Validierung etc.)
```

---

## 8. API-Layer

**Framework:** FastAPI (`src/api/rest.py`)

### Endpunkte

| Methode | Pfad | Beschreibung |
|---------|------|--------------|
| `POST` | `/api/chat` | Synchrone Anfrage → vollständige Antwort |
| `POST` | `/api/chat/stream` | Streaming (NDJSON, Token-by-Token) |
| `GET` | `/api/chat/history/{thread_id}` | Gespeicherten Verlauf laden (leere Liste bei unbekanntem Thread, kein 404) |
| `POST` | `/api/feedback` | Nutzerfeedback → Langfuse |
| `GET` | `/health` | Health-Check → `"OK"` |

### ChatRequest

```python
{
    user_query: SerializableChatMessage,
    thread_id: str | None,               # Konversations-ID (Persistenz)
    course_id: int | None,               # Moodle-Kursfilter
    module_id: int | list[int] | None,   # Einzelnes Modul ODER Liste mehrerer Module
                                         # desselben Kurses (gleichrangiger OR-Filter);
                                         # course_id ist Pflicht, sobald module_id gesetzt ist
    model: Models,                       # LLM-Auswahl (Default: GEMMA4_31B)
    start_socratic: bool = False,        # Lernmodus v1 starten (nur wirksam mit ENABLE_SOCRATIC)
    start_socratic_v2: bool = False      # Lernmodus v2 starten (nur wirksam mit ENABLE_SOCRATIC_V2)
}
```

Kurs- und Modul-IDs werden serverseitig auf Existenz geprüft (400 bei unbekannter ID); eine leere `module_id`-Liste wird zu `None` normalisiert.

### Verfügbare Modelle

| Enum-Wert | Beschreibung |
|-----------|--------------|
| `GEMMA4_31B` | Gemma 4 via GWDG-API (**Standard** in API & Frontend) |
| `AZURE_FALLBACK` | Azure-Modell, automatischer Fallback bei GWDG-Ausfall (Timeout + Unavailability-Fenster) |
| `MINI` | Hilfstasks: Spracherkennung, Lernziel-Extraktion (kostenoptimiert) |
| `LLAMA3` | Legacy-Alias → mappt auf Gemma4 (bis alle Frontends umgestellt sind) |

### Authentifizierung

- Header: `Api-Key`
- Validierung gegen `REST_API_KEYS`-Liste
- CORS: Beschränkt auf `ki-campus.org`, `moodle.ki-campus.org`, Staging-URLs

---

## 9. Streaming-Architektur

```
POST /api/chat/stream
      │
      ├─ Thread-ID + Trace-ID vorab anlegen
      │
      ├─ Worker-Thread startet:
      │    ├─ Langfuse Root-Trace setzen
      │    ├─ Graph ausführen mit TokenCallbackContext
      │    └─ Tokens via Callback → Queue
      │
      └─ Main-Thread yieldet aus Queue als NDJSON:

         {"type":"meta",  "thread_id":..., "response_id":...}  ← zuerst
         {"type":"token", "token":"Hallo"}                      ← mehrfach
         {"type":"token", "token":" Welt"}
         ...
         {"type":"final", "message":..., "thread_id":..., "response_id":...}
         {"type":"error", ...}                                  ← optional
```

**CitationStreamResolver** (`src/llm/streaming.py`): Löst `[docN]`-Marker **live während des Streamings** in klickbare Markdown-Links mit Anzeigetitel auf (Übergabe an die LLM-Schicht via ContextVar). Der ältere **CitationStreamFilter** (Marker nur ausblenden) bleibt für den `no_vectordb`-Pfad und als Fallback erhalten. **SmartStreamCallback** puffert den Stream-Anfang, um ein reines „NO ANSWER FOUND"-Sentinel abzufangen und durch die Fallback-Nachricht zu ersetzen — Sentinels erreichen die UI nie.

---

## 10. Konversationspersistenz

**Mechanismus:** LangGraph Checkpointer

| Umgebung | Checkpointer |
|----------|-------------|
| Dev + Produktion | `BoundedMemorySaver` (in-memory, erbt von `MemorySaver`; max. `MAX_CHAT_THREADS` = 500 Threads mit FIFO-Verdrängung, lock-geschützt) |
| Langfristige DB-Persistenz | *bewusst nicht umgesetzt (Angebots-Scope); LangGraph böte drop-in `SqliteSaver`/`PostgresSaver`* |

**Flow:**
```
Request mit thread_id
      ▼
graph.get_state(config) → lädt Checkpoint
      ▼
State mit neuer Anfrage aktualisieren
      ▼
Graph ausführen
      ▼
graph.update_state(config, values) → persistiert State
      ▼
Nächster Request mit gleicher thread_id lädt History
```

**History-Limit:** Letzte 6 Nachrichten (`CHAT_HISTORY_LIMIT`, Env-Var). Ausnahme: In aktiven Socratic-v2-Sessions gilt ein erweitertes Limit von 40 Nachrichten (`CHAT_HISTORY_LIMIT_SOCRATIC_V2`), damit die Lernsession-Dramaturgie den vollen Verlauf sieht.

**Endpoint:** `GET /api/chat/history/{thread_id}` liefert den gespeicherten Verlauf (für Sitzungs-Wiederherstellung nach Reload, s. `CHAT_PERSISTENCE.md`).

---

## 11. Observability (Langfuse)

**Was wird getrackt:**

- API-Requests (`/api/chat`, `/api/chat/stream`)
- LLM-Calls (Chat + Embeddings)
- Retrieval-Operationen
- Reranking
- Antwortgenerierung
- Zitate-Parsing
- Nutzerfeedback (Score 0–1 + Kommentar)

**Konfiguration:** `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`

**Lokales Setup:** Über `docker-compose.yaml` (Langfuse + PostgreSQL)

---

## 12. Frontend (Streamlit)

**Framework:** Streamlit (`src/frontend/frontend.py`) — reines **Test-Frontend** (produktive Oberflächen: Moodle/Drupal)

**Features:**
- Passwort-Gate via `FRONTEND_PASSWORD` (übersprungen, wenn nicht gesetzt — lokale Dev)
- Kurs-/Modul-Baum-Browser (Sidebar) mit Checkboxen: ganzer Kurs oder mehrere Module **eines** Kurses wählbar (→ `module_id`-Liste)
- Buttons „Lernmodus starten (v1)" / „✨ Lernmodus starten (v2)" (setzen `start_socratic`/`start_socratic_v2` für die nächste Nachricht)
- Moodle-Simulation: Session-ID anzeigen/laden/neu, `?thread_id=`-URL-Parameter simuliert localStorage über Reloads, 30-min-TTL clientseitig
- Chat-Interface mit Streaming-Antworten, Thinking-Indikator
- Feedback (Daumen hoch/runter + Kommentar) → `/api/feedback`
- LLM-Auswahl (Default `GEMMA4_31B`)

---

## 13. Deployment

### Container-Übersicht

| Komponente | Container | Port |
|-----------|-----------|------|
| FastAPI REST API | `rest-api` | 8000 |
| Streamlit Frontend | `app` | 8501 |
| Data Loader | `loader` | — |
| Langfuse UI | `langfuse` | 3000 |
| PostgreSQL | `postgres` | 5432 |

### Build & Deploy (Taskfile)

```bash
task build-api              # REST API Image bauen
task build-frontend         # Streamlit Frontend bauen
task build-loader           # Data Loader bauen
task push-latest-images     # Push to Azure Container Registry
task deploy-loader ENV=dev  # Deploy to Azure Functions
```

---

## 14. Konfiguration & Umgebungsvariablen

**Datei:** `src/env.py` (`EnvHelper`-Klasse)

**Prioritätsreihenfolge:** Env-Variablen > `.env`-Datei > Azure Key Vault > Pydantic-Defaults

```bash
# Azure OpenAI
AZURE_OPENAI_URL
AZURE_OPENAI_API_KEY
AZURE_FALLBACK_DEPLOYMENT / AZURE_FALLBACK_MODEL
AZURE_MINI_DEPLOYMENT / AZURE_MINI_MODEL
AZURE_OPENAI_EMBEDDER_DEPLOYMENT / MODEL

# GWDG API (Open Models)
GWDG_API_KEY
GWDG_URL
GWDG_MODEL

# Azure AI Search
AZURE_SEARCH_ENDPOINT
AZURE_SEARCH_INDEX

# Langfuse
LANGFUSE_HOST
LANGFUSE_PUBLIC_KEY
LANGFUSE_SECRET_KEY

# REST API
REST_API_URL
REST_API_KEYS          # kommagetrennte Liste gültiger API-Keys

# Feature-Flags
ENABLE_SOCRATIC        # Lernmodus v1 (Default: False)
ENABLE_SOCRATIC_V2     # Lernmodus v2 (Default: False)
FRONTEND_PASSWORD      # Passwort-Gate fürs Streamlit-Test-Frontend (Default: "UNSET" = aus)

# Retrieval / Reranking
RERANKER_TYPE          # llm | azure_semantic | bge (Default: azure_semantic)
MIN_RERANKER_SCORE     # Score-Schwelle 0–1 (Default: 0.5); darunter → No-Answer-Fallback

# Chat-/Betriebsparameter
MAX_CHAT_THREADS               # max. parallele Threads im BoundedMemorySaver (Default: 500)
CHAT_HISTORY_LIMIT             # Historie pro Turn (Default: 6)
CHAT_HISTORY_LIMIT_SOCRATIC_V2 # Historie in aktiven v2-Sessions (Default: 40)
GWDG_TIMEOUT_SECONDS           # Timeout vor Azure-Fallback (Default: 7)
GWDG_UNAVAILABLE_RESET_SECONDS # Unavailability-Fenster (Default: 300)

# Datenquellen
DATA_SOURCE_MOODLE_URL / TOKEN
DATA_SOURCE_MOOCHUP_HPI_URL / DATA_SOURCE_MOOCHUP_MOODLE_URL
DRUPAL_URL / CLIENT_ID / SECRET / USERNAME / PASSWORD / GRANT_TYPE
DRUPAL_AUTH_REQUIRED   # True → Loader bricht ohne gültige Drupal-Credentials hart ab (für Prod-Läufe)
VIMEO_PAT

# Ingestion-Schutz
STALE_DELETE_MIN_SEEN_RATIO  # Mindestanteil gesehener Bestand, sonst keine Stale-Deletion (Default: 0.5)

# Deployment
ENVIRONMENT            # DEV oder PRODUCTION
DEBUG_MODE
AUDIO_TRANSCRIPTION_ENABLED
```

---

## 15. Key-File-Referenz

| Datei | Zweck |
|-------|-------|
| `src/api/rest.py` | FastAPI-App, alle Endpunkte |
| `src/llm/assistant.py` | Haupt-Orchestrator (`KICampusAssistant`) |
| `src/llm/state/models.py` | State-Definition (`GraphState`) |
| `src/llm/graphs/simple_hop.py` | Standard-RAG-Subgraph |
| `src/llm/graphs/no_vector_db.py` | Konversationale Antworten |
| `src/llm/graphs/summarize.py` | Zusammenfassungs-Subgraph (voller Scope statt Top-k) |
| `src/llm/graphs/socratic.py` | Lernmodus v1 |
| `src/llm/graphs/socratic_v2.py` | Lernmodus v2 (Graph) |
| `src/llm/tools/socratic_v2_*.py` | v2-Nodes: opening / core / consolidation |
| `src/llm/state/socratic_v2_routing.py` | v2-Moves, Exit-Keywords, Routing-Konstanten |
| `src/vectordb/doc_types.py` | Inhaltstyp-Konstanten + `RETRIEVAL_EXCLUDED_TYPES` |
| `src/llm/streaming.py` | CitationStreamResolver / SmartStreamCallback |
| `src/llm/objects/retriever.py` | Retrieval-Logik (inkl. `retrieve_all`, `retrieve_items`, `lookup_glossary`) |
| `src/llm/objects/contextualizer.py` | Query-Kontextualisierung & Routing |
| `src/llm/objects/reranker.py` | LLM-basiertes Reranking |
| `src/llm/objects/rerankers/` | Pluggable Reranker-Backends (LLM, Azure Semantic, BGE, Passthrough) |
| `src/llm/objects/question_answerer.py` | Antwortgenerierung |
| `src/llm/objects/citation_parser.py` | `[docN]` → Markdown-Links |
| `src/llm/tools/contextualize.py` | Router-Node |
| `src/vectordb/azure_search.py` | Vektordatenbank-Client |
| `src/loaders/get_data.py` | Ingestion-Orchestrator |
| `src/loaders/moodle.py` | Moodle-Connector |
| `src/loaders/drupal.py` | Drupal-Connector |
| `src/frontend/frontend.py` | Streamlit UI |
| `src/env.py` | Umgebungskonfiguration |
| `docker-compose.yaml` | Lokale Dev-Services |
| `pyproject.toml` | Python-Abhängigkeiten |
| `Taskfile.yml` | Build/Deploy-Automatisierung |

---

## 16. Bekannte Issues & Roadmap

**Offene Punkte** (aus `TODO.md`):

| # | Issue | Priorität |
|---|-------|-----------|
| 1 | Azure Container Registry Auth: Admin → Service Principal migrieren | Mittel |
| 2 | Langfuse-Ersatz wird evaluiert | Niedrig |
| 3 | Model-Enum: `LLAMA3` → `GEMMA4_31B` umbenennen | Niedrig |
| 4 | Loader-Logs in Blob Storage: Connection String → Managed Identity | Mittel |
| 5 | Chat-Persistenz: In-Memory (`BoundedMemorySaver`) ist bewusster Scope; DB-Checkpointer (`SqliteSaver`/`PostgresSaver`) nur bei geänderter Anforderung | — (kein offener Punkt) |
| 6 | Managed Identity für alle Azure-Ressourcen | Mittel |

---

*Zuletzt aktualisiert: 16. Juli 2026 (Summarize-Modus, Lernmodus v2, strukturierte Inhaltstypen, Multi-Modul-Scope, aktualisierte Defaults & Env-Vars)*
