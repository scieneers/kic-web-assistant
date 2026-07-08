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
│  └──────────────┘   │  socratic                      │  │
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
   └─ socratic ─────────► Lernziel ermitteln            │
                           ► Diagnose                      │
                           ► Hinweise / Reflektion         │
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
| `socratic` | Geführtes Lernen (opt-in via `enable_socratic`) | Student arbeitet sich durch Stoff |
| `exit_complete` | Sonderfall: Socratic-Session beenden | "exit" / "beende den Lernmodus" |

> **Hinweis:** Der frühere `multi_hop`-Subgraph (Query-Dekomposition → paralleles Retrieval → Synthese) wurde im Zuge der Single-Hop-Umstellung entfernt. Alle Retrieval-Fragen — auch Vergleichsfragen — laufen über `simple_hop` (Hybrid-Retrieval + Reranking).

---

## 5. Subgraphen im Detail

### 5.1 simple_hop (`src/llm/graphs/simple_hop.py`)

```
START ─┬─► retrieve_chunks ──► rerank_chunks ─┐
       └─► detect_language ───────────────────┴─► generate_answer ─► parse_citations ─► END
```

- **Retrieval:** Azure AI Search Hybrid (BM25 + Vektor), Top-10
- **Reranking:** Backend per `RERANKER_TYPE` (Env) wählbar — `llm` (Default), `azure_semantic`, `bge`.
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

### 5.3 socratic (`src/llm/graphs/socratic.py`)

Drei Phasen: `contract` → `diagnose` → `core`

- Verfolgt `attempt_count` und `number_given_hints`
- Führt Studierende zur Antwort, statt sie direkt zu liefern
- Abbruch via Schlüsselwörter: "exit", "quit", "stop", "beende den lernmodus"

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
| `type` | Filterable | Inhaltstyp (Resource, Module, …) |
| `course_id` | Filterable (Int64) | Moodle-Kurs |
| `module_id` | Filterable (Int64) | Moodle-Modul |
| `url` | Filterable | Link zur Originalquelle |
| `is_important` | Filterable (bool) | Priorisierung |
| `source_doc_key` | Filterable | Stabiler Dokument-Identifier für Change Detection |
| `content_hash` | String | Content-Fingerprint für Change Detection |
| `dense` | Vector (Dim. nach Embedding-Modell, aktuell 3072) | HNSW, Cosine Distance |
| `metadata_json` | String | Vollständige Metadaten |

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
| `POST` | `/api/feedback` | Nutzerfeedback → Langfuse |
| `GET` | `/health` | Health-Check → `"OK"` |

### ChatRequest

```python
{
    user_query: SerializableChatMessage,
    thread_id: Optional[str],   # Konversations-ID (Persistenz)
    course_id: Optional[int],   # Moodle-Kursfilter
    module_id: Optional[int],   # Moodle-Modulfilter
    model: Models               # LLM-Auswahl
}
```

### Verfügbare Modelle

| Enum-Wert | Beschreibung |
|-----------|--------------|
| `AZURE_FALLBACK` | GPT-4/3.5 via Azure (Standard, zuverlässigste) |
| `MINI` | Hilfstasks: Sprache, Reranking (kostenoptimiert) |
| `GEMMA4_31B` | Open Model via GWDG-API (Fallback) |
| `LLAMA3` | Legacy-Alias → mappt auf Gemma4 |

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

**CitationStreamFilter:** Blendet `[docN]`-Marker während des Streamings aus; am Ende werden sie vom CitationParser in Markdown-Links umgewandelt.

---

## 10. Konversationspersistenz

**Mechanismus:** LangGraph Checkpointer

| Umgebung | Checkpointer |
|----------|-------------|
| Dev | `MemorySaver` (in-memory) |
| Produktion | PostgreSQL *(geplant, noch nicht umgesetzt)* |

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

**History-Limit:** Letzte 6 Nachrichten (Context-Window-Management)

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

**Framework:** Streamlit (`src/frontend/frontend.py`)

**Features:**
- Kurs-/Modul-Baum-Browser (linke Sidebar)
- Chat-Interface mit Streaming-Antworten
- Thinking-Indikator-Animation
- Feedback (Daumen hoch/runter + Kommentar)
- Session-Management via Thread-IDs

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

# Datenquellen
DATA_SOURCE_MOODLE_URL / TOKEN
DRUPAL_URL / CLIENT_ID / SECRET / USERNAME / PASSWORD
VIMEO_PAT

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
| `src/llm/graphs/socratic.py` | Lernmodus-Subgraph |
| `src/llm/objects/retriever.py` | Retrieval-Logik |
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
| 5 | PostgreSQL-Checkpointer für Produktion einsetzen (aktuell: MemorySaver) | Hoch |
| 6 | Managed Identity für alle Azure-Ressourcen | Mittel |

---

*Zuletzt aktualisiert: Juni 2026*
