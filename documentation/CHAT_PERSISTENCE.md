# Chat-Persistenz — Flow & Konzept

## Warum entsteht bei einem Reload überhaupt eine neue Session?

Das Backend hält Konversationen in einem **Redis-Sitzungsspeicher** (LangGraph-Checkpointer, geteilt über alle Backend-Worker; lokal ohne `REDIS_URL`: in-memory `BoundedMemorySaver`). Es weiß dabei nicht, welche Browser-Session zu welchem Eintrag gehört — es kennt nur die `thread_id`, die das Frontend bei jeder Anfrage mitschickt.

Lädt die Seite neu, verliert das Frontend diese `thread_id`:
- **Streamlit**: `st.session_state` wird beim Browser-Refresh zurückgesetzt.
- **Moodle-Plugin**: JavaScript-Variablen verschwinden beim Reload, sofern sie nicht in `localStorage` gesichert werden.

Das Backend hat die Konversation noch im Speicher — es gibt nur niemanden mehr, der mit der richtigen `thread_id` anklopft. Der nächste Request ohne `thread_id` legt deshalb eine neue Session an.

**Lösung:** Das Frontend legt die `thread_id` persistent ab und schickt sie beim nächsten Seitenaufruf mit.

---

## Komponenten im Überblick

```
Browser / Frontend              Backend API               LangGraph-Checkpointer
─────────────────────           ─────────────────         ──────────────────────────────
localStorage / session_state ─► POST /api/chat/stream ──► KICampusAssistant
                                GET /api/chat/history      _get_or_create_state()
                                                           Redis (TTL 24 h, alle Worker)
                                                           lokal ohne REDIS_URL:
                                                           BoundedMemorySaver (max 500)
```

---

## Flow: Neue Konversation

```
Frontend                         Backend
───────                          ───────
POST /api/chat/stream
  thread_id: null
                            → thread_id = uuid4()   (neu generiert)
                            → Checkpointer (Redis): kein Eintrag für diese ID
                            → initial_state: chat_history = []
                            → Graph ausführen
                            → State unter thread_id speichern
                            ← meta-Event: thread_id = "abc-123"

Frontend speichert thread_id
```

---

## Flow: Folgenachricht (gleiche Session)

```
Frontend                         Backend
───────                          ───────
POST /api/chat/stream
  thread_id: "abc-123"
                            → Checkpointer (Redis): get_state("abc-123") — egal, welcher Worker antwortet
                            → chat_history = [vorige Nachrichten, max. CHAT_HISTORY_LIMIT (Default 6);
                                              in aktiven Socratic-v2-Sessions: CHAT_HISTORY_LIMIT_SOCRATIC_V2 (Default 40)]
                            → Graph mit History ausführen
                            → chat_history += [user, assistant]
                            ← Antwort + thread_id: "abc-123"
```

---

## Flow: Session nach Page-Reload wiederherstellen

```
Frontend                         Backend
───────                          ───────
GET /api/chat/history/abc-123
                            → Checkpointer (Redis): get_state("abc-123")
                            → gibt chat_history zurück
                            ← { thread_id: "abc-123", messages: [...] }

Frontend zeigt History an, setzt thread_id
```

---

## Flow: TTL-Ablauf (30 Minuten Inaktivität)

Die für die Nutzenden sichtbare TTL-Logik liegt im **Frontend** — es entscheidet, wann eine Session als "abgelaufen" gilt und verwirft die `thread_id`. Zusätzlich verfallen Sessions serverseitig per Redis-TTL nach 24 h ohne Zugriff (`CHAT_TTL_MINUTES`, bei jedem Zugriff erneuert) — das ist Speicherhygiene, kein UX-Mechanismus; bei den üblichen 30-Minuten-Client-TTLs greift sie praktisch nie zuerst.

```
Frontend                         Backend
───────                          ───────
User war 30+ min inaktiv
Frontend löscht gespeicherte thread_id

POST /api/chat/stream
  thread_id: null              ← keine ID mehr
                            → thread_id = uuid4()   (neue Session)
                            → leere chat_history
                            ← neue thread_id
```

---

## Flow: Server-Neustart

```
Site/Container-Gruppe wird neu gestartet (Deployment, az webapp restart):
  Redis-Sidecar startet leer   ← Sessions weg

Frontend schickt thread_id: "abc-123"
  → kein Eintrag gefunden
  → neue leere Session für "abc-123"
```

Redis läuft seit der IaC-Umstellung (2026-07-20) als **eigener Sidecar-Container** neben der API, nicht mehr im API-Image selbst — das entkoppelt Redis-Version/-Konfiguration vom API-Image. Ob ein reguläres API-Deploy (`task deploy-api`, ruft `az webapp restart` auf) nur den API-Container oder die gesamte Site inkl. Sidecar neu startet, hängt von Azure App Service ab und ist noch nicht verifiziert — im Zweifel gilt weiterhin: jeder Site-Neustart leert Redis, das ist akzeptiert, da keine dauerhafte Speicherung gewünscht ist. Ein Recycling einzelner Gunicorn-Worker im laufenden Betrieb (`max_requests`) betraf die Sessions schon vorher nicht, da Redis als eigener Prozess unabhängig davon lief.

**Wichtig:** Der Sidecar löst nicht das Scale-out-Problem — bei mehreren App-Service-Instanzen bekommt jede Instanz ihre eigene Kopie des Sidecars (kein geteilter Redis zwischen Instanzen). Für Betrieb mit >1 Instanz bleibt Azure Managed Redis (nur `REDIS_URL`-App-Setting) die Lösung.

---

## API-Schnittstelle für das Frontend-Team

### POST /api/chat/stream

**Request-Body (JSON):**

```json
{
  "user_query": {
    "role": "user",
    "content": "Erkläre mir Gradient Descent"
  },
  "thread_id": "550e8400-e29b-41d4-a716-446655440000",
  "course_id": 79,
  "module_id": [1234, 1235],
  "model": "Gemma4",
  "start_socratic": false,
  "start_socratic_v2": false
}
```

| Feld | Typ | Pflicht | Beschreibung |
|------|-----|---------|--------------|
| `user_query.role` | `string` | ja | Immer `"user"` |
| `user_query.content` | `string` | ja | Die Frage des Nutzers |
| `thread_id` | `string \| null` | nein | Bestehende Session-ID. `null` → neue Session |
| `course_id` | `integer \| null` | nein | Kurs-ID für gefilterte Suche (z.B. `79`) |
| `module_id` | `integer \| integer[] \| null` | nein | Einzelne Modul-ID **oder Liste mehrerer Module desselben Kurses** (gleichrangiger OR-Filter, z. B. „alle bisher bearbeiteten Module"); erfordert `course_id`. Leere Liste = `null` |
| `model` | `string` | nein | Zu nutzendes LLM. Default: `"Gemma4"`. Weitere Werte: `"Azure-Fallback"`, `"Llama3"` (Legacy-Alias → Gemma4) |
| `start_socratic` | `boolean` | nein | Lernmodus v1 für diese Session starten (nur wirksam, wenn Backend-Flag `ENABLE_SOCRATIC` gesetzt ist) |
| `start_socratic_v2` | `boolean` | nein | Lernmodus v2 starten (Backend-Flag `ENABLE_SOCRATIC_V2`; braucht Kurs-/Modul-Scope). Alternativ per Trigger-Phrase im Chat |

**Response: NDJSON-Stream** (`Content-Type: application/x-ndjson`)

Jede Zeile ist ein JSON-Objekt, abgeschlossen mit `\n`. Reihenfolge:

```
{"type": "meta",  "thread_id": "abc-123", "response_id": "xyz-456"}
{"type": "token", "token": "Gradient"}
{"type": "token", "token": " Descent"}
...
{"type": "final", "message": "Gradient Descent ist ...", "thread_id": "abc-123", "response_id": "xyz-456"}
```

Bei Fehler statt `final`:
```
{"type": "error", "message": "...", "response_id": "xyz-456", "thread_id": "abc-123"}
```

| Event | Felder | Wann |
|-------|--------|------|
| `meta` | `thread_id`, `response_id` | Erstes Event — immer, vor den Tokens |
| `token` | `token` | Einmal pro Token während der Generierung |
| `final` | `message`, `thread_id`, `response_id` | Letztes Event bei Erfolg |
| `error` | `message`, `response_id`, `thread_id` | Letztes Event bei Fehler |

**Wichtig:** `thread_id` kommt bereits im `meta`-Event — das Frontend kann sie sofort in `localStorage` speichern, ohne auf `final` zu warten.

---

### GET /api/chat/history/{thread_id}

Lädt den gespeicherten Gesprächsverlauf, z.B. nach Page-Reload.

**Response (JSON):**

```json
{
  "thread_id": "abc-123",
  "messages": [
    {"role": "user",      "content": "Erkläre mir Gradient Descent"},
    {"role": "assistant", "content": "Gradient Descent ist ..."}
  ]
}
```

Gibt `messages: []` zurück wenn die Session unbekannt ist, serverseitig abgelaufen (Redis-TTL) oder der Container neu gestartet wurde — kein Fehler, einfach leerer Chat.

---

## Was muss wo aufgerufen werden?

### Moodle-Frontend (noch zu implementieren)

`localStorage`-Key-Strategie: **pro Kurs eine eigene ID** → `kic_thread_id_course{course_id}`. Damit haben verschiedene Kurse getrennte Konversationen.

| Zeitpunkt | Aktion |
|-----------|--------|
| Seite lädt | `localStorage.getItem("kic_thread_id_course79")` lesen |
| `thread_id` vorhanden | `GET /api/chat/history/{thread_id}` → History rendern |
| `thread_id` nicht vorhanden | leerer Chat, neue Session beim ersten POST |
| Nach jeder Antwort | `localStorage.setItem("kic_thread_id_course79", thread_id)` |
| Nach 30 min Inaktivität (clientseitig) | `localStorage.removeItem("kic_thread_id_course79")` → nächster POST ohne ID |
| "Neues Gespräch"-Button | `localStorage.removeItem("kic_thread_id_course79")` |

### Streamlit (Simulation, bereits implementiert)

Simuliert das Moodle-Verhalten für lokale Tests. Analog zu `localStorage` wird die `thread_id` in der **URL** (`?thread_id=...`) gespeichert — überlebt so den Browser-Refresh.

| Zeitpunkt | Aktion |
|-----------|--------|
| App-Start (frischer Load) | `st.experimental_get_query_params()` prüfen → falls `thread_id` vorhanden: `GET /api/chat/history/{id}` → Session automatisch wiederherstellen |
| Nach Restore | `st.rerun()` damit Sidebar ("↩ Session wiederhergestellt") sofort sichtbar |
| Nach jeder Antwort | `st.experimental_set_query_params(thread_id=...)` → URL aktualisieren, dann `st.rerun()` |
| Vor jedem POST | TTL-Check: `time.time() - last_activity > 1800` → `reset_history()` |
| "Session-ID laden" (Sidebar) | Manuell: `GET /api/chat/history/{id}` → messages + thread_id setzen |
| "Neue Session" / Reset | `reset_history()` + `st.experimental_set_query_params()` → URL leeren |

**Testflow:**
1. Nachricht senden → URL wechselt zu `?thread_id=abc-123`
2. F5 → Session automatisch wiederhergestellt, Sidebar zeigt "↩ Session wiederhergestellt (Reload)"

---

## Speicher & Verdrängung

Die Auswahl trifft `build_checkpointer()` in `src/llm/assistant.py` anhand von `REDIS_URL`:

**Deployment (`REDIS_URL` gesetzt — Default im API-Image):** `RedisSaver` aus `langgraph-checkpoint-redis`.

- Ein gemeinsamer Sessionstore für **alle Gunicorn-Worker** — welcher Worker eine Anfrage bedient, spielt keine Rolle.
- Redis läuft als **Sidecar-Container neben der API** (IaC, nicht Teil dieses Repos), erreichbar über `REDIS_URL=redis://localhost:6379` (Azure-App-Service-Sidecars teilen den Network-Namespace mit dem Hauptcontainer). `src/api/entrypoint.sh` wartet beim Start kurz, falls der Sidecar noch nicht bereit ist. Wechsel auf **Azure Managed Redis** = nur `REDIS_URL` überschreiben (`rediss://:<key>@<name>.<region>.redis.azure.net:10000`), kein Code-Umbau.
- Verdrängung über **TTL** statt FIFO: `CHAT_TTL_MINUTES` (Default 1440 = 24 h), bei jedem Zugriff erneuert. `maxmemory`/Eviction-Policy des Sidecars sind IaC-Konfiguration, nicht Teil dieses Repos.
- Kein Snapshotting/AOF gewünscht — Sessions sind bewusst flüchtig (Konfiguration ebenfalls im Sidecar/IaC).

**Lokal / Tests (`REDIS_URL` nicht gesetzt):** `BoundedMemorySaver` (ersetzt den Standard-`MemorySaver` von LangGraph).

- Hält intern ein `OrderedDict` mit der Erstellungsreihenfolge aller `thread_ids`; sobald `MAX_CHAT_THREADS` (Default 500) überschritten wird, fliegt der älteste Thread raus (FIFO, lock-geschützt).

**Unabhängig vom Backend** gelten `CHAT_HISTORY_LIMIT` (Default 6) und `CHAT_HISTORY_LIMIT_SOCRATIC_V2` (Default 40 — erweitertes Historien-Fenster für aktive Lernmodus-v2-Sessions); alle Settings in `src/env.py`.

Getestet in `src/tests/llms/test_checkpointer_factory.py` (Auswahl-Logik) und `src/tests/llms/test_bounded_memory_saver.py` (FIFO-Verdrängung).

---

## Was wurde zur bestehenden Implementierung ergänzt?

### Schon vorhanden (vor diesen Änderungen)

- `MemorySaver` als LangGraph-Checkpointer in `KICampusAssistant.__init__`
- `thread_id` in `ChatRequest` / `ChatResponse` / Streaming-Events
- `_get_or_create_state()`: lädt History aus Checkpoint oder erstellt neue Session
- Streamlit-Frontend: sendet und empfängt `thread_id`, speichert in `st.session_state`

### Neu ergänzt

| Was | Wo | Warum |
|-----|----|-------|
| `BoundedMemorySaver` (max 500 Threads, FIFO) | `assistant.py` | Verhindert unbegrenztes Speicherwachstum |
| `get_chat_history(thread_id)` | `assistant.py` | Liest History ohne neuen Chat-Turn zu starten |
| `ChatHistoryResponse` Model | `rest.py` | Typisiertes Response-Objekt |
| `GET /api/chat/history/{thread_id}` | `rest.py` | Endpunkt für Page-Reload-Restore |
| `SESSION_TTL` + TTL-Check vor POST | `frontend.py` | Simuliert 30-min-Inaktivitäts-Reset (Moodle: `localStorage` löschen) |
| `_load_session_from_backend()` | `frontend.py` | Ruft GET-Endpunkt auf, befüllt UI mit gespeicherter History |
| Moodle-Simulation Sidebar | `frontend.py` | Session-ID anzeigen + Laden/Reset für lokale Tests |
| `st.rerun()` nach Streaming | `frontend.py` | Sidebar zeigt thread_id sofort nach erster Antwort |
| Tests für `BoundedMemorySaver` | `src/tests/llms/test_bounded_memory_saver.py` | 5 Unit- + 4 Integrationstests |
| `build_checkpointer()`: Redis-Checkpointer mit TTL, in-memory Fallback | `assistant.py`, `env.py` | Ein gemeinsamer Sessionstore für alle Gunicorn-Worker |
| Tests für die Checkpointer-Auswahl | `src/tests/llms/test_checkpointer_factory.py` | 3 Unit-Tests inkl. `rediss://`-Pfad |
| Redis als Sidecar-Container (IaC, außerhalb dieses Repos); `entrypoint.sh` wartet auf Erreichbarkeit statt Redis selbst zu starten | `src/api/Dockerfile`, `src/api/entrypoint.sh` | Redis-Lifecycle von API-Image entkoppelt; Managed-Redis-Wechsel weiterhin nur per `REDIS_URL` |
