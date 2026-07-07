# Chat-Persistenz — Flow & Konzept

## Warum entsteht bei einem Reload überhaupt eine neue Session?

Das Backend hält Konversationen in einem **In-Memory-Store** (`BoundedMemorySaver`). Es weiß dabei nicht, welche Browser-Session zu welchem Eintrag gehört — es kennt nur die `thread_id`, die das Frontend bei jeder Anfrage mitschickt.

Lädt die Seite neu, verliert das Frontend diese `thread_id`:
- **Streamlit**: `st.session_state` wird beim Browser-Refresh zurückgesetzt.
- **Moodle-Plugin**: JavaScript-Variablen verschwinden beim Reload, sofern sie nicht in `localStorage` gesichert werden.

Das Backend hat die Konversation noch im Speicher — es gibt nur niemanden mehr, der mit der richtigen `thread_id` anklopft. Der nächste Request ohne `thread_id` legt deshalb eine neue Session an.

**Lösung:** Das Frontend legt die `thread_id` persistent ab und schickt sie beim nächsten Seitenaufruf mit.

---

## Komponenten im Überblick

```
Browser / Frontend              Backend API               LangGraph / BoundedMemorySaver
─────────────────────           ─────────────────         ──────────────────────────────
localStorage / session_state ─► POST /api/chat/stream ──► KICampusAssistant
                                GET /api/chat/history      _get_or_create_state()
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
                            → BoundedMemorySaver: kein Eintrag für diese ID
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
                            → BoundedMemorySaver.get_state("abc-123")
                            → chat_history = [vorige Nachrichten, max. 6]
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
                            → BoundedMemorySaver.get_state("abc-123")
                            → gibt chat_history zurück
                            ← { thread_id: "abc-123", messages: [...] }

Frontend zeigt History an, setzt thread_id
```

---

## Flow: TTL-Ablauf (30 Minuten Inaktivität)

Die TTL-Logik liegt vollständig im **Frontend** — das Backend hat keine Meinung darüber, wann eine Session "abläuft".

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
Backend startet neu:
  BoundedMemorySaver = {}   ← leer

Frontend schickt thread_id: "abc-123"
  → kein Eintrag gefunden
  → neue leere Session für "abc-123"
```

Alle Konversationen gehen beim Neustart verloren — das ist bewusst so, da keine Datenbankpersistenz gewünscht ist.

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
  "module_id": null,
  "model": "Gemma4"
}
```

| Feld | Typ | Pflicht | Beschreibung |
|------|-----|---------|--------------|
| `user_query.role` | `string` | ja | Immer `"user"` |
| `user_query.content` | `string` | ja | Die Frage des Nutzers |
| `thread_id` | `string \| null` | nein | Bestehende Session-ID. `null` → neue Session |
| `course_id` | `integer \| null` | nein | Kurs-ID für gefilterte Suche (z.B. `79`) |
| `module_id` | `integer \| null` | nein | Modul-ID; erfordert `course_id` |
| `model` | `string` | nein | Zu nutzendes LLM. Default: `"Gemma4"`. Weitere Werte: `"Azure-Fallback"`, `"Llama3"` |

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

Gibt `messages: []` zurück wenn die Session unbekannt oder der Server neu gestartet wurde — kein Fehler, einfach leerer Chat.

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

## Speicherbegrenzung

`BoundedMemorySaver` ersetzt den Standard-`MemorySaver` von LangGraph.

- Hält intern ein `OrderedDict` mit der Erstellungsreihenfolge aller `thread_ids`
- Sobald `max_threads` (Standard: 500) überschritten wird → ältester Thread wird aus dem internen LangGraph-Speicher entfernt
- FIFO-Verdrängung (kein LRU — Overhead pro Request nicht gerechtfertigt)
- Eingebaut in `src/llm/assistant.py`, getestet in `src/tests/llms/test_bounded_memory_saver.py`

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
