# KI-Campus Assistant — API-Schnittstelle für das Frontend

Diese Dokumentation beschreibt vollständig, wie ein Frontend (Moodle-Plugin, Drupal-Widget, eigene Web-App) den KI-Campus-Chat-Assistenten anbindet. Stand: 14.08.2026.

---

## 1. Überblick

| | |
|---|---|
| **Base URL** | wird separat mitgeteilt; in dieser Doku als `<BASE_URL>` geschrieben |
| **Protokoll** | HTTPS, JSON (`Content-Type: application/json`) |
| **Authentifizierung** | Header `Api-Key: <euer Schlüssel>` bei allen `/api/*`-Endpunkten |
| **Interaktive Referenz** | `GET /docs` (Swagger UI), `GET /openapi.json` |

### Authentifizierung

Alle Endpunkte unter `/api/` erwarten den Header:

```
Api-Key: <euer Schlüssel>
```

Der Schlüssel gehört **nicht in den Browser-Code** — Requests idealerweise serverseitig absetzen (Moodle-PHP-Backend, Drupal-Modul, BFF/Proxy). Ein reiner Browser-Client, der den Key mitschickt, gibt ihn an jeden Nutzenden weiter.

- Header fehlt → **403**
- Header falsch → **401** (`{"detail": "Invalid API Key"}`)

### CORS

Direkte Browser-Aufrufe sind nur von diesen Origins erlaubt:

```
https://ki-campus.org
https://moodle.ki-campus.org
https://ki-campus.loom.de
https://ki-campus.staging.piipe.de
```

Weitere Origins müssen im Backend freigeschaltet werden — bitte vorab melden.

CORS ist ein reiner Browser-Mechanismus: Er greift nur, wenn **JavaScript im Browser** die API direkt aufruft. Serverseitige Aufrufe — Moodle-PHP, ein eigener Proxy/BFF, Skripte, `curl` — sind davon nicht betroffen und funktionieren unabhängig von dieser Liste.

### Timeouts

Der Server bricht Requests nach **230 Sekunden** ab (Azure-App-Service-Limit).

---

## 2. Endpunktübersicht

| Methode | Pfad | Auth | Zweck |
|---------|------|------|-------|
| `POST` | `/api/chat/stream` | ja | **Empfohlen.** Chat-Antwort als Token-Stream (NDJSON) |
| `POST` | `/api/chat` | ja | Chat-Antwort in einem Stück (ohne Streaming) |
| `GET` | `/api/chat/history/{thread_id}` | ja | Gespeicherten Gesprächsverlauf laden (z. B. nach Page-Reload) |
| `POST` | `/api/feedback` | ja | Daumen hoch/runter + Freitext zu einer Antwort |
| `GET` | `/health` | nein | Healthcheck (Monitoring/Loadbalancer) |

---

## 3. POST /api/chat/stream — Chat mit Streaming (empfohlen)

Liefert die Antwort tokenweise, damit im UI sofort Text erscheint.

### Request

```http
POST /api/chat/stream
Api-Key: <euer Schlüssel>
Content-Type: application/json
```

```json
{
  "user_query": {
    "role": "user",
    "content": "Erkläre mir Gradient Descent"
  },
  "thread_id": "550e8400-e29b-41d4-a716-446655440000",
  "course_id": 79,
  "module_id": [1234, 1235],
  "start_socratic": false,
  "start_socratic_v2": false
}
```

| Feld | Typ | Pflicht | Default | Beschreibung |
|------|-----|---------|---------|--------------|
| `user_query` | `object` | **ja**¹ | — | Die neue Nutzernachricht. Genau **eine** Nachricht, kein Array — der bisherige Verlauf liegt serverseitig unter der `thread_id` |
| `user_query.role` | `string` | ja | — | Immer `"user"` |
| `user_query.content` | `string` | ja | — | Der Text der Nutzerfrage |
| `thread_id` | `string \| null` | nein | `null` | Bestehende Session-ID. `null` → Backend erzeugt eine neue (UUID4) und liefert sie im `meta`-Event zurück |
| `course_id` | `integer \| null` | nein | `null` | Kurs-ID (Moodle). Schränkt die Suche auf diesen Kurs ein. Ohne Angabe: Suche über die allgemeinen KI-Campus-Inhalte (Drupal) |
| `module_id` | `integer \| integer[] \| null` | nein | `null` | Modul-/Themen-ID oder **Liste** mehrerer Module desselben Kurses (ODER-Filter, z. B. „alle bisher bearbeiteten Module"). Leere Liste `[]` = kein Filter. **Erfordert `course_id`** |
| `model` | `string` | nein | `"Gemma4"` | **Bitte weglassen.** Die Modellwahl trifft das Backend, s. [Abschnitt 8](#8-modellwahl-feld-model) |
| `start_socratic` | `boolean` | nein | `false` | Startet den Lernmodus v1 mit dieser Nachricht, s. [Abschnitt 10](#10-lernmodus-sokratischer-tutor) |
| `start_socratic_v2` | `boolean` | nein | `false` | Startet den Sokratischen Tutor (v2), z. B. über den Modus-Schalter. `user_query` kann dabei entfallen, s. [Abschnitt 10](#10-lernmodus-sokratischer-tutor) |
| `stop_socratic_v2` | `boolean` | nein | `false` | Beendet den Sokratischen Tutor (v2) ohne Schlüsselwort-Nachricht, z. B. über den Modus-Schalter. `user_query` kann dabei entfallen. Nicht zusammen mit `start_socratic_v2` setzen (→ 400) |
| `ui_language` | `string \| null` | nein | `null` | Oberflächen-/Gerätesprache als Code oder Name (`"en"`, `"de-DE"`, `"English"`). Der Sokratische Tutor nutzt sie als Startsprache, wenn der Chat noch keine eigene Nachricht enthält. Die Sprache der Nachrichten selbst hat immer Vorrang. Kann bei jedem Request mitgeschickt werden |

¹ Ausnahme: Bei reinen Schalter-Requests (`start_socratic_v2` oder `stop_socratic_v2`) darf `user_query` fehlen. Wird trotzdem ein Text mitgeschickt, gilt er als Signal und landet **nicht** im Gesprächsverlauf.

**Scope-Regeln in Kurzform:**

| `course_id` | `module_id` | Ergebnis |
|---|---|---|
| `null` | `null` | Globale Suche über KI-Campus-Website-Inhalte |
| `79` | `null` | Suche im gesamten Kurs 79 |
| `79` | `1234` bzw. `[1234, 1235]` | Suche in diesen Modulen des Kurses 79 |
| `null` | gesetzt | **400** — `module_id` ohne `course_id` ist ungültig |

### Response: NDJSON-Stream

`Content-Type: application/x-ndjson`, HTTP **200**. Jede Zeile ist ein eigenständiges JSON-Objekt, abgeschlossen mit `\n`:

```
{"type":"meta","thread_id":"abc-123","response_id":"xyz-456"}
{"type":"token","token":"Gradient"}
{"type":"token","token":" Descent"}
{"type":"token","token":" ist ein Optimierungsverfahren"}
{"type":"final","message":"Gradient Descent ist ein Optimierungsverfahren ... [<a href=\"https://moodle.ki-campus.org/mod/page/view.php?id=1234\">1</a>]\n\n**Zum Weiterlesen:**\n1. <a href=\"https://moodle.ki-campus.org/mod/page/view.php?id=1234\">Optimierung</a>","thread_id":"abc-123","response_id":"xyz-456","learning_mode_active":false}
```

| Event | Felder | Wann |
|-------|--------|------|
| `meta` | `thread_id`, `response_id` | **Immer als erstes**, noch bevor das LLM antwortet |
| `token` | `token` | Beliebig oft während der Generierung |
| `final` | `message`, `thread_id`, `response_id`, `learning_mode_active` | Letztes Event im Erfolgsfall. `learning_mode_active`: ob nach dieser Antwort eine Tutor-Session läuft, s. [Abschnitt 10](#10-lernmodus-sokratischer-tutor) |
| `error` | `message`, `thread_id`, `response_id` | Letztes Event im Fehlerfall (statt `final`) |

Fehlerfall:

```json
{"type":"error","message":"Upstream LLM timeout","thread_id":"abc-123","response_id":"xyz-456"}
```

### Vier Regeln, die das Frontend beachten muss

1. **`final.message` ist die verbindliche Antwort — nicht die Summe der Tokens.**
   Beim Rendern die gestreamten Tokens durch `final.message` **ersetzen**, nicht anhängen. In bestimmten Fällen (z. B. „keine passenden Inhalte gefunden") kommen bewusst **gar keine** `token`-Events, und `final.message` enthält stattdessen einen freundlichen Hinweistext.
2. **`thread_id` sofort aus dem `meta`-Event speichern**, nicht erst bei `final`. Bricht die Verbindung mitten im Stream ab, ist die Session so trotzdem fortsetzbar.
3. **Antworten enthalten HTML** (Quellenangaben als `<a href="…">`), s. [Abschnitt 9](#9-antwortformat-quellen-fallbacks-sprache). Der Text muss als HTML gerendert werden (sanitized, nur `<a>` nötig) — sonst sehen Nutzende rohes Markup.
4. **Fehler können vor *und* im Stream auftreten.** Validierungs-/Auth-Fehler kommen als normaler HTTP-Fehlerstatus **vor** dem Stream (400/401/403/422). Ist der Status erst einmal 200, kommt ein späterer Fehler nur noch als `error`-Event.

### Beispiel: curl

```bash
curl -N -X POST "$BASE_URL/api/chat/stream" \
  -H "Api-Key: $KIC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "user_query": {"role": "user", "content": "Was lerne ich in diesem Modul?"},
        "course_id": 79,
        "module_id": 1234,
        "thread_id": null
      }'
```

Beim Lesen des Streams ist zu beachten, dass eine Zeile erst dann ein vollständiges Event ist, wenn sie mit `\n` abgeschlossen wurde — ein Chunk aus dem Netzwerk kann mitten in einer Zeile enden.

---

## 4. POST /api/chat — Chat ohne Streaming

Identischer Request-Body wie `/api/chat/stream`. Liefert die vollständige Antwort in einem JSON-Objekt — einfacher zu integrieren, aber der Nutzende sieht bis zur fertigen Antwort nichts.

### Response (200)

```json
{
  "message": "Gradient Descent ist ein Optimierungsverfahren ... [<a href=\"https://moodle.ki-campus.org/mod/page/view.php?id=1234\">1</a>]\n\n**Zum Weiterlesen:**\n1. <a href=\"https://moodle.ki-campus.org/mod/page/view.php?id=1234\">Optimierung</a>",
  "response_id": "xyz-456",
  "thread_id": "abc-123",
  "learning_mode_active": false
}
```

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `message` | `string` | Die Antwort des Assistenten (enthält HTML-Quellenlinks) |
| `response_id` | `string` | ID der Antwort — wird für `/api/feedback` benötigt. Fällt das Tracing aus, steht hier `"TRACING_UNAVAILABLE"`; in dem Fall kein Feedback senden |
| `thread_id` | `string` | Session-ID für Folgeanfragen — speichern und beim nächsten Request mitschicken |
| `learning_mode_active` | `boolean` | Ob nach dieser Antwort eine Tutor-Session läuft — danach den Modus-Schalter setzen, s. [Abschnitt 10](#10-lernmodus-sokratischer-tutor) |

---

## 5. GET /api/chat/history/{thread_id}

Lädt den serverseitig gespeicherten Gesprächsverlauf, z. B. nach einem Page-Reload.

```http
GET /api/chat/history/abc-123
Api-Key: <euer Schlüssel>
```

### Response (200)

```json
{
  "thread_id": "abc-123",
  "messages": [
    {"role": "user",      "content": "Erkläre mir Gradient Descent", "fallback_type": null},
    {"role": "assistant", "content": "Gradient Descent ist ...",      "fallback_type": null}
  ],
  "learning_mode_active": false
}
```

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `thread_id` | `string` | Die angefragte ID (wird unverändert zurückgegeben) |
| `messages` | `array` | Nachrichten in chronologischer Reihenfolge |
| `messages[].role` | `string` | `"user"` oder `"assistant"` |
| `messages[].content` | `string` | Nachrichtentext (Assistant-Nachrichten enthalten HTML-Quellenlinks) |
| `messages[].fallback_type` | `string \| null` | Nur gesetzt, wenn die Antwort ein systemseitiger Fallback war (z. B. „keine Inhalte gefunden"). Für die Anzeige irrelevant, kann ignoriert werden |
| `learning_mode_active` | `boolean` | Ob in dieser Session gerade der Sokratische Tutor läuft — nach einem Page-Reload den Modus-Schalter danach setzen |

**Wichtig:** Unbekannte, abgelaufene oder nach einem Server-Neustart verlorene Sessions liefern **200 mit `messages: []`** — kein 404. Das Frontend zeigt dann einfach einen leeren Chat.

**Hinweis zur Länge:** Der Endpunkt liefert **nicht den kompletten Gesprächsverlauf**, sondern nur das serverseitig vorgehaltene Fenster: die letzten **8 Nachrichten** (4 Frage-Antwort-Paare), in einer aktiven Lernmodus-v2-Session bis zu 42. Ältere Nachrichten werden verworfen, sobald das Fenster überläuft — sie sind danach auch über diesen Endpunkt nicht mehr abrufbar.

Wenn im UI der vollständige Verlauf einer laufenden Sitzung stehen bleiben soll, muss das Frontend ihn selbst mitschreiben (z. B. im `sessionStorage`) und diesen Endpunkt nur für die Wiederherstellung nach einem Reload nutzen.

---

## 6. POST /api/feedback

Meldet Daumen hoch/runter (plus optionalen Freitext) zu einer konkreten Antwort zurück.

```http
POST /api/feedback
Api-Key: <euer Schlüssel>
Content-Type: application/json
```

```json
{
  "response_id": "xyz-456",
  "score": 1,
  "feedback": "Hat mir sehr geholfen"
}
```

| Feld | Typ | Pflicht | Beschreibung |
|------|-----|---------|--------------|
| `response_id` | `string` | **ja** | Die `response_id` aus `meta`/`final`/`ChatResponse` |
| `score` | `integer` | **ja** | `1` = Daumen hoch, `0` = Daumen runter. Andere Werte → **400** |
| `feedback` | `string \| null` | nein | Freitext-Kommentar |

### Response

**200** mit Body `null`. Der Endpunkt liefert keine Nutzdaten zurück — nur der Statuscode zählt.

Fehlgeschlagene Feedback-Übertragung darf den Chat nicht blockieren: bei einem 5xx den Fehler im Frontend still verwerfen bzw. loggen, der Chat bleibt nutzbar.

---

## 7. GET /health

```http
GET /health
```

Antwort **200** mit Body `"OK"`. Ohne Authentifizierung, gedacht für Monitoring und Loadbalancer.

---

## 8. Modellwahl (Feld `model`)

**Das Feld bitte weglassen.** Welches LLM antwortet, ist Betriebssache des Backends: Standard ist das GWDG-gehostete Gemma 4 (31B), und bei Störung oder Timeout schaltet das Backend selbstständig auf ein Azure-Modell um. Das Frontend muss dafür nichts implementieren und merkt vom Wechsel nichts.

Wird das Feld weggelassen, greift der Default. Nur für Backend-seitige Tests und Modellvergleiche ist es überhaupt gedacht:

| Wert | Bedeutung |
|------|-----------|
| `"Gemma4"` | Default — GWDG-gehostetes Gemma 4 (31B) |
| `"Azure-Fallback"` | Azure-OpenAI-Modell; wird bei GWDG-Störung ohnehin automatisch verwendet |
| `"Llama3"` | Legacy-Alias, zeigt intern ebenfalls auf Gemma 4 |
| `"Azure-Mini"` | Internes Hilfsmodell, nicht für Chat-Anfragen |

Ein unbekannter Wert führt zu **422**.

---

## 9. Antwortformat: Quellen, Fallbacks, Sprache

### Quellenangaben sind HTML

Belegte Aussagen bekommen im Text eine kurze, klickbare Nummer. Die Quellen stehen gesammelt als nummerierte Liste unter der Antwort (Feedback S. 17):

```html
Gradient Descent ist ein Optimierungsverfahren …[<a href="https://moodle.ki-campus.org/mod/page/view.php?id=1234">1</a>] Die Lernrate bestimmt …[<a href="https://moodle.ki-campus.org/mod/hvp/view.php?id=987">2</a>]

**Zum Weiterlesen:**
1. <a href="https://moodle.ki-campus.org/mod/page/view.php?id=1234">Optimierungsverfahren</a>
2. <a href="https://moodle.ki-campus.org/mod/hvp/view.php?id=987">Einführung Deep Learning</a> (ab Minute 4:12)
```

- Die Nummern verlinken auf dieselbe Seite wie der Listeneintrag. Wird dieselbe Seite mehrfach zitiert, bekommt sie überall dieselbe Nummer und steht nur einmal in der Liste.
- Bei Video-Transkripten steht der Zeitstempel hinter dem Listeneintrag.
- Die Überschrift kommt in der Antwortsprache („Zum Weiterlesen:“, „Further reading:“).
- Antworten ohne Quellen (Smalltalk, Fallback-Texte, Sokratischer Tutor) haben keine Liste.

Konsequenzen für das Frontend:

- Der Antworttext muss **als HTML gerendert** werden. Bei Sanitizing bitte `<a href>` erlauben. Die Links zeigen auf die Quellseite im KI-Campus — je nach Inhalt auf die Website (`ki-campus.org`) oder auf die Moodle-Instanz.
- Der übrige Text ist **Markdown** — mindestens Absätze, Fett (`**…**`) und nummerierte Listen müssen dargestellt werden, sonst erscheint die Quellenliste als Fließtext mit Sternchen.
- Die Links entstehen bereits **während des Streamings** — sie kommen also schon in den `token`-Events als HTML-Fragmente an und können über mehrere Tokens verteilt sein. Die Quellenliste kommt als letzter Token-Block nach dem Antworttext. Deshalb: gepufferten Gesamttext rendern, nicht einzelne Tokens isoliert.

### Fallback-Antworten

Findet der Assistent keine passenden Inhalte, antwortet er mit einem festen, freundlichen Hinweistext statt mit einer geratenen Antwort. Für das Frontend sind das ganz normale Antworten — kein Fehlerstatus, keine Sonderbehandlung nötig. Zu beachten ist nur Regel 1 aus [Abschnitt 3](#vier-regeln-die-das-frontend-beachten-muss): in diesem Fall kommen keine `token`-Events, die Antwort steht ausschließlich in `final.message`.

Eine Sonderform: War eine modul-eingeschränkte Frage nicht beantwortbar, bietet der Assistent an, im ganzen Kurs zu suchen — „Dazu habe ich in diesem Modul leider nichts gefunden. Soll ich stattdessen im gesamten Kurs suchen? Antworte einfach mit ‚Ja'." Antwortet der Nutzende in der nächsten Nachricht zustimmend („Ja", „gerne", „ok", auch frei formuliert), wiederholt das Backend die Suche ohne Modul-Filter. **Voraussetzung:** Die Folgenachricht läuft mit derselben `thread_id` — nur dort liegt das Angebot; sonst wird „Ja" als normale Frage behandelt. `course_id` und `module_id` schickt das Frontend dabei unverändert weiter; für diesen einen Turn ignoriert das Backend den Modul-Filter und sucht kursweit.

### Sprache

Die Antwortsprache wird automatisch aus der Nutzerfrage erkannt (Deutsch, Englisch, Spanisch, Französisch, Italienisch, Türkisch u. a.), inklusive der Fallback-Texte. Das Frontend muss keine Sprache mitgeben.

Im Sokratischen Tutor gilt dasselbe für die ganze Session: Er startet in der Sprache des bisherigen Chats (nicht in der Sprache des Start-Signaltexts). Schreibt die lernende Person später in einer anderen Sprache, wechselt er mit. Kurze Antworten wie „b“ oder „ja“ ändern die Sprache nicht. Bittet die Person um eine Übersetzung („auf Englisch bitte“, „translate that“), übersetzt der Tutor seine letzte Nachricht 1:1 und macht in der neuen Sprache weiter. Bei einem ganz neuen Chat ohne eigene Nachricht hilft das optionale Feld `ui_language`, sonst startet er auf Deutsch. Quizfragen aus dem Kurs erscheinen in der Kurssprache.

---

## 10. Lernmodus (sokratischer Tutor)

Der Assistent kann statt direkter Antworten in einen geführten Lernmodus wechseln, der mit Fragen zum eigenen Denken anleitet. **Aktuell sind beide Varianten — v1 und v2 — parallel aktiv, damit sie im Test verglichen werden können.** Das Frontend braucht dafür zwei getrennte Auslöser (z. B. zwei Buttons).

**Start:** Im Request `start_socratic: true` (v1) bzw. `start_socratic_v2: true` (v2) setzen.

- **Empfohlen für den Modus-Schalter (v2):** einen eigenen Request nur mit dem Flag schicken, **ohne** `user_query` (Beispiel unten). Im Chat erscheint dann keine Nutzernachricht, nur die Begrüßung des Tutors — auch nicht nach einem Page-Reload über `/api/chat/history`. Ein trotzdem mitgeschickter Text wird ebenfalls nicht gespeichert.
- Das Flag nur bei **genau dieser einen** Nachricht senden. Danach hält das Backend den Modus über die `thread_id` fest; erneutes Setzen ist unnötig.
- Immer nur eines der beiden Flags setzen.
- Lernmodus v2 braucht einen Kurs-/Modul-Scope. Ohne `course_id` startet keine Session; der Assistent antwortet stattdessen mit der Bitte, zuerst einen Kurs oder ein Modul auszuwählen. Dasselbe gilt, wenn im gewählten Modul keine nutzbaren Inhalte liegen.
- Beide Modi hängen an Backend-Feature-Flags. Wird eine Variante später abgeschaltet, ignoriert das Backend das Flag **stillschweigend** und antwortet als normaler Chat — es gibt keinen Fehler und keinen Endpunkt, über den das Frontend den Flag-Status abfragen kann.

**Name:** In der Oberfläche heißt v2 **„Sokratischer Tutor“**. Alle Texte des Backends (Begrüßung, Hinweise, Beenden) verwenden diesen Namen.

**Beenden:** Der Nutzende schreibt eines dieser Schlüsselwörter (Groß-/Kleinschreibung egal, Satzzeichen am Ende werden ignoriert):

```
exit · quit · stop · stopp · beende den tutor · beende den sokratischen tutor · tutor beenden
end tutor · stop tutor · ich möchte aufhören · beende den lernmodus (alt, weiterhin gültig)
```

Der Assistent bestätigt dann mit „Du hast den Sokratischen Tutor beendet. …“ (in der Session-Sprache).

**Beenden über den Schalter:** einen Request nur mit `stop_socratic_v2: true` schicken, **ohne** `user_query`. Die Session endet sofort, im Verlauf steht nur die Bestätigung des Tutors, keine Nutzernachricht. Das Flag ist idempotent: Läuft gerade keine Session (z. B. weil der Tutor schon selbst beendet hat), kommt nur die Bestätigung, ohne Fehler und ohne dass eine leere Chatfrage beantwortet wird.

**Beispiel Modus-Schalter (Moodle):**

```json
// Schalter an
{"thread_id": "550e8400-…", "course_id": 79, "module_id": [1234], "start_socratic_v2": true, "ui_language": "de"}

// Schalter aus
{"thread_id": "550e8400-…", "course_id": 79, "stop_socratic_v2": true, "ui_language": "de"}
```

Die Antwort (Begrüßung bzw. Bestätigung) kommt wie jede andere Antwort, inkl. `learning_mode_active` im `final`-Event. Ob das Frontend sie anzeigt, entscheidet ihr; im Verlauf (`/api/chat/history`) steht sie.

**Status (`learning_mode_active`):** Jede Antwort meldet, ob danach eine Tutor-Session läuft: `final`-Event im Stream, Feld in `/api/chat` und in `/api/chat/history/{thread_id}`. **Der Modus-Schalter sollte immer diesem Wert folgen**, nicht dem lokalen Klickzustand. Eine Session endet nämlich auch ohne Schalter:

- per Schlüsselwort im Chat („stopp“),
- wenn der Tutor sie selbst beendet (die Person will nur eine schnelle Antwort),
- nach der Abschluss-Zusammenfassung.

Außerdem startet sie gar nicht erst, wenn kein Kurs/Modul gewählt ist. Nach einem Page-Reload liefert `/api/chat/history/{thread_id}` den Status mit.

---

## 11. Session-Handling (thread_id)

Der Gesprächsverlauf liegt **serverseitig** unter der `thread_id`. Das Frontend schickt bei jeder Nachricht nur die neue Nutzerfrage plus diese ID — nie den ganzen Verlauf.

### Möglicher Ablauf (Beispiel Moodle)

Wo und wie die ID abgelegt wird, entscheidet das Frontend. Ein denkbarer Weg — `localStorage`-Key pro Kurs, damit verschiedene Kurse getrennte Konversationen haben: `kic_thread_id_course{course_id}`.

| Zeitpunkt | Aktion |
|-----------|--------|
| Seite lädt | `localStorage.getItem("kic_thread_id_course79")` lesen |
| ID vorhanden | `GET /api/chat/history/{thread_id}` → Verlauf rendern |
| ID nicht vorhanden | Leerer Chat; die erste Anfrage geht mit `thread_id: null` raus |
| `meta`-Event empfangen | `localStorage.setItem("kic_thread_id_course79", thread_id)` |
| 30 min Inaktivität (clientseitig) | `localStorage.removeItem(...)` → nächste Anfrage startet eine neue Session |
| Button „Neues Gespräch" | `localStorage.removeItem(...)` → nächste Anfrage mit `thread_id: null` |

### Serverseitige Lebensdauer

- Sessions verfallen nach **24 h ohne Zugriff** (serverseitige TTL, wird bei jedem Zugriff erneuert; konfigurierbar).
- Ein Server-Neustart (Deployment) kann Sessions verwerfen.

In beiden Fällen liefert `GET /api/chat/history/{id}` einfach eine leere Liste, und ein `POST` mit dieser ID startet stillschweigend eine neue, leere Konversation — **kein Fehler**. Die für Nutzende sichtbare „Sitzung abgelaufen"-Logik (üblicherweise 30 min) liegt bewusst im Frontend.

---

## 12. Fehlercodes

| Status | Bedeutung | Typischer Auslöser |
|--------|-----------|--------------------|
| **200** | OK | — |
| **400** | Fachliche Validierung fehlgeschlagen | `module_id` ohne `course_id`; unbekannte `course_id`/`module_id`; `score` außerhalb 0–1; `user_query` fehlt (ohne Schalter-Flag); `start_socratic_v2` und `stop_socratic_v2` gleichzeitig |
| **401** | Ungültiger API-Key | Falscher `Api-Key`-Header |
| **403** | Kein API-Key | `Api-Key`-Header fehlt |
| **422** | Schema-Validierung fehlgeschlagen | Pflichtfeld fehlt, falscher Typ, unbekannter `model`-Wert |
| **500** | Serverfehler | Unerwarteter Backend-Fehler (bei `/api/chat/stream` nur *vor* Streamstart; danach als `error`-Event) |

Fehlerbody-Format (FastAPI-Standard):

```json
{"detail": "no course found with the given id: 999."}
```

Bekannte 400-Meldungen:

| `detail` | Ursache |
|----------|---------|
| `module_id is required when course_id is set.` | Wird — trotz des irreführenden Wortlauts — ausgelöst, wenn `module_id` **ohne** `course_id` gesendet wird |
| `no course found with the given id: <id>.` | Die `course_id` ist im Suchindex nicht vorhanden (Kurs nicht ingestiert oder falsche ID) |
| `no module found with the given id(s): [<ids>].` | Mindestens eine `module_id` ist im Suchindex nicht vorhanden |
| `user_query is required (only start_socratic_v2 / stop_socratic_v2 requests may omit it).` | `user_query` fehlt bei einem normalen Chat-Request |
| `start_socratic_v2 and stop_socratic_v2 cannot both be set.` | Beide Schalter-Flags im selben Request |

Die beiden „not found"-Fälle sind ein häufiger Stolperstein bei der Erstintegration: Gültig sind nur Kurse und Module, die ins Backend eingelesen wurden. Wird eine ID abgewiesen, obwohl sie in Moodle existiert, ist der Kurs im Assistenten schlicht noch nicht verfügbar — neue und geänderte Inhalte zieht der nächtliche Ingest nach, danach also noch einmal probieren.

---

## 13. Das Wichtigste auf einen Blick

| Thema | Kurz gefasst |
|-------|--------------|
| Zugang | Base URL und API-Key kommen von uns; der Key gehört nicht in ausgelieferten Browser-Code ([Abschnitt 1](#1-überblick)) |
| CORS | Nur relevant, wenn JavaScript im Browser die API direkt aufruft — serverseitige Aufrufe sind nicht betroffen |
| Chat | `/api/chat/stream` liefert NDJSON-Events, `/api/chat` die fertige Antwort in einem Stück ([3](#3-post-apichatstream--chat-mit-streaming-empfohlen), [4](#4-post-apichat--chat-ohne-streaming)) |
| Session | Die `thread_id` kommt im `meta`-Event und gehört bei jeder Folgenachricht wieder in den Request; ohne sie beginnt eine neue Konversation ([11](#11-session-handling-thread_id)) |
| Wiederherstellung | `/api/chat/history/{id}` liefert das serverseitig vorgehaltene Fenster — nicht den kompletten Verlauf ([5](#5-get-apichathistorythread_id)) |
| Darstellung | Antworten enthalten HTML-Quellenlinks; `final.message` ist die verbindliche Fassung der Antwort ([9](#9-antwortformat-quellen-fallbacks-sprache)) |
| Feedback | `response_id` der Antwort plus `score` 0 oder 1 ([6](#6-post-apifeedback)) |
| Fehler | Vor dem Stream als HTTP-Status, innerhalb des Streams als `error`-Event ([12](#12-fehlercodes)) |
| Sokratischer Tutor | Start und Ende über die Flags `start_socratic_v2` / `stop_socratic_v2` ohne sichtbare Nachricht (oder Beenden per Schlüsselwort im Chat); `learning_mode_active` in jeder Antwort steuert den Modus-Schalter ([10](#10-lernmodus-sokratischer-tutor)) |
| `model` | Kann entfallen — die Modellwahl liegt im Backend ([8](#8-modellwahl-feld-model)) |
