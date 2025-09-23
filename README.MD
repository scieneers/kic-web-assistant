# KIC Web Assistant – Lokale Einrichtung

Der folgende Abschnitt beschreibt, wie das Projekt für die lokale Entwicklung aufgesetzt wird. Es enthält alle Schritte von Repository-Klon über Abhängigkeiten bis zur Aktivierung des Virtual Environments.

## Abhängigkeiten
- python 3.11
- poetry 1.7.1 & `poetry install`
- install pre-commit hooks: [`pre-commit`](https://github.com/pre-commit/pre-commit) `install`

### 1. Repository klonen
```bash
git clone git@github.com:scieneers/kic-web-assistant.git
cd kic-web-assistant
```

### 2. Zum Lab-Branch wechseln
```bash
git checkout lab
```

### 3. Abhängigkeiten installieren
```bash
poetry install --all-groups
```

### 4. Virtual Environment aktivieren
```bash
source .venv/bin/activate
```

### 5. Pre-commit Hooks einrichten
```bash
pre-commit install
```

### 6. Umgebungsvariablen konfigurieren
Kopiere die `.env`-Datei in das Hauptverzeichnis des Projekts.

## Lokale Entwicklung
### VS Code

Wenn du VS Code verwendst, gibt es eine Debugging-Konfiguration für das Front- und Backend, mit der du Front- und Backend lokal starten und debuggen kannst.


### VectorDB [Qdrant](https://github.com/qdrant/qdrant-client)
`docker pull qdrant/qdrant:v1.53.3`
`docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant:v1.53.3`


# Related Documentation

## KIC Loader
Der KIC Loader ist verantwortlich für das Sammeln von Kursinhalten aus Moochup, Moodle und Drupal, die Verarbeitung dieser Inhalte zu Embeddings und das Speichern in der Qdrant-Vektordatenbank.

Für detaillierte Informationen zu Einrichtung, Triggern und Deployment siehe die [KIC Loader Dokumentation](./LOADER.md).

## KIC Frontend
Das KIC Frontend ist eine **Streamlit Web App**, mit der die Funktionalität der REST-API getestet werden kann.

Für detaillierte Informationen zu Nutzung, Filtern und Deployment siehe die [KIC Frontend Dokumentation](./FRONTEND.md).

## KIC Backend
Das KIC Backend ist die Kernkomponente des RAG-Systems. Es verarbeitet die Anfragen, sucht relevante Inhalte in der Vektor-Datenbank und liefert die Antworten zurück an das Frontend.

Für detaillierte Informationen zu Architektur, Fallback-Mechanismus und Deployment siehe die [KIC Backend Dokumentation](./BACKEND.md).
