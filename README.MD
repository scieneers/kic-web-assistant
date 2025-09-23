# Quellcode

Dieser Quellcode ist auf [`GitHub`](https://github.com/scieneers/kic-web-assistant) gehostet.
Für die Lab-Umgebung bitte ausschließlich den `lab`-Branch sowie Branches verwenden, die von diesem Branch geforkt wurden.


## Related Documentation

### KIC Loader
Der KIC Loader ist verantwortlich für das Sammeln von Kursinhalten aus Moochup, Moodle und Drupal, die Verarbeitung dieser Inhalte zu Embeddings und das Speichern in der Qdrant-Vektordatenbank.

Für detaillierte Informationen zu Einrichtung, Triggern und Deployment siehe die [KIC Loader Dokumentation](./LOADER.md).

### KIC Frontend
Das KIC Frontend ist eine **Streamlit Web App**, mit der die Funktionalität der REST-API getestet werden kann.

Für detaillierte Informationen zu Nutzung, Filtern und Deployment siehe die [KIC Frontend Dokumentation](./FRONTEND.md).

### KIC Backend
Das KIC Backend ist die Kernkomponente des RAG-Systems. Es verarbeitet die Anfragen, sucht relevante Inhalte in der Vektor-Datenbank und liefert die Antworten zurück an das Frontend.

Für detaillierte Informationen zu Architektur, Fallback-Mechanismus und Deployment siehe die [KIC Backend Dokumentation](./BACKEND.md).


# Abhängigkeiten
- python 3.11
- poetry 1.7.1 & `poetry install`
- task: `brew install go-task`
- install pre-commit hooks: [`pre-commit`](https://github.com/pre-commit/pre-commit) `install`

# Lokale Entwicklung
## VS Code

Wenn du VS Code verwendst, gibt es eine Debugging-Konfiguration für das Front- und Backend, mit der du Front- und Backend lokal starten und debuggen kannst.


## VectorDB [Qdrant](https://github.com/qdrant/qdrant-client)
`docker pull qdrant/qdrant:v1.53.3`
`docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant:v1.53.3`
