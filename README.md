# KI-Campus Web Assistant

KI-gestützter Chat-Assistent für [ki-campus.org](https://ki-campus.org): beantwortet Fragen zu Plattform- und Kursinhalten per RAG (Azure AI Search + LLM via GWDG/Azure), mit Streaming-API, Chat-Persistenz und sokratischem Lernmodus.

**Dokumentation:** [ARCHITECTURE.md](documentation/ARCHITECTURE.md) (Systemüberblick) · [SCRIPTS.md](documentation/SCRIPTS.md) (alle Kommandos & Ops-Skripte) · [CHAT_PERSISTENCE.md](documentation/CHAT_PERSISTENCE.md) (API-Schnittstelle für Frontends)

# Dependencies
- python 3.11
- [uv](https://docs.astral.sh/uv/) & `uv sync`
- task: `brew install go-task`
- install pre-commit hooks: [`pre-commit`](https://github.com/pre-commit/pre-commit) `install`

# Local development

## Konfiguration
Alle Umgebungsvariablen sind in [`src/env.py`](src/env.py) definiert und dokumentiert (Priorität: Env > `.env` > Azure Key Vault > Defaults); eine kommentierte Übersicht steht in [ARCHITECTURE.md, Abschnitt 14](documentation/ARCHITECTURE.md#14-konfiguration--umgebungsvariablen).

## VectorDB: Azure AI Search
The vector store is [Azure AI Search](https://learn.microsoft.com/azure/search/). Auth is keyless via `DefaultAzureCredential`, so for local development run `az login` and make sure your IP is in the service's `allowed_ip_ranges`.

Minimal environment variables:
- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_INDEX`

## Backend (FastAPI) starten
```bash
uv run uvicorn src.api.rest:app --port 8000
```

## Test-Frontend (Streamlit) starten
```bash
uv run streamlit run src/frontend/frontend.py
```
Das Streamlit-Frontend ist ein reines Test-Frontend (optional passwortgeschützt via `FRONTEND_PASSWORD`); produktive Oberflächen sind die Moodle-/Drupal-Integrationen.

## Tests
```bash
uv run pytest -m "not integration"   # schnell, ohne Credentials/Netz
uv run pytest                        # inkl. Integrationstests (braucht az login + Env)
```

## Docker / Deployment
Build & Deploy laufen über das [Taskfile](Taskfile.yml) gegen die Azure Container Registry:
```bash
task build-api              # REST API Image
task build-frontend         # Streamlit Frontend
task build-loader           # Data Loader
task push-latest-images     # Push zur Azure Container Registry
task deploy-loader ENV=dev  # Deploy zu Azure Functions
```
Lokal ein Image mit Azure-Credentials laufen lassen:
```bash
docker run -it --rm -p 80:80 -v ~/.azure:/home/appuser/.azure kicacrdev.azurecr.io/rest-api:latest
```
Hinweis für Apple-Silicon-Macs: Images für die Cloud mit `--platform linux/amd64` bauen.

# Data Extraction

# Moodle
To access content from Moodle, you need access to Moodle courses via the REST API. To set up the integration, do the following steps:
0. Get admin access to moodle.
1. Enable Web Services: _Site Administration_ -> _General_ -> _Advanced Features_ -> _Enable web services_
2. Enable REST Protocol: _Site Administration_ -> _Server_ -> _Web Services_ -> _Manage Protocols_ -> _Enable REST protocol_
3. (Optional): Create a technical new user/roles
4. Create a new external service _Site Administration_ -> _Server_ -> _External services_. Give it a name and enable _Enabled_, _Authorized users only_ and _Can download files_ (under _Show more..._).
5. Add the user as an _Authorised User_ to the external service.
6. Add the following functions to the external service:
    - core_block_get_course_blocks
    - core_course_get_categories
    - core_course_get_contents
    - core_course_get_course_content_items
    - core_course_get_course_module
    - core_course_get_courses
    - core_course_get_module
7. Create a token for the user and external service under _Site Administration_ -> _Server_ -> _Manage tokens_. This allows you to authenticate against the REST API.

You can try it out with a GET request against this url (swap TOKEN for your token und FUNCTION against the function to test):
https://ki-campus-test.fernuni-hagen.de/webservice/rest/server.php?wstoken=TOKEN&wsfunction=FUNCTION&moodlewsrestformat=json
