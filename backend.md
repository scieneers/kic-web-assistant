# KIC Backend – REST API

Das **KIC Backend** ist die Kernkomponente des RAG-Systems und nutzt das Framework **LangChain**, um die verschiedenen Bestandteile zusammenzuführen.

---

## Übersicht

- **Name der Web App:** `kic-restapi-lab`
- **Laufzeitumgebung:** Docker-Image (Python/FastAPI)
- **Container Registry (Lab):** `kicwaacrlab`
- **Docker Image Name:** `rest-api`
- **Aktuelles Tag (Lab):** `1.8.7`
- **Einstiegspunkt:** `src/api/rest.py`

---

## Funktionsweise

1. **Kontextualisierung der Frage**
   - Bei einem Chat-Call wird zunächst die Frage des Users kontextualisiert.
   - Dies geschieht im **Contextualizer**: `src/llm/tools/contextualizer.py`

2. **Embedding & Suche in Vektor-Datenbank**
   - Die Frage wird in ein Embedding umgewandelt und in der Vektor-Datenbank nach passenden Inhalten aus Moodle und Drupal gesucht.

3. **Question Answerer**
   - Gefundene Inhalte, der Prompt, die Frage des Users und ggf. vorherige Chats werden an das Sprachmodell übergeben.
   - Dies geschieht im **Question Answerer**: `src/llm/tools/question_answerer.py`

4. **Antwort an das Frontend**
   - Die Antwort des Modells wird zurück an das entsprechende Frontend gegeben
     - Streamlit / Drupal-Plugin / Moodle-Plugin

---

## Fallback-Mechanismus

- Für Chat-Anfragen gibt es einen **Fallback** auf das in Azure gehostete GPT-4o.
- Falls das GWDG-Sprachmodell eine Exception auslöst oder nicht innerhalb von 7 Sekunden antwortet:
  - Es wird ein Call gegen GPT-4o gemacht
  - Alle folgenden Chats werden für **5 Minuten** ebenfalls an GPT-4o weitergeleitet
  - Danach wird erneut versucht, das GWDG-Modell zu erreichen

---

## Deployment & Images

### Neues Docker-Image bauen & hochladen

```bash
IMAGE_NAME="rest-api"
IMAGE_TAG="1.8.7"
DOCKERFILE="src/api/Dockerfile"
PLATFORM="linux/amd64"
ACR_NAME="kicwaacrlab"

ACR_IMAGE="${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}"

az acr login --name "$ACR_NAME"

docker buildx build --pull --no-cache \
  --platform "$PLATFORM" \
  -t "$ACR_IMAGE" \
  -f "$DOCKERFILE" .
