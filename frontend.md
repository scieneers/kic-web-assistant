# KIC Frontend – Streamlit Web App

Das **KIC Frontend** ist eine einfache **Streamlit Web App**, mit der die Funktionalität der REST-API getestet werden kann.

---

## Übersicht

- **Name der Web App:** `kic-frontend-lab`
- **Laufzeitumgebung:** Docker-Image (Python/Streamlit)
- **Container Registry (Lab):** `kicwaacrlab`
- **Docker Image Name:** `kic-frontend`
- **Aktuelles Tag (Lab):** `1.8.1`
- **Einstiegspunkt:** `src/frontend/frontend.py`

---

## Nutzung & Funktionalität

- Beim ersten Laden der App werden **alle Kursinformationen geladen** (Dauer ca. 1 Minute).
- Aktuell sind in der Lab-Umgebung nur die beiden LLMs GPT-4 (Azure) und Llama3 (GWDG) verfügbar.
- Danach kann links ein beliebiger **Kurs** oder **Modul** ausgewählt werden.
  - Die REST-API wird sich in ihren Antworten nur auf Inhalte dieses Kurses oder Moduls beziehen.
- Wenn die Option **„Alle Inhalte aus Drupal“** ausgewählt wird:
  - Es werden **nur Inhalte aus Drupal** berücksichtigt
  - Inhalte aus Moodle werden **nicht** in den Antworten verwendet

---

## Deployment & Images

### Neues Docker-Image bauen & hochladen

```bash
IMAGE_NAME="kic-frontend"
IMAGE_TAG="1.8.1"
DOCKERFILE="src/frontend/Dockerfile"
PLATFORM="linux/amd64"
ACR_NAME="kicwaacrlab"

ACR_IMAGE="${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}"

az acr login --name "$ACR_NAME"

docker buildx build --pull --no-cache \
  --platform "$PLATFORM" \
  -t "$ACR_IMAGE" \
  -f "$DOCKERFILE" .
