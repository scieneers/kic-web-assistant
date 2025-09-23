# KIC Loader – Azure Function App

Der **KIC Loader** ist eine als **Azure Function App** deployte Anwendung, die regelmäßig Kursinhalte aus verschiedenen Systemen sammelt, verarbeitet und als Embeddings in einer Vektor-Datenbank (Qdrant) ablegt.

---

## Übersicht

- **Name der Function App:** `kicwa-dataloader-lab`
- **Laufzeitumgebung:** Docker-Image (Python)
- **Container Registry (Lab):** `kicwaacrlab`
- **Docker Image Name:** `kic-loader`
- **Aktuelles Tag (Lab):** `1.0.3`
- **Einstiegspunkt:** `src/loaders/function_app.py`

---

## Trigger

Die Function App wird über einen **Timer-Trigger** gestartet:

- **Beim Start der App:** einmalig
- **Regelmäßig:** jeden **Donnerstag** um **04:00 UTC**
- **Durchschnittliche Laufzeit:** ca. **5 Stunden**

---

## Verarbeitete Datenquellen

Der Loader sammelt Inhalte aus folgenden Systemen:

- **Moochup** – grundlegende Kursinformationen
- **Moodle** – Textinhalte aus ca. 140 Kursen inkl.
  - Kursinhalte
  - Transkripte von Videos (lokal eingebunden, YouTube, H5P)
- **Drupal (ki-campus.org)** – Textinhalte der Webseite

---

## Verarbeitungsschritte

1. **Datensammlung** aus den oben genannten Quellen
2. **Chunking** der Inhalte (Aufteilung in kleinere Einheiten)
3. **Embedding-Erstellung** (Umwandlung in Vektor-Repräsentationen)
4. **Speichern in Qdrant**
   - Ziel-Collection: `web-assistant`
   - Falls bereits vorhanden, wird die Collection **überschrieben**
5. **Produktivumgebung (nur Prod):**
   - Übertragung der neuen Collection ins Produktivsystem
   - ⚠️ **In der Lab-Umgebung deaktiviert**

---

## Deployment & Images

### Neues Docker-Image bauen & hochladen

```bash
IMAGE_NAME="kic-loader"
IMAGE_TAG="1.0.2"
DOCKERFILE="src/loaders/Dockerfile"
PLATFORM="linux/amd64"
ACR_NAME="kicwaacrlab"

ACR_IMAGE="${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}"

az acr login --name "$ACR_NAME"

docker buildx build --pull --no-cache \
  --platform "$PLATFORM" \
  -t "$ACR_IMAGE" \
  -f "$DOCKERFILE" .
