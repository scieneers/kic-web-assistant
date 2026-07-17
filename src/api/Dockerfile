#  ╭──────────────────────────────────────────────────────────╮
#  │                        Base - Stage                      │
#  ╰──────────────────────────────────────────────────────────╯
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="$VENV_PATH/bin:$PATH"
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 -ms /bin/bash appuser

RUN apt-get update

#  ╭──────────────────────────────────────────────────────────╮
#  │                       Build - Stage                      │
#  ╰──────────────────────────────────────────────────────────╯
FROM base AS build

RUN apt-get install -y \
    build-essential && \
    pip install uv

WORKDIR $PYSETUP_PATH
COPY uv.lock pyproject.toml ./
RUN uv sync --extra api --extra reranker --no-dev --no-install-project

USER appuser
WORKDIR /home/appuser

COPY --chown=appuser:appuser . ./

#  ╭──────────────────────────────────────────────────────────╮
#  │                     Runtime - Stage                      │
#  ╰──────────────────────────────────────────────────────────╯
FROM base AS runtime

# Install Azure CLI for defaultcredentials cli login in lockally hosted container
RUN apt-get install -y curl gnupg
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# Redis 8 aus dem offiziellen Redis-Repo — Debian liefert nur Redis 7 ohne die
# JSON/Search-Module, die langgraph-checkpoint-redis benötigt. Läuft als
# Checkpointer-Store für den Chat-State im selben Container (entrypoint.sh);
# per REDIS_URL-App-Setting auf Azure Managed Redis umschaltbar.
# Der Versions-Check bricht den Build ab, falls doch ein Redis <8 aufgelöst wird.
RUN curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg && \
    chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb bookworm main" > /etc/apt/sources.list.d/redis.list && \
    apt-get update && apt-get install -y redis && \
    redis-server --version && \
    redis-server --version | grep -qE "v=([89]|[1-9][0-9])\."

USER appuser
WORKDIR /home/appuser

COPY --from=build $VENV_PATH $VENV_PATH
COPY --from=build --chown=appuser:appuser /home/appuser /home/appuser

RUN mkdir nltk_data
ENV NLTK_DATA=nltk_data
RUN python -m nltk.downloader stopwords punkt punkt_tab
EXPOSE 80

# Default: eingebetteter Redis (entrypoint.sh startet ihn). Für Azure Managed
# Redis dieses Setting in den App Settings überschreiben (rediss://-URL).
ENV REDIS_URL="redis://localhost:6379"

CMD ["bash", "src/api/entrypoint.sh"]
