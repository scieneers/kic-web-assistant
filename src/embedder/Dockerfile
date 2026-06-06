#  ╭──────────────────────────────────────────────────────────╮
#  │                        Base - Stage                      │
#  ╰──────────────────────────────────────────────────────────╯
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="$VENV_PATH/bin:$PATH"

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 -ms /bin/bash appuser

#  ╭──────────────────────────────────────────────────────────╮
#  │                       Build - Stage                      │
#  ╰──────────────────────────────────────────────────────────╯
FROM base AS build

RUN apt-get update && apt-get install -y \
    build-essential && \
    pip install uv

WORKDIR $PYSETUP_PATH
COPY uv.lock pyproject.toml ./
RUN uv sync --no-dev --no-install-project

USER appuser
WORKDIR /home/appuser
RUN huggingface-cli download intfloat/multilingual-e5-large
COPY --chown=appuser:appuser . ./

#  ╭──────────────────────────────────────────────────────────╮
#  │                     Runtime - Stage                      │
#  ╰──────────────────────────────────────────────────────────╯
FROM base AS runtime

USER appuser
WORKDIR /home/appuser

COPY --from=build $VENV_PATH $VENV_PATH
COPY --from=build --chown=appuser:appuser /home/appuser /home/appuser
CMD ["uvicorn", "src.embedder.embedding_api:app", "--host", "0.0.0.0", "--port", "80"]
