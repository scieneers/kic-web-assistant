#  ╭──────────────────────────────────────────────────────────╮
#  │                        Base - Stage                      │
#  ╰──────────────────────────────────────────────────────────╯
FROM python:3.11-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 -ms /bin/bash appuser

#  ╭──────────────────────────────────────────────────────────╮
#  │                       Build - Stage                      │
#  ╰──────────────────────────────────────────────────────────╯
FROM base as build

RUN apt-get update && apt-get install -y \
    build-essential && \
    pip install 'poetry==1.7.1'

WORKDIR $PYSETUP_PATH
COPY ./poetry.lock ./pyproject.toml ./
RUN poetry install --with embedder --no-interaction --no-ansi

USER appuser
WORKDIR /home/appuser
RUN huggingface-cli download intfloat/multilingual-e5-large
COPY --chown=appuser:appuser . ./

#  ╭──────────────────────────────────────────────────────────╮
#  │                     Runtime - Stage                      │
#  ╰──────────────────────────────────────────────────────────╯
FROM base as runtime

USER appuser
WORKDIR /home/appuser

COPY --from=build $VENV_PATH $VENV_PATH
COPY --from=build --chown=appuser:appuser /home/appuser /home/appuser
CMD ["uvicorn", "src.embedder.embedding_api:app", "--host", "0.0.0.0", "--port", "80"]
