#  ╭──────────────────────────────────────────────────────────╮
#  │                        Base - Stage                      │
#  ╰──────────────────────────────────────────────────────────╯
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="$VENV_PATH/bin:$PATH"

EXPOSE 8501

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
RUN uv sync --extra frontend --extra api --extra reranker --no-dev --no-install-project

USER appuser
WORKDIR /home/appuser
COPY --chown=appuser:appuser . ./

#  ╭──────────────────────────────────────────────────────────╮
#  │                     Runtime - Stage                      │
#  ╰──────────────────────────────────────────────────────────╯
FROM base AS runtime

USER appuser
COPY --from=build $VENV_PATH $VENV_PATH
COPY --from=build --chown=appuser:appuser /home/appuser /home/appuser
WORKDIR /home/appuser

RUN mkdir nltk_data
ENV NLTK_DATA=nltk_data
RUN python -m nltk.downloader stopwords punkt punkt_tab
CMD ["python", "-m", "streamlit", "run", "src/frontend/frontend.py"]
