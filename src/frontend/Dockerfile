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

EXPOSE 8501

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
COPY poetry.lock pyproject.toml ./
RUN poetry install --with frontend,api --no-interaction --no-ansi

USER appuser
WORKDIR /home/appuser
COPY --chown=appuser:appuser . ./

#  ╭──────────────────────────────────────────────────────────╮
#  │                     Runtime - Stage                      │
#  ╰──────────────────────────────────────────────────────────╯
FROM base as runtime

USER appuser
COPY --from=build $VENV_PATH $VENV_PATH
COPY --from=build --chown=appuser:appuser /home/appuser /home/appuser
WORKDIR /home/appuser

RUN mkdir nltk_data
ENV NLTK_DATA=nltk_data
RUN python -m nltk.downloader stopwords punkt punkt_tab
CMD python -m streamlit run src/frontend/frontend.py
