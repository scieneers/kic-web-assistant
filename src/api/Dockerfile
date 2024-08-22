#  ╭──────────────────────────────────────────────────────────╮
#  │                        Base - Stage                      │
#  ╰──────────────────────────────────────────────────────────╯
FROM python:3.11-slim AS base

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

RUN apt-get update

#  ╭──────────────────────────────────────────────────────────╮
#  │                       Build - Stage                      │
#  ╰──────────────────────────────────────────────────────────╯
FROM base AS build

RUN apt-get install -y \
    build-essential && \
    pip install 'poetry==1.7.1'

WORKDIR $PYSETUP_PATH
COPY poetry.lock pyproject.toml ./
RUN poetry install --with api --no-interaction --no-ansi

USER appuser
WORKDIR /home/appuser

COPY --chown=appuser:appuser . ./

#  ╭──────────────────────────────────────────────────────────╮
#  │                     Runtime - Stage                      │
#  ╰──────────────────────────────────────────────────────────╯
FROM base AS runtime

# Install Azure CLI for defaultcredentials cli login in lockally hosted container
RUN apt-get install -y curl
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

USER appuser
WORKDIR /home/appuser

COPY --from=build $VENV_PATH $VENV_PATH
COPY --from=build --chown=appuser:appuser /home/appuser /home/appuser

RUN mkdir nltk_data
ENV NLTK_DATA=nltk_data
RUN python -m nltk.downloader stopwords punkt punkt_tab
EXPOSE 80

CMD ["gunicorn" , "-b", "0.0.0.0:80", "-c", "src/api/gunicorn.conf.py", "src.api.rest:app"]
