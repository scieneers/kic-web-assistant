# To enable ssh & remote debugging on app service change the base image to the one below
# FROM mcr.microsoft.com/azure-functions/python:4-python3.11-appservice
FROM mcr.microsoft.com/azure-functions/python:4-python3.11

ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    POETRY_HOME='/usr/local' \
    POETRY_VERSION=1.7.1 \
    PYTHONPATH="$PYTHONPATH:/home/site/wwwroot:/home/site"

WORKDIR /home/site/wwwroot

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    python3-venv \
    python3-pip \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /usr/local/bin/poetry /usr/bin/poetry

# Copy only dependency files first (cache layer)
COPY poetry.lock pyproject.toml ./

RUN poetry config virtualenvs.create false \
    && poetry install --with azfunction --no-interaction --no-ansi

# Copy the actual source
COPY . /home/site
COPY src/loaders/. /home/site/wwwroot
