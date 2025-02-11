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

RUN curl -sSL https://install.python-poetry.org | python3 -

COPY poetry.lock pyproject.toml ./
RUN poetry config virtualenvs.create false && poetry install --with azfunction --no-interaction --no-ansi

COPY . /home/site
COPY src/loaders/. /home/site/wwwroot
