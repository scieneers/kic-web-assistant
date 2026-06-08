# To enable ssh & remote debugging on app service change the base image to the one below
# FROM mcr.microsoft.com/azure-functions/python:4-python3.11-appservice
FROM mcr.microsoft.com/azure-functions/python:4-python3.11

ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true \
    AzureWebJobsFeatureFlags=EnableWorkerIndexing \
    PYTHONPATH="$PYTHONPATH:/home/site/wwwroot:/home/site"

WORKDIR /home/site/wwwroot

RUN pip install uv

COPY uv.lock pyproject.toml ./
# `uv sync` only targets a project venv, but the Azure Functions host runs the system
# interpreter (/usr/local/bin/python) and won't see a venv. Export the locked deps and
# install them into the system environment instead (--system is valid for `uv pip`).
RUN uv export --frozen --no-dev --no-emit-project --extra azfunction --format requirements-txt > requirements.txt \
    && uv pip install --system -r requirements.txt

COPY . /home/site
COPY src/loaders/. /home/site/wwwroot
