FROM python:3.11-slim

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 -ms /bin/bash appuser

RUN apt-get update && apt-get install -y \
    build-essential && \
    pip install 'poetry==1.7.1'

COPY poetry.lock pyproject.toml ./
RUN poetry config virtualenvs.create false && \
    poetry install --with embedder --no-interaction --no-ansi

USER appuser
WORKDIR /home/appuser

RUN huggingface-cli download intfloat/multilingual-e5-large

COPY . .
CMD ["uvicorn", "src.embedder.embedding_api:app", "--host", "0.0.0.0", "--port", "80"]
