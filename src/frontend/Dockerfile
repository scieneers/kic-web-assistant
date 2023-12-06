FROM python:3.11-slim

EXPOSE 8501

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 -ms /bin/bash appuser

RUN apt-get update && apt-get install -y \
    build-essential && \
    pip install 'poetry==1.7.1'

COPY poetry.lock pyproject.toml ./
RUN poetry config virtualenvs.create false && \
    poetry install --only frontend --no-interaction --no-ansi

USER appuser
WORKDIR /home/appuser

COPY . .
CMD streamlit run src/frontend/frontend.py