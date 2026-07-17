#!/usr/bin/env bash
set -euo pipefail

# Startet den im Image mitinstallierten Redis nur, wenn REDIS_URL auf
# localhost zeigt (Default im Dockerfile). Für den Wechsel auf Azure Managed
# Redis genügt es, REDIS_URL als App-Setting zu überschreiben
# (rediss://:<access-key>@<name>.<region>.redis.azure.net:10000) — dann
# startet hier kein lokaler Redis.
if [[ "${REDIS_URL:-}" == *"localhost"* || "${REDIS_URL:-}" == *"127.0.0.1"* ]]; then
    # Kein Snapshotting/AOF: Chat-Sessions sind bewusst flüchtig, Verlust bei
    # Container-Neustart ist akzeptiert. maxmemory als Sicherheitsnetz neben
    # der TTL des Checkpointers (CHAT_TTL_MINUTES).
    redis-server \
        --bind 127.0.0.1 \
        --port 6379 \
        --save "" \
        --appendonly no \
        --maxmemory 256mb \
        --maxmemory-policy allkeys-lru &

    for _ in $(seq 1 40); do
        redis-cli -p 6379 ping >/dev/null 2>&1 && break
        sleep 0.25
    done
    if ! redis-cli -p 6379 ping >/dev/null 2>&1; then
        echo "Lokaler Redis ist nicht gestartet — Abbruch." >&2
        exit 1
    fi
fi

exec gunicorn -b 0.0.0.0:80 -c src/api/gunicorn.conf.py src.api.rest:app
