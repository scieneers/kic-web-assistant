#!/usr/bin/env bash
set -euo pipefail

if [[ "${REDIS_URL:-}" == *"localhost"* || "${REDIS_URL:-}" == *"127.0.0.1"* ]]; then
    echo "Warte auf Redis-Sidecar (Port 6379)..."
    ready=false
    # 120 Versuche à 1 Sekunde = 2 Minuten Puffer für Image Pull & Container Start
    for i in $(seq 1 120); do
        if (: < /dev/tcp/127.0.0.1/6379) 2>/dev/null; then
            ready=true
            echo "Redis-Sidecar Port ist offen (Versuch $i)."
            break
        fi
        sleep 1
    done
    if ! $ready; then
        echo "Redis-Sidecar unter ${REDIS_URL} nach 120s nicht erreichbar — Abbruch." >&2
        exit 1
    fi
fi

exec gunicorn -b 0.0.0.0:80 -c src/api/gunicorn.conf.py src.api.rest:app