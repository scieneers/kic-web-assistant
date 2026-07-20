#!/usr/bin/env bash
set -euo pipefail

# Redis läuft als eigener Sidecar-Container (IaC), nicht mehr in diesem
# Image. Bei REDIS_URL=localhost (Default, Sidecar teilt sich den
# Network-Namespace mit diesem Container) kurz warten, falls der Sidecar
# beim Cold-Start noch nicht bereit ist — es gibt keine garantierte
# Startreihenfolge zwischen App-Service-Sidecar-Containern. Für Azure
# Managed Redis (rediss://-URL) entfällt das Warten, da der Dienst bereits
# läuft.
if [[ "${REDIS_URL:-}" == *"localhost"* || "${REDIS_URL:-}" == *"127.0.0.1"* ]]; then
    ready=false
    for _ in $(seq 1 40); do
        (: < /dev/tcp/127.0.0.1/6379) 2>/dev/null && ready=true && break
        sleep 0.25
    done
    if ! $ready; then
        echo "Redis-Sidecar unter localhost:6379 nicht erreichbar — Abbruch." >&2
        exit 1
    fi
fi

exec gunicorn -b 0.0.0.0:80 -c src/api/gunicorn.conf.py src.api.rest:app
