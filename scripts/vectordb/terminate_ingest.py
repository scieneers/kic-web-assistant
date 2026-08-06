#!/usr/bin/env python3
"""Terminate a running ingest orchestration on Azure."""

import os
import sys
import json
import argparse
import urllib.request
import urllib.error
from pathlib import Path


def _load_env() -> dict[str, str]:
    env_file = Path(__file__).resolve().parents[2] / ".env"
    result: dict[str, str] = {}
    if not env_file.exists():
        return result
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


_env = _load_env()
FUNCTION_APP_URL = _env.get("DATALOADER_FUNCTION_APP_URL") or os.getenv("DATALOADER_FUNCTION_APP_URL", "")
MASTER_KEY = _env.get("DATALOADER_MASTER_KEY") or os.getenv("DATALOADER_MASTER_KEY", "")

DEFAULT_INSTANCE_ID = "ingest_singleton"


def get_status(instance_id: str, base_url: str, key: str) -> dict | None:
    url = f"{base_url.rstrip('/')}/api/check_status/{instance_id}?code={key}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def terminate(instance_id: str, base_url: str, master_key: str, reason: str) -> None:
    url = (
        f"{base_url.rstrip('/')}/runtime/webhooks/durabletask/instances"
        f"/{instance_id}/terminate?reason={urllib.parse.quote(reason)}&code={master_key}"
    )
    req = urllib.request.Request(url, method="POST", data=b"")
    try:
        with urllib.request.urlopen(req) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"HTTP {e.code}: {body}")
        sys.exit(1)


def main():
    import urllib.parse  # noqa: PLC0415 — needed for quote inside terminate()

    parser = argparse.ArgumentParser(description="Terminate an Azure ingest orchestration")
    parser.add_argument(
        "instance_id",
        nargs="?",
        default=DEFAULT_INSTANCE_ID,
        help=f"Orchestration instance ID (default: {DEFAULT_INSTANCE_ID})",
    )
    parser.add_argument("--url", default=FUNCTION_APP_URL, help="Function App base URL")
    parser.add_argument("--key", default=MASTER_KEY, help="Master key")
    parser.add_argument("--reason", default="manual termination", help="Reason for termination")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    if not args.key:
        print("ERROR: DATALOADER_MASTER_KEY not set in .env (or pass --key).")
        print("Find it in: Portal → Function App → App keys → _master")
        sys.exit(1)

    # Check current status first
    try:
        status = get_status(args.instance_id, args.url, args.key)
    except Exception:
        status = None  # status check uses function key; master key may not work for it

    if status:
        runtime = status.get("runtimeStatus", "unknown")
        symbols = {"Running": "⏳", "Completed": "✅", "Failed": "❌", "Terminated": "🛑", "Pending": "🕐"}
        print(f"\nCurrent status: {symbols.get(runtime, '❓')}  {runtime}")
        if runtime in ("Completed", "Failed", "Terminated"):
            print(f"Instance is already in a terminal state ({runtime}) — nothing to terminate.")
            sys.exit(0)

    if not args.yes:
        answer = input(f"\nTerminate '{args.instance_id}'? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    terminate(args.instance_id, args.url, args.key, args.reason)
    print(f"🛑  Termination request sent for '{args.instance_id}'.")
    print("    It may take a few seconds for the status to update.")


if __name__ == "__main__":
    main()
