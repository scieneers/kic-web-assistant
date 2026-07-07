#!/usr/bin/env python3
"""Check the status of a running ingest orchestration on Azure."""

import os
import sys
import json
import argparse
import urllib.request
import urllib.error
from pathlib import Path

DEFAULT_INSTANCE_ID = "ingest_singleton"


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
FUNCTION_KEY = _env.get("DATALOADER_FUNCTION_KEY") or os.getenv("DATALOADER_FUNCTION_KEY", "")


def check_status(instance_id: str, base_url: str, key: str) -> dict:
    url = f"{base_url.rstrip('/')}/api/check_status/{instance_id}?code={key}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if e.code == 404:
            print(f"Instance '{instance_id}' not found.")
            sys.exit(1)
        print(f"HTTP {e.code}: {body}")
        sys.exit(1)


def fetch_logs(log_url: str) -> None:
    try:
        with urllib.request.urlopen(log_url) as resp:
            print(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"Could not fetch logs (HTTP {e.code}): {e.read().decode()}")


def main():
    parser = argparse.ArgumentParser(description="Check Azure ingest orchestration status")
    parser.add_argument(
        "instance_id",
        nargs="?",
        default=DEFAULT_INSTANCE_ID,
        help=f"Orchestration instance ID (default: {DEFAULT_INSTANCE_ID})",
    )
    parser.add_argument("--url", default=FUNCTION_APP_URL, help="Function App base URL")
    parser.add_argument("--key", default=FUNCTION_KEY, help="Function key")
    parser.add_argument("--logs", action="store_true", help="Fetch and print the run log")
    parser.add_argument("--log-url", help="Fetch logs from this SAS URL directly (skips status check)")
    args = parser.parse_args()

    if args.log_url:
        fetch_logs(args.log_url)
        return

    if not args.url or not args.key:
        print("ERROR: DATALOADER_FUNCTION_APP_URL and DATALOADER_FUNCTION_KEY not set in .env (or pass --url / --key).")
        sys.exit(1)

    status = check_status(args.instance_id, args.url, args.key)

    runtime = status.get("runtimeStatus", "unknown")
    symbols = {"Running": "⏳", "Completed": "✅", "Failed": "❌", "Terminated": "🛑", "Pending": "🕐"}
    icon = symbols.get(runtime, "❓")

    print(f"\n{icon}  Status: {runtime}")
    print(f"   Instance:    {status.get('instanceId')}")
    print(f"   Created:     {status.get('createdTime')}")
    print(f"   Last update: {status.get('lastUpdatedTime')}")

    output = status.get("output")
    log_url = None
    if isinstance(output, dict):
        counts = output.get("counts", {})
        log_url = output.get("log_url")
        if counts:
            print("\n   Counts:")
            for k, v in counts.items():
                print(f"     {k}: {v}")
        if log_url:
            print(f"\n   Log URL: {log_url}")
    elif output:
        print(f"   Output: {json.dumps(output, ensure_ascii=False, indent=4)}")

    print()

    if args.logs:
        if not log_url:
            print("No log_url in output. Run may still be in progress or log is unavailable.")
            sys.exit(1)
        print("─" * 80)
        fetch_logs(log_url)
        print("─" * 80)


if __name__ == "__main__":
    main()
