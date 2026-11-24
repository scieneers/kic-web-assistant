#!/usr/bin/env python3
"""Start the durable ingest orchestration on Azure."""

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
FUNCTION_KEY = _env.get("DATALOADER_FUNCTION_KEY") or os.getenv("DATALOADER_FUNCTION_KEY", "")


def start_ingest(base_url: str, key: str) -> dict:
    url = f"{base_url.rstrip('/')}/api/start_ingest?code={key}"
    req = urllib.request.Request(url, method="POST", data=b"", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode()), resp.status
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if e.code == 409:
            try:
                data = json.loads(body)
                print(f"\nIngest already running.")
                print(f"   Instance:  {data.get('instanceId')}")
                print(f"   Status:    {data.get('runtimeStatus')}")
                print(f"   Status URL:{data.get('statusQueryGetUri')}")
            except Exception:
                print(f"HTTP 409: {body}")
            sys.exit(1)
        print(f"HTTP {e.code}: {body}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Start Azure ingest orchestration")
    parser.add_argument("--url", default=FUNCTION_APP_URL, help="Function App base URL")
    parser.add_argument("--key", default=FUNCTION_KEY, help="Function key")
    args = parser.parse_args()

    if not args.url or not args.key:
        print("ERROR: DATALOADER_FUNCTION_APP_URL and DATALOADER_FUNCTION_KEY not set in .env (or pass --url / --key).")
        sys.exit(1)

    print("Starting ingest...")
    response, status_code = start_ingest(args.url, args.key)

    print(f"\nIngest started (HTTP {status_code})")
    print(f"   Instance:   {response.get('instanceId')}")
    print(f"   Run ID:     {response.get('run_id')}")
    if response.get("log_url"):
        print(f"   Log URL:    {response.get('log_url')}")
    if response.get("statusQueryGetUri"):
        print(f"\n   Status URL: {response.get('statusQueryGetUri')}")
    print()
    print("Check status with:")
    print(f"   python scripts/check_ingest_status.py")


if __name__ == "__main__":
    main()
