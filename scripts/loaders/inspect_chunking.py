"""Manual chunking inspection script.

Usage (example):

  # ensure Azure login if you use Key Vault auth
  az login

  # run against a single Moodle module id (Production creds from Key Vault)
  python -m src.tests.loaders.inspect_chunking --module-id 2195 --max-chunks 5

This script prints:
 - raw extracted module text stats
 - normalized text stats
 - first N chunks with token counts

It is intentionally NOT a pytest test because it may hit external services.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from src.loaders.APICaller import APICaller
from src.loaders.moodle import Moodle
from src.loaders.helper import chunk_text_hierarchical, count_tokens, normalize_text_for_rag


def setup_production_moodle() -> Moodle:
    key_vault_name = os.environ.get("KEY_VAULT_NAME", "kicwa-keyvault-lab")
    key_vault_uri = f"https://{key_vault_name}.vault.azure.net/"
    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=key_vault_uri, credential=credential)

    prod_url = secret_client.get_secret("DATA-SOURCE-PRODUCTION-MOODLE-URL").value
    prod_token = secret_client.get_secret("DATA-SOURCE-PRODUCTION-MOODLE-TOKEN").value

    moodle = Moodle()
    moodle.base_url = prod_url
    moodle.api_endpoint = f"{prod_url}webservice/rest/server.php"
    moodle.token = prod_token
    moodle.function_params["wstoken"] = prod_token
    moodle.download_params["token"] = prod_token
    return moodle


def find_module(moodle: Moodle, module_id: int):
    courses = moodle.get_courses()
    for course in courses:
        try:
            topics = moodle.get_course_contents(course.id)
            for topic in topics:
                for module in topic.modules:
                    if module.id == module_id:
                        return course, module
        except Exception:
            continue
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-id", type=int, required=True)
    ap.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE_TOKENS", "900")))
    ap.add_argument("--chunk-overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP_TOKENS", "150")))
    ap.add_argument("--max-chunks", type=int, default=5)
    args = ap.parse_args()

    moodle = setup_production_moodle()
    course, module = find_module(moodle, args.module_id)
    if not module:
        print(f"Module {args.module_id} not found")
        return 2

    print("=" * 80)
    print(f"Course: {course.id} - {course.fullname}")
    print(f"Module: {module.id} - {module.name} ({module.modname})")
    print("URL:", getattr(module, "url", None))
    print("=" * 80)

    # Extract content (currently we support PAGE content best here)
    if module.type and str(module.type) == "ModuleTypes.PAGE":
        moodle.extract_page(module)
    elif module.modname == "page":
        moodle.extract_page(module)
    else:
        # As a fallback, try to download the first content file
        if module.contents and module.contents[0].fileurl:
            caller = APICaller(url=module.contents[0].fileurl, params=moodle.download_params)
            raw = caller.getText()
            module.text = raw

    raw_text = module.text or ""
    norm_text = normalize_text_for_rag(raw_text)

    def _stats(label: str, t: str):
        triple_nl = t.count("\n\n\n")
        print(
            f"{label}: chars={len(t):,} tokens≈{count_tokens(t):,} "
            f"blank_lines_runs(>=3_newlines)={'yes' if triple_nl else 'no'}"
        )

    _stats("RAW", raw_text)
    _stats("NORM", norm_text)

    chunks = chunk_text_hierarchical(
        raw_text,
        chunk_size_tokens=args.chunk_size,
        chunk_overlap_tokens=args.chunk_overlap,
        normalize=True,
    )

    print("\n" + "-" * 80)
    print(f"Chunks: {len(chunks)} (showing first {min(args.max_chunks, len(chunks))})")
    print("-" * 80)
    for i, ch in enumerate(chunks[: args.max_chunks]):
        print(f"\n[chunk {i+1}/{len(chunks)}] tokens={count_tokens(ch)}")
        print(ch)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

