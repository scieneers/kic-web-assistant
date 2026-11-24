#!/usr/bin/env python3
"""Backfill the top-level `modname`/`h5p_content_type` fields from metadata_json.

Documents ingested before these index fields existed carry the values only
inside the metadata_json blob. This script promotes them to the new filterable
top-level fields WITHOUT re-embedding (merge-only; the dense vector and all
other fields stay untouched).

Run AFTER a schema update (scripts/vectordb/update_index_schema.py) and AFTER
an ingest run — the ingest already populates the fields on every document it
re-writes, so this only needs to touch the untouched rest. NEVER run in
parallel with an ingestion run (the merge would race deletions).

Deliberately uses merge (not merge_or_upload): a document deleted between the
scan and the merge yields a per-doc 404 and is skipped — merge_or_upload would
resurrect a zombie skeleton doc invisible to all source-scoped cleanups.

Usage:
    uv run python scripts/vectordb/backfill_type_fields.py [--index NAME] [--dry-run]
"""

import argparse
import json

from src.env import env
from src.vectordb.azure_search import VectorDBAzureSearch

# Only Moodle docs carry modname; restrict the scan to rows that still miss it.
SCAN_FILTER = "source eq 'Moodle' and modname eq null"


def main():
    parser = argparse.ArgumentParser(description="Backfill modname/h5p_content_type from metadata_json (merge-only).")
    parser.add_argument("--index", default=None, help=f"Index name (default: from .env → {env.AZURE_SEARCH_INDEX})")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report, write nothing.")
    args = parser.parse_args()

    db = VectorDBAzureSearch()
    index_name = args.index or env.AZURE_SEARCH_INDEX

    print(f"Scanning '{index_name}' with filter: {SCAN_FILTER}")
    # Materialize the full result BEFORE writing — merging while the search
    # iterator is still paging would shift skip-based pagination (same reason
    # delete_by_filter collects keys first).
    rows = list(
        db._client(index_name).search(
            search_text="*",
            filter=SCAN_FILTER,
            select=["id", "metadata_json"],
        )
    )
    print(f"Found {len(rows)} documents without modname.")

    updates: list[dict] = []
    skipped_no_metadata = 0
    for row in rows:
        try:
            md = json.loads(row.get("metadata_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            md = {}
        modname = md.get("modname")
        if not modname:
            # Pre-dates the loader version that wrote modname into metadata —
            # cannot be backfilled, needs a real re-ingest instead.
            skipped_no_metadata += 1
            continue
        update = {"id": row["id"], "modname": modname}
        if md.get("h5p_content_type"):
            update["h5p_content_type"] = md["h5p_content_type"]
        updates.append(update)

    print(f"Backfillable: {len(updates)} | not backfillable (no modname in metadata_json): {skipped_no_metadata}")

    if args.dry_run:
        print("Dry run — nothing written.")
        return

    merged = db.merge_documents(index_name, updates)
    print(f"Merged fields into {merged} documents.")
    if skipped_no_metadata:
        print(
            f"NOTE: {skipped_no_metadata} documents lack modname in metadata_json and need a re-ingest "
            "(they pre-date the metadata field)."
        )


if __name__ == "__main__":
    main()
