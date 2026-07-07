#!/usr/bin/env python3
"""Delete documents from the Azure AI Search index to force a full re-embed.

Examples:
    # Delete all docs from one source (keeps other sources intact):
    uv run python scripts/reset_index.py --source MOODLE
    uv run python scripts/reset_index.py --source MOOCHUP
    uv run python scripts/reset_index.py --source Drupal

    # Wipe and recreate the entire index (nuclear option):
    uv run python scripts/reset_index.py --all
"""

import argparse
import sys

from src.vectordb.azure_search import VectorDBAzureSearch
from src.env import env


def main():
    parser = argparse.ArgumentParser(description="Reset Azure AI Search index or individual sources.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", metavar="SOURCE", help="Delete all docs for this source (e.g. Moodle, Moochup, Drupal)")
    group.add_argument("--all", action="store_true", help="Drop and recreate the entire index")
    parser.add_argument("--index", default=None, help=f"Index name (default: from .env → {env.AZURE_SEARCH_INDEX})")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    index_name = args.index or env.AZURE_SEARCH_INDEX

    if args.all:
        msg = f"Drop and recreate index '{index_name}' (ALL data will be lost)"
    else:
        msg = f"Delete all '{args.source}' documents from index '{index_name}'"

    if not args.yes:
        print(f"\n⚠️  {msg}")
        answer = input("   Type 'yes' to confirm: ").strip()
        if answer.lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    db = VectorDBAzureSearch()

    if args.all:
        print(f"Deleting index '{index_name}'...")
        db.index_client.delete_index(index_name)
        print("Index deleted. Re-creating schema...")
        db.create_index(index_name, 3072)
        print("Done. Index is empty and ready for a full ingest.")
    else:
        odata_filter = f"source eq '{args.source}'"
        print(f"Deleting all docs where {odata_filter} ...")
        deleted = db.delete_by_filter(index_name, odata_filter)
        print(f"Done. Deleted {deleted} documents.")
        print(f"Run: RUN_SOURCES={args.source.upper()} uv run python -m src.loaders.get_data")


if __name__ == "__main__":
    main()
