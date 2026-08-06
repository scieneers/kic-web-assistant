#!/usr/bin/env python3
"""Populate a small test index with a handful of Moodle courses.

Useful for evaluating and testing the delta-update logic without touching
the production 'aichat' index or waiting for a full ingest run.

Usage:
    # Default: 3 Moodle courses into 'aitestlaurien'
    uv run python scripts/ingest_test_index.py

    # Custom course count or index name:
    uv run python scripts/ingest_test_index.py --index aitestlaurien --courses 5

    # Reset the test index first, then re-ingest (to test a clean run):
    uv run python scripts/ingest_test_index.py --reset

    # Only re-ingest without resetting (to test the delta / skip logic):
    uv run python scripts/ingest_test_index.py
"""

import argparse
import os
import sys

# Set env vars BEFORE importing src (env is a module-level singleton).
# These are read via os.getenv() inside Moodle.get_courses() and Fetch_Data.extract(),
# so setting them here is safe.

def _parse_args():
    parser = argparse.ArgumentParser(description="Ingest a small test index for update-logic evaluation")
    parser.add_argument(
        "--index",
        default="aitestlaurien",
        help="Azure AI Search index name to write into (default: aitestlaurien)",
    )
    parser.add_argument(
        "--courses",
        type=int,
        default=3,
        help="Number of Moodle courses to ingest (default: 3)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the test index before ingesting (full clean run)",
    )
    parser.add_argument(
        "--sources",
        default="MOODLE,MOOCHUP",
        help="Comma-separated sources to ingest (default: MOODLE). Options: MOODLE,MOOCHUP,DRUPAL",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    os.environ["MOODLE_COURSE_LIMIT"] = str(args.courses)
    os.environ["RUN_SOURCES"] = args.sources

    # Import after env vars are set so any module that reads os.getenv() at import time
    # already sees the right values.
    from src.vectordb.azure_search import VectorDBAzureSearch
    from src.loaders.get_data import Fetch_Data

    index_name = args.index

    if args.reset:
        print(f"Resetting index '{index_name}'...")
        db = VectorDBAzureSearch()
        try:
            db.index_client.delete_index(index_name)
            print("  Deleted existing index.")
        except Exception:
            print("  Index did not exist yet, skipping delete.")
        db.create_index(index_name, 3072)
        print("  Index recreated (empty).")

    print(f"\nStarting ingestion into '{index_name}'")
    print(f"  Courses: {args.courses}")
    print(f"  Sources: {args.sources}")
    print()

    result = Fetch_Data(index_name=index_name).extract()

    print("\nDone.")
    counts = result.get("counts", {})
    for key, value in counts.items():
        if value:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
