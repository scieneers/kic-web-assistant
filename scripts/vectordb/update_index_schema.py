#!/usr/bin/env python3
"""Apply the current index schema to an existing Azure AI Search index.

Updates fields and semantic configuration without deleting any documents.
Safe to run on a live index — uses create_or_update_index internally.

Usage:
    uv run python scripts/update_index_schema.py
    uv run python scripts/update_index_schema.py --index my-other-index
"""

import argparse

from src.env import env
from src.vectordb.azure_search import VectorDBAzureSearch

VECTOR_SIZE = 3072


def main():
    parser = argparse.ArgumentParser(description="Update Azure AI Search index schema (non-destructive).")
    parser.add_argument("--index", default=None, help=f"Index name (default: from .env → {env.AZURE_SEARCH_INDEX})")
    args = parser.parse_args()

    index_name = args.index or env.AZURE_SEARCH_INDEX

    print(f"Updating schema for index '{index_name}' (documents are not affected)...")
    db = VectorDBAzureSearch()
    db.create_index(index_name, VECTOR_SIZE)
    print("Done.")


if __name__ == "__main__":
    main()
