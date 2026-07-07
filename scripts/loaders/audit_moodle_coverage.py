#!/usr/bin/env python3
"""Vergleicht Moodle-Kursinhalt mit dem Index: welche Module fehlen?

Holt für die angegebenen Kurse alle Module aus der Moodle-API und prüft,
welche davon im Azure AI Search Index vorhanden sind.

Beispiele:
    # Alle Kurse im Index prüfen (holt course_ids aus dem Index)
    uv run python scripts/audit_moodle_coverage.py --index aitestlaurien

    # Bestimmte Kurse prüfen
    uv run python scripts/audit_moodle_coverage.py --index aitestlaurien --course-ids 123 456 789
"""

import argparse
import sys

from src.env import env
from src.vectordb.azure_search import VectorDBAzureSearch
from src.loaders.moodle import Moodle


def get_indexed_course_ids(db: VectorDBAzureSearch, index_name: str) -> list[int]:
    client = db._client(index_name)
    results = client.search(
        search_text="*",
        filter="source eq 'Moodle' and module_id eq null and course_id ne null",
        select=["course_id"],
        top=1000,
    )
    return sorted({int(r["course_id"]) for r in results if r.get("course_id") is not None})


def get_indexed_module_ids(db: VectorDBAzureSearch, index_name: str, course_id: int) -> set[int]:
    client = db._client(index_name)
    results = client.search(
        search_text="*",
        filter=f"source eq 'Moodle' and course_id eq {course_id} and module_id ne null",
        select=["module_id"],
        top=1000,
    )
    return {int(r["module_id"]) for r in results if r.get("module_id") is not None}


def main():
    parser = argparse.ArgumentParser(description="Moodle coverage audit vs. Azure AI Search index")
    parser.add_argument("--index", default=env.AZURE_SEARCH_INDEX, help="Index name")
    parser.add_argument("--course-ids", nargs="+", type=int, metavar="ID",
                        help="Moodle course IDs to check (default: alle im Index gefundenen)")
    args = parser.parse_args()

    db = VectorDBAzureSearch()
    moodle = Moodle()

    if args.course_ids:
        course_ids = args.course_ids
    else:
        print(f"Lade course_ids aus Index '{args.index}'...")
        course_ids = get_indexed_course_ids(db, args.index)
        if not course_ids:
            print("Keine Moodle-Kurse im Index gefunden.")
            sys.exit(0)

    print(f"\nIndex: {args.index}")
    print(f"Kurse zu prüfen: {course_ids}\n")
    print("=" * 70)

    total_moodle_modules = 0
    total_indexed_modules = 0
    total_missing = 0

    for course_id in course_ids:
        print(f"\nKurs {course_id}")
        print("-" * 50)

        # Moodle-Seite: alle sichtbaren Module
        try:
            topics = moodle.get_course_contents(course_id)
        except Exception as e:
            print(f"  FEHLER beim Laden aus Moodle: {e}")
            continue

        moodle_modules: dict[int, str] = {}
        for topic in topics:
            for module in getattr(topic, "modules", []):
                mod_id = getattr(module, "id", None)
                mod_name = getattr(module, "name", "?")
                mod_type = getattr(module, "modname", "?")
                if mod_id is not None and getattr(module, "visible", 1):
                    moodle_modules[int(mod_id)] = f"{mod_type}: {mod_name}"

        # Index-Seite: welche module_ids sind vorhanden?
        indexed_module_ids = get_indexed_module_ids(db, args.index, course_id)

        moodle_ids = set(moodle_modules.keys())
        missing = moodle_ids - indexed_module_ids
        extra = indexed_module_ids - moodle_ids  # im Index aber nicht in Moodle

        total_moodle_modules += len(moodle_ids)
        total_indexed_modules += len(indexed_module_ids)
        total_missing += len(missing)

        print(f"  Moodle Module:  {len(moodle_ids)}")
        print(f"  Im Index:       {len(indexed_module_ids)}")

        if missing:
            print(f"  FEHLT ({len(missing)}):")
            for mid in sorted(missing):
                print(f"    module_id={mid}  {moodle_modules.get(mid, '?')}")
        else:
            print("  Alle Module im Index vorhanden.")

        if extra:
            print(f"  Im Index aber nicht in Moodle ({len(extra)}) — evtl. geloescht:")
            for mid in sorted(extra):
                print(f"    module_id={mid}")

    print("\n" + "=" * 70)
    print("Zusammenfassung:")
    print(f"  Moodle Module gesamt: {total_moodle_modules}")
    print(f"  Im Index:             {total_indexed_modules}")
    print(f"  Fehlend:              {total_missing}")
    if total_moodle_modules > 0:
        coverage = (total_indexed_modules / total_moodle_modules) * 100
        print(f"  Abdeckung:            {coverage:.1f}%")


if __name__ == "__main__":
    main()
