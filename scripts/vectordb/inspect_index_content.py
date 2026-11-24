#!/usr/bin/env python3
"""Zeigt den tatsächlich eingebetteten Text pro Modul im Index.

Hilft zu prüfen ob Content-Extraktion funktioniert: leere Texte,
sehr kurze Chunks oder HTML-Reste deuten auf Extraktionsprobleme hin.

Beispiele:
    # Alle Kurse im Test-Index
    uv run python scripts/inspect_index_content.py --index aitestlaurien

    # Bestimmte Kurse
    uv run python scripts/inspect_index_content.py --index aitestlaurien --course-ids 123 456

    # Nur Module mit verdächtig wenig Inhalt (< 200 Zeichen)
    uv run python scripts/inspect_index_content.py --index aitestlaurien --short-only
"""

import argparse
import json
import sys
import textwrap

from src.env import env
from src.vectordb.azure_search import VectorDBAzureSearch

SNIPPET_LEN = 200
SHORT_THRESHOLD = 200  # Zeichen gesamt pro Modul


def fetch_chunks_for_course(client, course_id: int) -> list[dict]:
    results = client.search(
        search_text="*",
        filter=f"source eq 'Moodle' and course_id eq {course_id}",
        select=["module_id", "type", "fullname", "url", "text", "metadata_json"],
        top=1000,
    )
    return list(results)


def main():
    parser = argparse.ArgumentParser(description="Zeigt eingebetteten Text pro Modul im Index")
    parser.add_argument("--index", default=env.AZURE_SEARCH_INDEX)
    parser.add_argument("--course-ids", nargs="+", type=int, metavar="ID")
    parser.add_argument("--short-only", action="store_true",
                        help=f"Nur Module mit weniger als {SHORT_THRESHOLD} Zeichen Gesamttext")
    parser.add_argument("--snippet", type=int, default=SNIPPET_LEN,
                        help=f"Zeichenanzahl für Textauszug (default: {SNIPPET_LEN})")
    args = parser.parse_args()

    db = VectorDBAzureSearch()
    client = db._client(args.index)

    if args.course_ids:
        course_ids = args.course_ids
    else:
        results = client.search(
            search_text="*",
            filter="source eq 'Moodle' and module_id eq null and course_id ne null",
            select=["course_id"],
            top=1000,
        )
        course_ids = sorted({int(r["course_id"]) for r in results if r.get("course_id")})
        if not course_ids:
            print(f"Keine Moodle-Kurse in Index '{args.index}' gefunden.")
            sys.exit(0)

    print(f"Index: {args.index}")
    print(f"Kurse: {course_ids}")
    if args.short_only:
        print(f"Filter: nur Module mit < {SHORT_THRESHOLD} Zeichen")

    total_modules = 0
    empty_modules = 0
    short_modules = 0

    for course_id in course_ids:
        chunks = fetch_chunks_for_course(client, course_id)
        if not chunks:
            print(f"\nKurs {course_id}: keine Chunks im Index")
            continue

        # Gruppieren nach module_id (None = Kurs-Summary)
        by_module: dict = {}
        for chunk in chunks:
            key = chunk.get("module_id")
            by_module.setdefault(key, []).append(chunk)

        course_summary = by_module.pop(None, [])
        course_name = course_summary[0].get("fullname", "?") if course_summary else "?"
        print(f"\n{'=' * 70}")
        print(f"Kurs {course_id}: {course_name}")
        print(f"  {len(by_module)} Module, {len(course_summary)} Kurs-Summary-Chunks")

        # Kurs-Summary
        if course_summary:
            total_text = " ".join(c.get("text") or "" for c in course_summary)
            _print_entry("KURS-SUMMARY", None, None, course_name, total_text, args, course_summary)

        # Module
        for module_id, mod_chunks in sorted(by_module.items()):
            total_modules += 1
            fullname = mod_chunks[0].get("fullname") or "?"
            mod_type = mod_chunks[0].get("type") or "?"
            url = mod_chunks[0].get("url") or ""
            total_text = " ".join(c.get("text") or "" for c in mod_chunks)

            char_count = len(total_text.strip())
            if char_count == 0:
                empty_modules += 1
            elif char_count < SHORT_THRESHOLD:
                short_modules += 1

            if args.short_only and char_count >= SHORT_THRESHOLD:
                continue

            _print_entry(mod_type, module_id, url, fullname, total_text, args, mod_chunks)

    print(f"\n{'=' * 70}")
    print("Zusammenfassung:")
    print(f"  Module gesamt:         {total_modules}")
    print(f"  Leer (0 Zeichen):      {empty_modules}")
    print(f"  Kurz (< {SHORT_THRESHOLD} Zeichen):  {short_modules}")


def _parse_modname(chunks: list[dict]) -> tuple[str, str]:
    """Extrahiert modname und h5p_content_type aus metadata_json (erstes Chunk)."""
    for chunk in chunks:
        raw = chunk.get("metadata_json")
        if not raw:
            continue
        try:
            md = json.loads(raw)
            modname = md.get("modname", "")
            h5p = md.get("h5p_content_type", "")
            return modname, h5p
        except Exception:
            pass
    return "", ""


def _print_entry(mod_type, module_id, url, fullname, total_text, args, chunks=None):
    stripped = total_text.strip()
    char_count = len(stripped)
    flag = " ⚠ LEER" if char_count == 0 else (f" ⚠ KURZ ({char_count} Zeichen)" if char_count < SHORT_THRESHOLD else "")

    modname, h5p = _parse_modname(chunks or [])
    type_label = modname or mod_type
    if h5p:
        type_label += f" ({h5p})"

    id_str = f"module_id={module_id}" if module_id is not None else ""
    print(f"\n  [{type_label}] {fullname}{flag}")
    if id_str:
        print(f"    {id_str}  {url}")
    if char_count == 0:
        print("    <kein Text>")
    else:
        snippet = stripped[:args.snippet].replace("\n", " ")
        wrapped = textwrap.fill(snippet, width=66, initial_indent="    ", subsequent_indent="    ")
        print(wrapped)
        if char_count > args.snippet:
            print(f"    ... ({char_count} Zeichen total)")


if __name__ == "__main__":
    main()
