"""
Vergleich zweier Azure AI Search Indizes.

Usage:
  uv run python scripts/vectordb/compare_indexes.py
  uv run python scripts/vectordb/compare_indexes.py --a aichat --b kic-content
"""
import argparse
from collections import defaultdict

from src.vectordb.azure_search import VectorDBAzureSearch

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_A = "aichat"
DEFAULT_B = "kic-content"

COLS = 80


# ── Helpers ──────────────────────────────────────────────────────────────────
def _hr(char="-"):
    print(char * COLS)


def _header(title: str):
    print()
    _hr("=")
    print(f"  {title}")
    _hr("=")


def _doc_key(doc: dict) -> str | None:
    return doc.get("source_doc_key") or doc.get("url")


def _scan(db: VectorDBAzureSearch, name: str) -> dict:
    client = db._client(name)

    # Server-side facets (single request)
    results = client.search(
        "*",
        facets=["source,count:20", "type,count:50"],
        top=0,
        include_total_count=True,
    )
    total_chunks = results.get_count()
    facets = results.get_facets() or {}

    # Full scan — SDK iterator paginates automatically
    print(f"  Scanne '{name}' ({total_chunks:,} Chunks)…", flush=True)
    chars: dict[str, int] = defaultdict(int)
    empty: dict[str, int] = defaultdict(int)
    urls: dict[str, set] = defaultdict(set)  # source → unique doc keys

    # Moodle-spezifisch: (course_id, module_id) → fullname
    moodle_modules: dict[tuple, str] = {}   # (course_id, module_id) → fullname
    moodle_courses: dict[int, str] = {}     # course_id → fullname

    for doc in client.search(
        "*",
        select=["source", "text", "url", "source_doc_key", "type",
                "course_id", "module_id", "fullname"],
    ):
        src = doc.get("source") or "unknown"
        text = doc.get("text") or ""
        chars[src] += len(text)
        if not text.strip():
            empty[src] += 1
        key = _doc_key(doc)
        if key:
            urls[src].add(key)

        if src == "Moodle":
            cid = doc.get("course_id")
            mid = doc.get("module_id")
            fname = doc.get("fullname") or ""
            if cid and mid:
                moodle_modules[(cid, mid)] = fname
            elif cid and not mid:
                moodle_courses[cid] = fname

    return {
        "name": name,
        "total_chunks": total_chunks,
        "chars": dict(chars),
        "empty": dict(empty),
        "urls": dict(urls),
        "source_facets": {f["value"]: f["count"] for f in facets.get("source", [])},
        "type_facets": {f["value"]: f["count"] for f in facets.get("type", [])},
        "moodle_modules": moodle_modules,
        "moodle_courses": moodle_courses,
    }


def _tokens(chars: int) -> int:
    return chars // 4


def _pct(a: int, b: int) -> str:
    if b == 0:
        return "  –"
    return f"{a / b * 100:5.1f}%"


# ── Report sections ───────────────────────────────────────────────────────────
def _overview(a: dict, b: dict):
    _header("ÜBERSICHT")
    na, nb = a["name"], b["name"]
    total_docs_a = sum(len(v) for v in a["urls"].values())
    total_docs_b = sum(len(v) for v in b["urls"].values())
    total_chars_a = sum(a["chars"].values())
    total_chars_b = sum(b["chars"].values())
    total_empty_a = sum(a["empty"].values())
    total_empty_b = sum(b["empty"].values())

    print(f"{'Metrik':<28} {na:>16}  {nb:>16}  {'Δ':>10}")
    _hr()
    rows = [
        ("Chunks gesamt", a["total_chunks"], b["total_chunks"]),
        ("Dokumente (unique URLs)", total_docs_a, total_docs_b),
        ("~Tokens gesamt", _tokens(total_chars_a), _tokens(total_chars_b)),
        ("Leere Chunks", total_empty_a, total_empty_b),
    ]
    for label, va, vb in rows:
        delta = vb - va
        sign = "+" if delta >= 0 else ""
        print(f"{label:<28} {va:>16,}  {vb:>16,}  {sign}{delta:>+10,}")

    print(f"\n{'Quellen':<28} {na:>16}  {nb:>16}")
    _hr()
    all_sources = sorted(set(a["source_facets"]) | set(b["source_facets"]))
    for src in all_sources:
        va = a["source_facets"].get(src, 0)
        vb = b["source_facets"].get(src, 0)
        marker = "  ← nur hier" if vb == 0 else ("  ← NEU" if va == 0 else "")
        print(f"  {src:<26} {va:>16,}  {vb:>16,}{marker}")


def _per_source(a: dict, b: dict):
    _header("DETAILS PRO QUELLE")
    all_sources = sorted(set(a["source_facets"]) | set(b["source_facets"]))
    for src in all_sources:
        print(f"\n  [{src}]")
        chunks_a = a["source_facets"].get(src, 0)
        chunks_b = b["source_facets"].get(src, 0)
        docs_a = len(a["urls"].get(src, set()))
        docs_b = len(b["urls"].get(src, set()))
        tok_a = _tokens(a["chars"].get(src, 0))
        tok_b = _tokens(b["chars"].get(src, 0))
        empty_a = a["empty"].get(src, 0)
        empty_b = b["empty"].get(src, 0)

        print(f"    {'':24} {a['name']:>16}  {b['name']:>16}")
        _hr("-")
        for label, va, vb in [
            ("Chunks", chunks_a, chunks_b),
            ("Dokumente", docs_a, docs_b),
            ("~Tokens", tok_a, tok_b),
            ("Leere Chunks", empty_a, empty_b),
        ]:
            if va == 0 and vb == 0:
                continue
            delta = vb - va
            sign = "+" if delta >= 0 else ""
            pct = _pct(abs(delta), va) if va else "  –"
            print(f"    {label:<24} {va:>16,}  {vb:>16,}  {sign}{delta:>+8,}  ({pct})")

        # Chunks pro Dokument
        if docs_a > 0:
            cpd_a = chunks_a / docs_a
            cpd_b = chunks_b / docs_b if docs_b else 0
            print(f"    {'Chunks/Dok (Ø)':<24} {cpd_a:>16.1f}  {cpd_b:>16.1f}")


def _types(a: dict, b: dict):
    _header("CHUNK-TYPEN")
    all_types = sorted(set(a["type_facets"]) | set(b["type_facets"]))
    print(f"  {'Typ':<30} {a['name']:>16}  {b['name']:>16}  {'Δ':>10}")
    _hr()
    for t in all_types:
        va = a["type_facets"].get(t, 0)
        vb = b["type_facets"].get(t, 0)
        delta = vb - va
        sign = "+" if delta >= 0 else ""
        print(f"  {t:<30} {va:>16,}  {vb:>16,}  {sign}{delta:>+10,}")


def _overlap(a: dict, b: dict):
    _header("DOKUMENT-ÜBERSCHNEIDUNG (nach URL)")
    all_sources = sorted(set(a["urls"]) | set(b["urls"]))
    only_a_total, only_b_total, both_total = 0, 0, 0

    for src in all_sources:
        sa = a["urls"].get(src, set())
        sb = b["urls"].get(src, set())
        only_a = sa - sb
        only_b = sb - sa
        both = sa & sb
        only_a_total += len(only_a)
        only_b_total += len(only_b)
        both_total += len(both)
        print(f"\n  [{src}]")
        print(f"    In beiden:              {len(both):>6,}")
        print(f"    Nur in {a['name']:<12}  {len(only_a):>6,}")
        print(f"    Nur in {b['name']:<12}  {len(only_b):>6,}")

        if only_a and len(only_a) <= 10:
            print(f"    Beispiele nur in {a['name']}:")
            for u in sorted(only_a)[:5]:
                print(f"      {u}")
        if only_b and len(only_b) <= 10:
            print(f"    Beispiele nur in {b['name']}:")
            for u in sorted(only_b)[:5]:
                print(f"      {u}")

    print()
    _hr()
    print(f"  Gesamt – in beiden:     {both_total:>6,}")
    print(f"  Gesamt – nur {a['name']:<10}  {only_a_total:>6,}")
    print(f"  Gesamt – nur {b['name']:<10}  {only_b_total:>6,}")


def _moodle_detail(a: dict, b: dict):
    _header("MOODLE-VERGLEICH (nach course_id / module_id)")

    # ── Kurse ────────────────────────────────────────────────────────────────
    ca = set(a["moodle_courses"])
    cb = set(b["moodle_courses"])
    only_a_c = ca - cb
    only_b_c = cb - ca
    both_c = ca & cb

    print(f"\n  Kurse (type=Kurs)")
    print(f"    In beiden:              {len(both_c):>6,}")
    print(f"    Nur in {a['name']:<14}  {len(only_a_c):>6,}")
    print(f"    Nur in {b['name']:<14}  {len(only_b_c):>6,}")

    if only_a_c:
        print(f"\n    Nur in {a['name']} ({len(only_a_c)}):")
        for cid in sorted(only_a_c)[:15]:
            print(f"      course_id={cid:>6}  {a['moodle_courses'][cid]}")
        if len(only_a_c) > 15:
            print(f"      … und {len(only_a_c) - 15} weitere")

    if only_b_c:
        print(f"\n    Nur in {b['name']} ({len(only_b_c)}):")
        for cid in sorted(only_b_c)[:15]:
            print(f"      course_id={cid:>6}  {b['moodle_courses'][cid]}")
        if len(only_b_c) > 15:
            print(f"      … und {len(only_b_c) - 15} weitere")

    # ── Module ───────────────────────────────────────────────────────────────
    ma = set(a["moodle_modules"])
    mb = set(b["moodle_modules"])
    only_a_m = ma - mb
    only_b_m = mb - ma
    both_m = ma & mb

    print(f"\n  Module (type=module / EmptyModule)")
    print(f"    In beiden:              {len(both_m):>6,}")
    print(f"    Nur in {a['name']:<14}  {len(only_a_m):>6,}")
    print(f"    Nur in {b['name']:<14}  {len(only_b_m):>6,}")

    # Kursweise aufschlüsseln für die Unterschiede
    def _by_course(module_set, name_map):
        by_course: dict[int, list] = defaultdict(list)
        for cid, mid in module_set:
            by_course[cid].append((mid, name_map.get((cid, mid), "")))
        return by_course

    if only_a_m:
        print(f"\n    Nur in {a['name']} — Top-Kurse nach Modul-Anzahl:")
        by_c = _by_course(only_a_m, a["moodle_modules"])
        for cid, mods in sorted(by_c.items(), key=lambda x: -len(x[1]))[:10]:
            cname = a["moodle_courses"].get(cid) or b["moodle_courses"].get(cid) or "?"
            print(f"      course_id={cid:>6} ({cname[:40]})  → {len(mods)} Module")

    if only_b_m:
        print(f"\n    Nur in {b['name']} — Top-Kurse nach Modul-Anzahl:")
        by_c = _by_course(only_b_m, b["moodle_modules"])
        for cid, mods in sorted(by_c.items(), key=lambda x: -len(x[1]))[:10]:
            cname = b["moodle_courses"].get(cid) or a["moodle_courses"].get(cid) or "?"
            print(f"      course_id={cid:>6} ({cname[:40]})  → {len(mods)} Module")

    # ── Module in beiden, aber unterschiedlicher Chunk-Count nicht messbar hier
    # ── Kurse die in einem Index mehr Module haben ────────────────────────────
    print(f"\n  Kurse mit größten Modul-Unterschieden (in {b['name']} − {a['name']}):")
    counts_a: dict[int, int] = defaultdict(int)
    counts_b: dict[int, int] = defaultdict(int)
    for cid, _ in ma:
        counts_a[cid] += 1
    for cid, _ in mb:
        counts_b[cid] += 1
    all_course_ids = set(counts_a) | set(counts_b)
    diffs = []
    for cid in all_course_ids:
        ca_cnt, cb_cnt = counts_a.get(cid, 0), counts_b.get(cid, 0)
        if ca_cnt != cb_cnt:
            cname = (b["moodle_courses"].get(cid) or a["moodle_courses"].get(cid) or "?")
            diffs.append((cb_cnt - ca_cnt, cid, cname, ca_cnt, cb_cnt))
    diffs.sort(key=lambda x: -abs(x[0]))
    for delta, cid, cname, ca_cnt, cb_cnt in diffs[:15]:
        sign = "+" if delta >= 0 else ""
        print(f"    course_id={cid:>6}  {cname[:35]:<35}  {ca_cnt:>4} → {cb_cnt:>4}  ({sign}{delta})")


def _quality(a: dict, b: dict):
    _header("INHALTSQUALITÄT")
    print(f"  {'Quelle':<20} {'∅ Zeichen/Chunk':>18}  {'∅ Tokens/Chunk':>16}")
    print(f"  {'':20} {a['name']:>8}  {b['name']:>8}  {a['name']:>8}  {b['name']:>8}")
    _hr()

    all_sources = sorted(set(a["source_facets"]) | set(b["source_facets"]))
    for src in all_sources:
        ca, cb = a["source_facets"].get(src, 0), b["source_facets"].get(src, 0)
        chars_a = a["chars"].get(src, 0)
        chars_b = b["chars"].get(src, 0)
        avg_chars_a = chars_a / ca if ca else 0
        avg_chars_b = chars_b / cb if cb else 0
        avg_tok_a = avg_chars_a / 4
        avg_tok_b = avg_chars_b / 4
        print(
            f"  {src:<20} {avg_chars_a:>8.0f}  {avg_chars_b:>8.0f}"
            f"  {avg_tok_a:>8.0f}  {avg_tok_b:>8.0f}"
        )

    # Leere Chunks %
    print()
    print(f"  {'Quelle':<20} {'Leere Chunks %':>18}")
    print(f"  {'':20} {a['name']:>8}  {b['name']:>8}")
    _hr()
    for src in all_sources:
        ca, cb = a["source_facets"].get(src, 0), b["source_facets"].get(src, 0)
        ea = a["empty"].get(src, 0)
        eb = b["empty"].get(src, 0)
        pct_a = f"{ea / ca * 100:.1f}%" if ca else "–"
        pct_b = f"{eb / cb * 100:.1f}%" if cb else "–"
        print(f"  {src:<20} {pct_a:>8}  {pct_b:>8}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default=DEFAULT_A)
    parser.add_argument("--b", default=DEFAULT_B)
    args = parser.parse_args()

    print(f"Vergleiche '{args.a}' ↔ '{args.b}'\n")
    db = VectorDBAzureSearch()
    a = _scan(db, args.a)
    b = _scan(db, args.b)

    _overview(a, b)
    _per_source(a, b)
    _types(a, b)
    _overlap(a, b)
    _moodle_detail(a, b)
    _quality(a, b)
    print()


if __name__ == "__main__":
    main()
