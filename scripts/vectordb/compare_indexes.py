"""
Vergleich zweier Azure AI Search Indizes.

Usage:
  uv run python scripts/vectordb/compare_indexes.py
  uv run python scripts/vectordb/compare_indexes.py --a aichat --b kic-content
"""
import argparse
import hashlib
from collections import Counter, defaultdict

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


# Bucket boundaries (chars) for the chunk-size histogram.
_SIZE_BUCKETS = [200, 500, 1000, 2000, 4000]


def _size_bucket(n: int) -> str:
    for b in _SIZE_BUCKETS:
        if n < b:
            lo = _SIZE_BUCKETS[_SIZE_BUCKETS.index(b) - 1] if _SIZE_BUCKETS.index(b) > 0 else 0
            return f"{lo}-{b}"
    return f"{_SIZE_BUCKETS[-1]}+"


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

    # Chunk-Ebene: pro Quelle, pro Dokument-Key
    doc_chunk_count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    doc_chars: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    size_hist: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Exakte Text-Duplikate: pro (source, doc_key) ein Counter über Text-Hashes.
    # Zählt Chunks mit identischem Text im selben Dokument (Hinweis auf doppeltes
    # Ingestion, nicht auf gewachsenen Inhalt).
    dup_hashes: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

    # Moodle-spezifisch: (course_id, module_id) → fullname
    moodle_modules: dict[tuple, str] = {}   # (course_id, module_id) → fullname
    moodle_courses: dict[int, str] = {}     # course_id → fullname
    moodle_module_chunks: dict[tuple, int] = defaultdict(int)  # (course_id, module_id) → chunk count
    moodle_module_chars: dict[tuple, int] = defaultdict(int)   # (course_id, module_id) → total chars

    for doc in client.search(
        "*",
        select=["source", "text", "url", "source_doc_key", "type",
                "course_id", "module_id", "fullname"],
    ):
        src = doc.get("source") or "unknown"
        text = doc.get("text") or ""
        n_chars = len(text)
        chars[src] += n_chars
        if not text.strip():
            empty[src] += 1
        size_hist[src][_size_bucket(n_chars)] += 1
        key = _doc_key(doc)
        if key:
            urls[src].add(key)
            doc_chunk_count[src][key] += 1
            doc_chars[src][key] += n_chars
            if text.strip():
                text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                dup_hashes[src][key][text_hash] += 1

        if src == "Moodle":
            cid = doc.get("course_id")
            mid = doc.get("module_id")
            fname = doc.get("fullname") or ""
            if cid and mid:
                moodle_modules[(cid, mid)] = fname
                moodle_module_chunks[(cid, mid)] += 1
                moodle_module_chars[(cid, mid)] += n_chars
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
        "moodle_module_chunks": dict(moodle_module_chunks),
        "moodle_module_chars": dict(moodle_module_chars),
        "doc_chunk_count": {s: dict(v) for s, v in doc_chunk_count.items()},
        "doc_chars": {s: dict(v) for s, v in doc_chars.items()},
        "size_hist": {s: dict(v) for s, v in size_hist.items()},
        "dup_hashes": {s: {k: dict(c) for k, c in v.items()} for s, v in dup_hashes.items()},
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

    # ── Module in beiden, aber unterschiedliche Chunk-Anzahl: liegt es an mehr
    #    Inhalt oder an anderer Chunk-Größe? ─────────────────────────────────
    mca, mcb = a["moodle_module_chunks"], b["moodle_module_chunks"]
    cha, chb = a["moodle_module_chars"], b["moodle_module_chars"]
    module_diffs = []
    for key in both_m:
        na, nb = mca.get(key, 0), mcb.get(key, 0)
        if na == nb:
            continue
        ca_chars, cb_chars = cha.get(key, 0), chb.get(key, 0)
        avg_a = ca_chars / na if na else 0
        avg_b = cb_chars / nb if nb else 0
        content_delta_pct = (cb_chars - ca_chars) / ca_chars if ca_chars else 0
        avg_delta_pct = (avg_b - avg_a) / avg_a if avg_a else 0
        if abs(content_delta_pct) < 0.10:
            cause = "Chunking"
        elif abs(avg_delta_pct) < 0.10:
            cause = "Inhalt"
        else:
            cause = "beides"
        module_diffs.append((abs(nb - na), key, na, nb, avg_a, avg_b, content_delta_pct, cause))
    module_diffs.sort(key=lambda x: -x[0])

    print(f"\n  Module in beiden Indizes, aber mit unterschiedlicher Chunk-Anzahl:")
    print(f"    {len(module_diffs)} von {len(both_m)} gemeinsamen Modulen betroffen")
    if module_diffs:
        by_cause = defaultdict(int)
        for *_rest, cause in module_diffs:
            by_cause[cause] += 1
        print(f"    Ursache (Δ Chunks): " + ", ".join(f"{c}={n}" for c, n in sorted(by_cause.items())))
        print(
            f"\n    {'Chunks ' + a['name']:>10}  {'Chunks ' + b['name']:>10}"
            f"  {'ØGröße ' + a['name']:>10}  {'ØGröße ' + b['name']:>10}"
            f"  {'ΔInhalt':>8}  {'Ursache':<10}  Modul"
        )
        _hr("-")
        for _, (cid, mid), na, nb, avg_a, avg_b, content_delta_pct, cause in module_diffs[:15]:
            fname = b["moodle_modules"].get((cid, mid)) or a["moodle_modules"].get((cid, mid)) or ""
            print(
                f"    {na:>10,}  {nb:>10,}  {avg_a:>10,.0f}  {avg_b:>10,.0f}"
                f"  {content_delta_pct:>+7.0%}  {cause:<10}  course_id={cid} module_id={mid}  {fname[:30]}"
            )
        if len(module_diffs) > 15:
            print(f"    … und {len(module_diffs) - 15} weitere Module")
        print(
            "\n    Legende: 'Chunking' = Textmenge ~gleich, andere Chunk-Größe/Splitting;"
            " 'Inhalt' = Textmenge hat sich geändert, ØGröße ~gleich; 'beides' = beides.\n"
            "    Achtung: exakte Text-Duplikate (dieselben Chunks 2x indiziert) erzeugen"
            " dasselbe Bild wie 'Inhalt' (ØGröße gleich, Textmenge verdoppelt) — siehe"
            " DUPLIKAT-CHECK weiter unten, um beides zu unterscheiden."
        )


def _chunk_size_histogram(a: dict, b: dict):
    _header("CHUNK-GRÖSSEN-VERTEILUNG (Zeichen/Chunk)")
    all_sources = sorted(set(a["size_hist"]) | set(b["size_hist"]))
    bucket_labels = [f"0-{_SIZE_BUCKETS[0]}"] + [
        f"{_SIZE_BUCKETS[i]}-{_SIZE_BUCKETS[i + 1]}" for i in range(len(_SIZE_BUCKETS) - 1)
    ] + [f"{_SIZE_BUCKETS[-1]}+"]

    for src in all_sources:
        ha = a["size_hist"].get(src, {})
        hb = b["size_hist"].get(src, {})
        if not ha and not hb:
            continue
        print(f"\n  [{src}]")
        print(f"    {'Bucket (Zeichen)':<20} {a['name']:>16}  {b['name']:>16}")
        _hr("-")
        for bucket in bucket_labels:
            va, vb = ha.get(bucket, 0), hb.get(bucket, 0)
            if va == 0 and vb == 0:
                continue
            delta = vb - va
            sign = "+" if delta >= 0 else ""
            print(f"    {bucket:<20} {va:>16,}  {vb:>16,}  ({sign}{delta:+,})")


def _chunks_per_doc_distribution(a: dict, b: dict):
    _header("CHUNKS-PRO-DOKUMENT-VERTEILUNG")
    bucket_edges = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 20), (21, None)]

    def _bucket_label(lo, hi):
        return f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi else f"{lo}+")

    def _histogram(doc_chunk_count: dict) -> dict:
        hist = defaultdict(int)
        for count in doc_chunk_count.values():
            for lo, hi in bucket_edges:
                if count >= lo and (hi is None or count <= hi):
                    hist[_bucket_label(lo, hi)] += 1
                    break
        return hist

    all_sources = sorted(set(a["doc_chunk_count"]) | set(b["doc_chunk_count"]))
    for src in all_sources:
        cca = a["doc_chunk_count"].get(src, {})
        ccb = b["doc_chunk_count"].get(src, {})
        if not cca and not ccb:
            continue
        ha = _histogram(cca)
        hb = _histogram(ccb)
        avg_a = sum(cca.values()) / len(cca) if cca else 0
        avg_b = sum(ccb.values()) / len(ccb) if ccb else 0
        print(f"\n  [{src}]  (Ø Chunks/Dok: {a['name']}={avg_a:.1f} {b['name']}={avg_b:.1f})")
        print(f"    {'Chunks/Dok':<20} {a['name']:>16}  {b['name']:>16}")
        _hr("-")
        for lo, hi in bucket_edges:
            label = _bucket_label(lo, hi)
            va, vb = ha.get(label, 0), hb.get(label, 0)
            if va == 0 and vb == 0:
                continue
            print(f"    {label:<20} {va:>16,}  {vb:>16,}")


def _doc_level_diff(a: dict, b: dict, top_n: int = 15):
    _header("GRÖSSTE UNTERSCHIEDE AUF DOKUMENT-EBENE (gleiches Dokument, andere Chunk-Zahl)")
    all_sources = sorted(set(a["doc_chunk_count"]) | set(b["doc_chunk_count"]))
    rows = []
    for src in all_sources:
        cca = a["doc_chunk_count"].get(src, {})
        ccb = b["doc_chunk_count"].get(src, {})
        cha = a["doc_chars"].get(src, {})
        chb = b["doc_chars"].get(src, {})
        common_keys = set(cca) & set(ccb)
        for key in common_keys:
            na, nb = cca[key], ccb[key]
            if na == nb:
                continue
            avg_a = cha.get(key, 0) / na if na else 0
            avg_b = chb.get(key, 0) / nb if nb else 0
            rows.append((abs(nb - na), src, key, na, nb, avg_a, avg_b))

    if not rows:
        print("\n  Keine gemeinsamen Dokumente mit abweichender Chunk-Zahl gefunden.")
        return

    rows.sort(key=lambda r: -r[0])
    print(f"\n  {'Quelle':<12} {'Chunks '+a['name']:>12}  {'Chunks '+b['name']:>12}"
          f"  {'Ø Zeichen '+a['name']:>14}  {'Ø Zeichen '+b['name']:>14}  Dokument")
    _hr("-")
    for _, src, key, na, nb, avg_a, avg_b in rows[:top_n]:
        short_key = key if len(key) <= 60 else key[:57] + "…"
        print(f"  {src:<12} {na:>12,}  {nb:>12,}  {avg_a:>14,.0f}  {avg_b:>14,.0f}  {short_key}")

    if len(rows) > top_n:
        print(f"\n  … und {len(rows) - top_n} weitere Dokumente mit abweichender Chunk-Zahl.")

    total_extra_chunks = sum(nb - na for _, _, _, na, nb, _, _ in rows)
    sign = "+" if total_extra_chunks >= 0 else ""
    print(f"\n  Netto-Effekt auf gemeinsame Dokumente: {sign}{total_extra_chunks:,} Chunks"
          f" ({a['name']} → {b['name']})")


def _duplicate_check(a: dict, b: dict, top_n: int = 15):
    _header("DUPLIKAT-CHECK (exakte Text-Duplikate im selben Dokument)")
    print(
        "  Zählt Chunks, deren Text im selben Dokument mehrfach identisch vorkommt.\n"
        "  Hoher Wert ⇒ vermutlich doppeltes Ingestion, nicht mehr Inhalt."
    )

    for idx in (a, b):
        dup_hashes = idx["dup_hashes"]
        all_sources = sorted(dup_hashes)
        total_dupes = 0
        total_chunks_scanned = 0
        rows = []
        for src in all_sources:
            src_dupes = 0
            src_chunks = 0
            for key, counts in dup_hashes[src].items():
                src_chunks += sum(counts.values())
                extra = sum(c - 1 for c in counts.values() if c > 1)
                if extra:
                    src_dupes += extra
                    rows.append((extra, src, key, counts))
            total_dupes += src_dupes
            total_chunks_scanned += src_chunks

        print(f"\n  [{idx['name']}]")
        pct = f"{total_dupes / total_chunks_scanned * 100:.1f}%" if total_chunks_scanned else "–"
        print(f"    Duplikat-Chunks gesamt: {total_dupes:,} von {total_chunks_scanned:,} ({pct})")

        by_source = defaultdict(int)
        for extra, src, _key, _counts in rows:
            by_source[src] += extra
        if by_source:
            print("    Nach Quelle: " + ", ".join(f"{s}={n:,}" for s, n in sorted(by_source.items(), key=lambda x: -x[1])))

        rows.sort(key=lambda r: -r[0])
        if rows:
            print(f"\n    Top Dokumente mit den meisten Duplikat-Chunks ({idx['name']}):")
            for extra, src, key, counts in rows[:top_n]:
                max_repeat = max(counts.values())
                short_key = key if len(key) <= 55 else key[:52] + "…"
                print(f"      +{extra:<6,} Duplikate  (max {max_repeat}x derselbe Chunk)  [{src}] {short_key}")
            if len(rows) > top_n:
                print(f"      … und {len(rows) - top_n} weitere Dokumente mit Duplikaten")


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
    _duplicate_check(a, b)
    _quality(a, b)
    _chunk_size_histogram(a, b)
    _chunks_per_doc_distribution(a, b)
    _doc_level_diff(a, b)
    print()


if __name__ == "__main__":
    main()
