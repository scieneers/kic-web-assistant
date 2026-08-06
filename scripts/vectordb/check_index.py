"""
Index-Übersicht: Dokumente und Token-Schätzung pro Source und Type.
Token-Schätzung: len(text) / 4 (grobe Annäherung, ~4 Zeichen/Token).
Ausführen: uv run python check_index.py
"""
from collections import defaultdict

from src.env import env
from src.vectordb.azure_search import VectorDBAzureSearch

db = VectorDBAzureSearch()
index = env.AZURE_SEARCH_INDEX
client = db._client(index)
print(f"Index: {index}\n")

# Facets: server-side counts in a single request, no document content needed
results = client.search(
    search_text="*",
    facets=["source,count:20", "type,count:50"],
    top=0,
    include_total_count=True,
)
total_docs = results.get_count()
facets = results.get_facets() or {}

# Full scan: text + source_doc_key only (no vectors)
print("Lese Texte für Token- und Dokument-Schätzung...")
char_counts: dict[str, int] = defaultdict(int)
doc_keys: dict[str, set] = defaultdict(set)
for doc in client.search(search_text="*", select=["source", "text", "source_doc_key", "url"], top=1000):
    source = doc.get("source") or "unknown"
    char_counts[source] += len(doc.get("text") or "")
    key = doc.get("source_doc_key") or doc.get("url")
    if key:
        doc_keys[source].add(key)
total_tokens = sum(char_counts.values()) // 4
total_docs_unique = sum(len(v) for v in doc_keys.values())

print(f"Gesamt: {total_docs_unique:,} Dokumente  {total_docs:,} Chunks  ~{total_tokens:,} Tokens\n")

source_facets = sorted(facets.get("source", []), key=lambda f: f["value"])
print(f"{'Source':<15} {'Docs':>8}  {'Chunks':>8}  {'~Tokens':>10}")
print("-" * 48)
for f in source_facets:
    src = f["value"]
    tokens = char_counts.get(src, 0) // 4
    docs = len(doc_keys.get(src, set()))
    print(f"{src:<15} {docs:>8,}  {f['count']:>8,}  {tokens:>10,}")

type_facets = sorted(facets.get("type", []), key=lambda f: f["value"])
print(f"\n{'Type':<25} {'Chunks':>8}")
print("-" * 35)
for f in type_facets:
    print(f"{f['value']:<25} {f['count']:>8,}")


# Waisen-Module: module hat course_id, aber kein eigener Kurs-Record
courses, modules = db.get_course_module_records()
known_course_ids = {r.payload["course_id"] for r in courses}
orphan_modules = [r for r in modules if r.payload["course_id"] not in known_course_ids]

if orphan_modules:
    print(f"\nModule ohne Kurs-Record ({len(orphan_modules)}):")
    for m in orphan_modules:
        print(f"  course_id={m.payload['course_id']}  module_id={m.payload['module_id']}  name={m.payload['fullname']}")
else:
    print("\nKeine Waisen-Module gefunden.")