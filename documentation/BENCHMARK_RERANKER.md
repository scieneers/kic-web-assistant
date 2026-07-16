# Reranker-Benchmark: Ergebnisse & Bewertung

**Stand:** 2026-07-09 · **Lauf-Typ:** offizieller Entscheidungslauf (siehe [SCRIPTS.md](SCRIPTS.md), Abschnitt „Reranker Evaluation") · **Ergebnisdatei:** `results_fair.json`

> **Nachtrag:** Empfehlung 1 ist umgesetzt — `RERANKER_TYPE`-Default steht seit Commit `67da6f5` auf `azure_semantic`. Wo dieses Dokument den LLM-Reranker als „aktuellen Produktions-Default" bezeichnet, beschreibt es den Stand **zum Zeitpunkt des Laufs** (09.07.); dieses Dokument ist ein historisches Ergebnis-Snapshot und wird nicht fortgeschrieben.

```bash
uv run python -m evaluation.benchmark --pool-sizes 10,30 --runs 1 --cooldown 2 \
  --model Gemma4 --judge-model Azure-Fallback --output results_fair.json
```

Dieser Lauf ersetzt die bisherigen explorativen Zahlen aus [ANALYSE_FEAT_IMPROVEMENTS.md](ANALYSE_FEAT_IMPROVEMENTS.md) Kap. 4.10 (`results_queries_generated50.json`, `results_queries_generated_v1.json`) — er läuft mit dem gehärteten Harness (strict-mode LLM-Reranker, Rate-Limit-Cooldown), auf dem aktuellen `queries.jsonl` (68 Queries) und **mit dem echten Produktionsmodell** (`Gemma4` = `GEMMA4_31B`, GWDG-gehostet) statt mit `Azure-Fallback` (GPT-4o). Das Modell-Detail ist der wichtigste Unterschied zu den alten Läufen — dazu mehr in Kap. 4.

---

## 1. Kurzfassung

| Reranker | Pool=10 NDCG@5 | Pool=30 NDCG@5 | Latenz p50 | Kosten/1k € | Bewertung |
|---|---|---|---|---|---|
| **Azure Semantic** | **0.875** | **0.841** | 419 ms | 0,09 € | ✅ Bester Kompromiss aus Qualität, Latenz, Kosten |
| **BGE (v2-m3, large)** | 0.862 | 0.835 | 2.925 ms | 0,00 € | ✅ Fast gleichwertig, kostenfrei — aber Latenz bei Pool 30 explodiert (s. Kap. 4.4) |
| BGE (base, small) | 0.820 | 0.772 | 531 ms | 0,00 € | ⚠️ Schwächer als v2-m3, kein Vorteil |
| No Rerank (Baseline) | 0.771 | 0.744 | 0 ms | 0,00 € | — Referenzpunkt |
| **LLM Reranker (Gemma4, Produktion)** | 0.724 | 0.738 | 2.162 ms | 0,00 €* | ❌ Schlechter als gar kein Reranking, langsam, unzuverlässig (s. Kap. 4.1/4.3) |

\* 0,00 € nur weil das Produktionsmodell GWDG-flatrate-gehostet ist — kein Free-Lunch, s. Kap. 4.5.

**Kernaussage:** Der aktuelle Produktions-Reranker (LLM-basiert, Modell `Gemma4`) verschlechtert die Ranking-Qualität gegenüber gar keinem Reranking — und ist gleichzeitig die langsamste und am wenigsten zuverlässige Option (6 von 68 Anfragen bei Pool 30 durch Rate-Limiting fehlgeschlagen). **Azure Semantic** und **BGE (bge-reranker-v2-m3)** liegen beide klar vorn und nah beieinander; zwischen den beiden ist die Differenz bei dieser Stichprobengröße nicht als „statistisch gesichert" zu werten (Kap. 5). Eine Umstellung des Defaults von `RERANKER_TYPE=llm` weg ist auf Basis dieses Laufs gut begründet.

---

## 2. Versuchsaufbau

### 2.1 Was wird verglichen? Die Reranker kurz erklärt

| Reranker | Funktionsprinzip | Wo/Wie | Bemerkung |
|---|---|---|---|
| **No Rerank (Baseline)** | Kein Reranking — Chunks bleiben in der Reihenfolge der Hybrid-Suche (Vektor + Keyword, RRF-fusioniert) | — | Referenzpunkt: schlägt ein Reranker das nicht, bringt er nichts |
| **LLM Reranker** | Ein LLM bekommt Query + Kandidaten-Chunks vorgelegt und vergibt pro Chunk eine Relevanz-Note (1–10, LlamaIndex `LLMRerank`); danach wird neu sortiert | ruft ein Chat-Modell per Prompt auf (hier: `Gemma4`, GWDG) | **Aktueller Produktions-Default**; Qualität hängt stark vom verwendeten Modell ab (Kap. 4.1) |
| **Azure Semantic** | In Azure AI Search eingebauter Semantic Ranker — ein von Microsoft trainiertes Cross-Encoder-Modell, das direkt im Such-Call mitläuft und einen Reranker-Score (0–4) liefert | Teil des Such-Calls selbst (`query_type=semantic`), kein separater Modell-Aufruf nötig | Für den Benchmark ein zweiter Such-Call pro Query; in Produktion ersetzt er das normale Retrieval statt zusätzlich zu laufen |
| **BGE (bge-reranker-v2-m3)** | Lokales Open-Source-Cross-Encoder-Modell (BAAI, ~568 MB) — bewertet Query+Chunk gemeinsam in einem Transformer-Durchlauf | läuft lokal auf der CPU, kein externer API-Call | mehrsprachig trainiert (100+ Sprachen inkl. DE/EN), 0 € Betriebskosten |
| **BGE (bge-reranker-base)** | Kleineres Schwestermodell (~280 MB) desselben Ansatzes | läuft lokal auf der CPU | primär englisch-trainiert, schneller, aber schwächer (Kap. 3) |

**Der wichtigste konzeptuelle Unterschied:** Cross-Encoder (Azure Semantic, beide BGE-Modelle) sind eigens für die Aufgabe „Query+Dokument → Relevanz-Score" trainiert — kompakt, schnell, zweckgebunden. Der LLM-Reranker zweckentfremdet dagegen ein allgemeines Chat-Modell für dieselbe Aufgabe über Prompting — flexibler, aber Qualität und Kosten hängen direkt am gewählten Modell (genau das zeigt sich in Kap. 4.1: Der Modellwechsel von GPT-4o zu Gemma4 kippt das Ergebnis von „schlägt Baseline" zu „schlägt Baseline nicht").

### 2.2 Woher kommen die Fragen?

Der Datensatz (`evaluation/dataset/queries.jsonl`, **68 Queries**) setzt sich aus zwei Quellen zusammen:

1. **Kuratierter Kern** (`create_dataset.py::SAMPLE_QUERIES`, von Hand geschrieben) — Fragetypen, die sich nicht sinnvoll aus dem Index generieren lassen: Kursentdeckung, kurze/vage Suchbegriffe, umgangssprachlich/mit Tippfehlern, Plattform-/Orga-Fragen, Mehrsprachigkeit (ES/FR/IT/TR), sowie eine feste No-Answer-Kalibrierungsmenge.
2. **Index-generierte Fragen** (`generate_questions.py`) — ein LLM sampelt echte Chunks aus dem Such-Index (Drupal + Moodle) und formuliert dazu eine realistische Lernendenfrage, **inklusive der aktuellen, echten `course_id`/`module_id`** aus dem Index. Das vermeidet hartkodierte IDs, die nach jedem Re-Ingest veralten würden.

Fragetypen (`kind`) und ihre Häufigkeit in diesem Datensatz:

| kind | n | Beschreibung |
|---|---|---|
| negative | 10 | plausibel klingend, aber (voraussichtlich) nicht durch die Wissensbasis gedeckt — Kalibrierung für No-Answer/`min_score` |
| concept | 8 | klassische Konzepterklärung („Was ist X?") |
| drupal_level | 8 | unscoped Besucherfrage auf ki-campus.org, aus echtem Drupal-Chunk generiert |
| course_level | 8 | Frage *innerhalb* eines Kurses, gescoped mit `course_id` |
| module_level | 8 | Frage *innerhalb* eines Moduls, gescoped mit `course_id`+`module_id` |
| comparison | 7 | braucht Inhalt aus zwei verschiedenen Drupal-Dokumenten |
| discovery | 4 | „Welche Kurse gibt es zu X?" |
| platform | 4 | Orga-/Plattformfragen (Zertifikat, Kosten, Träger) |
| technical | 3 | technische Vertiefung |
| short | 3 | 1-2-Wort-Suchen |
| colloquial | 3 | umgangssprachlich/Tippfehler |
| domain | 2 | domänenspezifisch (Medizin, Schule) |

Sprachverteilung: **de=41, en=23**, sowie je **1** für es/fr/it/tr (bewusst unscoped, testet Cross-Lingual-Retrieval, aber statistisch nicht belastbar — s. Kap. 5).

### 2.3 Wie wird „relevant" definiert? (Ground Truth)

Für jede Query werden mit dem echten Retriever (Hybrid-Suche, gleicher Scope wie in Produktion) bis zu **30 Kandidaten-Chunks** geholt. Für jedes (Query, Chunk)-Paar bewertet ein LLM-Judge (`Azure-Fallback` = GPT-4o) die Relevanz auf einer dreistufigen Skala:

- **2** = hoch relevant (beantwortet die Frage direkt und vollständig)
- **1** = teilweise relevant
- **0** = nicht relevant

Das ist ein **einzelner automatisierter Judge**, kein menschliches Label und keine Mehrfachbewertung/Inter-Rater-Agreement — dazu mehr in Kap. 5. Über den aktuellen Datensatz verteilen sich die 1.840 Chunk-Urteile so: 741× relevance=0, 877× relevance=1, 222× relevance=2.

### 2.4 Was bedeutet eine Kennzahl? (Metriken erklärt)

Alle vier Ranking-Metriken beantworten dieselbe Grundfrage — „wie gut sind die Top-5-Ergebnisse, die ein Reranker zurückgibt?" — aber jede auf eine andere Art und mit einer anderen Schwäche. Alle liegen auf einer Skala von 0 (schlechtestmöglich) bis 1 (bestmöglich) und werden über alle Queries gemittelt.

**NDCG@5** (Normalized Discounted Cumulative Gain) — die im Bericht wichtigste Metrik, weil sie zwei Dinge gleichzeitig belohnt: *dass* relevante Chunks überhaupt in den Top 5 sind, und *wo genau* (weiter oben = mehr wert). Beispielrechnung: Ein Reranker liefert an Position 1–5 die Relevanzwerte `[2, 0, 1, 0, 0]` (Skala 0/1/2 aus Kap. 2.3). Jede Position wird mit `log2(Position+1)` abgeschwächt — Position 1 gar nicht, Position 5 am stärksten. Discounted Gain = 2/log₂(2) + 0/log₂(3) + 1/log₂(4) + 0 + 0 = 2,0 + 0,5 = **2,5**. Der bestmögliche Wert für dieselbe Query wäre, dieselben Chunks absteigend sortiert (`[2,1,0,0,0]`): 2,0 + 1/log₂(3) = 2,0 + 0,63 = **2,63**. NDCG@5 = 2,5 / 2,63 ≈ **0,95**. Ein gemessener Wert von 0,875 (Azure Semantic, Pool 10) heißt also: im Schnitt über alle Queries liegen die Top-5-Rankings bei rund 87,5 % dessen, was mit den *im Kandidaten-Pool tatsächlich vorhandenen* Chunks bestmöglich erreichbar gewesen wäre — 1,0 wäre eine perfekte Sortierung, 0,0 hieße „nichts Brauchbares oben". Zwei Besonderheiten, die für dieses Dokument wichtig sind: (1) Der Ideal-Wert wird gegen den **kompletten** Kandidaten-Pool berechnet, nicht nur gegen die zurückgegebenen Chunks — ein System, das nur 2 statt 5 Chunks liefert, wird dadurch nicht künstlich aufgewertet. (2) Hat eine Query **gar keinen** relevanten Chunk im Pool (z. B. ein echter No-Answer-Fall), ist der Ideal-Wert selbst 0 — NDCG@5 ist dann für **jedes** System automatisch 0,0, unabhängig davon, was es zurückgibt (relevant für Kap. 4.6).

**Recall@5** — von allen relevanten Chunks, die irgendwo im Vereinigungs-Pool existieren, welcher Anteil hat es in die Top 5 geschafft? Beispiel: Für eine Query gibt es insgesamt 8 als relevant gelabelte Chunks, der Reranker bringt 3 davon in seine Top 5 → Recall@5 = 3/8 = 0,375. Anders als NDCG bestraft Recall@5 es nicht, wenn ein Treffer auf Position 5 statt 1 steht — es fragt nur „drin oder nicht". Das erklärt auch, warum Recall@5 in Kap. 3.1 für alle Systeme beim Wechsel von Pool 10 zu Pool 30 spürbar sinkt: Bei größerem Pool gibt es oft mehr relevante Chunks insgesamt (größerer Nenner), aber weiterhin nur 5 Plätze oben — der *Anteil* sinkt, obwohl die Top-5-Auswahl selbst nicht schlechter wurde.

**MRR** (Mean Reciprocal Rank) — schaut nur auf den *ersten* relevanten Treffer und ignoriert alles danach. Steht er auf Position 1 → Beitrag 1,0; Position 2 → 0,5; Position 3 → 0,33 (`1/Position`); usw., gemittelt über alle Queries. Ein MRR von 0,926 (Azure Semantic) heißt: im Schnitt taucht der erste brauchbare Treffer schon fast auf Position 1 auf. Guter Indikator für „bekommt der Nutzer sofort etwas Brauchbares zu sehen", sagt aber nichts darüber, ob Positionen 2–5 auch gut sind.

**P@5** (Precision@5) — von den 5 zurückgegebenen Chunks, wie viele sind überhaupt relevant (relevance > 0, egal ob 1 oder 2, egal an welcher Position)? P@5 = 0,788 (Azure Semantic, Pool 10) heißt: im Schnitt sind knapp 4 von 5 zurückgegebenen Chunks irgendwie brauchbar, im Schnitt einer ist „Ausschuss". Die einfachste der vier Metriken — blind für Reihenfolge und dafür, wie viele relevante Chunks anderswo im Pool liegen geblieben sind.

**Latenz p50 / p95** — Median (typischer Fall) bzw. 95.-Perzentil (die langsamsten 5 % aller gemessenen Aufrufe) der Antwortzeit des Reranker-Calls selbst, in Millisekunden. p95 ist für die Nutzererfahrung oft aussagekräftiger als p50, weil sie zeigt, wie schlimm die Ausreißer sind (z. B. BGE v2-m3 bei Pool 30: p50 = 45 s, p95 = 162 s — die Tail-Latenz ist mehr als dreimal so schlecht wie der Median, s. Kap. 4.4).

**Kosten/1.000 €** — auf 1.000 Anfragen hochgerechnete geschätzte Kosten, basierend auf dem gemessenen/berechneten Preis der einzelnen Aufrufe in diesem Lauf (Details zum Kostenmodell in Kap. 2.5).

**Warum meist NDCG@5 zum Vergleichen verwendet wird:** Ein Reranker kann bei P@5 gut aussehen (viele „irgendwie relevante" Chunks in den Top 5), aber bei NDCG@5 schwächer, wenn die *besten* Chunks nicht ganz oben landen. NDCG@5 ist die einzige der vier Metriken, die Relevanzgrad **und** Position gleichzeitig berücksichtigt — deshalb ist sie im ganzen Dokument die primäre Vergleichsgröße; Recall/MRR/P@5 liefern die Zusatzperspektive (Vollständigkeit, „erster Treffer", „Anteil Ausschuss").

### 2.5 Wie vergleicht der Benchmark die Reranker fair?

Kernmechanik in `evaluation/benchmark.py`:

- **Echter Request-Scope:** `course_id`/`module_id` kommen aus dem Datensatz-Record und werden explizit an index-suchende Backends (Azure Semantic) übergeben — nie aus Chunk-Metadaten geraten.
- **Union-Pool-Scoring (TREC-Pooling):** Pro Query laufen zuerst **alle** Reranker. Jeder zurückgegebene, aber noch unlabelte Chunk (z. B. weil Azure Semantic den ganzen Index statt nur den 30er-Pool durchsucht) wird per LLM nachgelabelt und in `judge_cache.jsonl` zwischengespeichert. Erst danach werden **alle** Systeme gegen denselben Vereinigungs-Pool bewertet — NDCG-/Recall-Nenner sind für alle identisch. In diesem Lauf griffen alle nötigen Nachlabelungen auf den Cache zurück (0 neu, 68 aus Cache), es gab **keine** ungelabelten Fallbacks (`unlabeled_as_zero = 0` für alle Reranker/Pools).
- **Pool-Größen-Experiment (`--pool-sizes 10,30`):** Misst, wie empfindlich jeder Reranker auf die Kandidatenmenge reagiert. Azure Semantic ist hiervon ausgenommen — es durchsucht immer den vollen Index (das integrierte Produktions-Setup), die Pool-Größe „limitiert" es nicht.
- **Marginale Azure-Latenz:** Zusätzlich zur vollen Semantic-Suche (419 ms p50) wird eine reine Hybrid-Suche mit gleichem Scope getimt (340 ms p50) — die marginale Kosten des semantischen Rankings selbst liegen bei **≈79 ms**, weil der semantische Call in Produktion das Retrieval *ersetzt* statt zusätzlich zu laufen.
- **Strict-Mode für den LLM-Reranker:** Ein LLM-Fehler zählt im Benchmark als **Fehler** (Spalte `errors`), nicht als stiller Passthrough-Fallback (der in Produktion aus Robustheitsgründen aktiv bleibt). Ohne diese Härtung würde ein API-Fehler unbemerkt als „gute" Passthrough-Antwort in die Metrik einfließen.
- **Kostenmodell:** LLM Reranker rechnet Zeichen→Token→Preis nach hinterlegtem €/Token-Satz je Modell (0 € für GWDG-Modelle, echte Kosten für Azure-GPT-4o/-Mini); Azure Semantic ist ein Pauschalpreis (0,09 €/1.000 Queries); BGE und Baseline sind lokal/kostenlos (0 €).

### 2.6 Was an diesem Lauf konkret anders war

- **Reranker-Modell = `Gemma4`** (GWDG-gehostet) statt `Azure-Fallback` (GPT-4o) in den früheren explorativen Läufen — das ist der aktuelle **Produktionsdefault**, macht diesen Lauf also produktionsnäher, ändert aber auch die Vergleichbarkeit zu den alten Zahlen fundamental (s. Kap. 4.1).
- `--runs 1` statt der früheren 3 Wiederholungen (bei 68 Queries reicht 1 Lauf für ausreichend viele Latenz-Samples für p50/p95; spart GWDG-Last).
- `--cooldown 2`: 2 Sekunden Pause nach jedem LLM-Reranker-Call gegen GWDG-Rate-Limits — hat die Rate-Limit-Probleme trotzdem nicht verhindert (Kap. 4.3).
- Gesamtlaufzeit: Pool=10 **11:44 min**, Pool=30 **2:13:05 h** (die Differenz erklärt sich fast vollständig durch wiederholte 120-Sekunden-Backoff-Wartezeiten des LLM-Rerankers, nicht durch die anderen Reranker).

---

## 3. Ergebnisse

### 3.1 Gesamt

**Pool = 10**

| Reranker | NDCG@5 | Recall@5 | MRR | P@5 | Latenz p50 | Latenz p95 | Kosten/1k € |
|---|---|---|---|---|---|---|---|
| No Rerank (Baseline) | 0.771 | 0.539 | 0.856 | 0.729 | 0,0 ms | 0,0 ms | 0,0000 |
| LLM Reranker | 0.724 | 0.413 | 0.853 | 0.638 | 2.162,0 ms | 6.296,8 ms | 0,0000 |
| **Azure Semantic** | **0.875** | **0.609** | **0.926** | **0.788** | 419,0 ms | 996,9 ms | 0,0900 |
| BGE (v2-m3, large) | 0.862 | 0.582 | 0.934 | 0.768 | 2.925,3 ms | 7.715,2 ms | 0,0000 |
| BGE (base, small) | 0.820 | 0.568 | 0.898 | 0.747 | 530,7 ms | 1.335,3 ms | 0,0000 |

**Pool = 30**

| Reranker | NDCG@5 | Recall@5 | MRR | P@5 | Latenz p50 | Latenz p95 | Kosten/1k € |
|---|---|---|---|---|---|---|---|
| No Rerank (Baseline) | 0.744 | 0.360 | 0.856 | 0.729 | 0,0 ms | 0,3 ms | 0,0000 |
| LLM Reranker | 0.738 | 0.277 | 0.839 | 0.671 | 8.081,0 ms | 49.161,3 ms | 0,0000 |
| **Azure Semantic** | **0.841** | **0.408** | **0.926** | **0.788** | 562,2 ms | 1.437,0 ms | 0,0900 |
| BGE (v2-m3, large) | 0.835 | 0.406 | 0.919 | 0.785 | 45.332,9 ms | 161.804,9 ms | 0,0000 |
| BGE (base, small) | 0.772 | 0.371 | 0.872 | 0.750 | 18.130,1 ms | 59.857,9 ms | 0,0000 |

Recall@5 sinkt für alle Systeme deutlich von Pool 10 auf Pool 30 (Nenner wird größer, gleicher Zähler) — das ist erwartbares Artefakt der Pool-Größen-Vergrößerung, kein Qualitätsverlust der Reranker selbst.

### 3.2 Nach Sprache (nur Pool=10; Pool=30 zeigt dasselbe Muster)

| Sprache | n | Bestes System | NDCG@5 | Baseline NDCG@5 | Bemerkung |
|---|---|---|---|---|---|
| **de** | 41 | Azure Semantic | 0.887 | 0.803 | robuste Stichprobe |
| **en** | 23 | Azure Semantic | 0.839 | 0.692 | robuste Stichprobe, größte relative Verbesserung durch Reranking |
| es | 1 | alle gleich | 1.000 | 1.000 | **n=1, nicht interpretierbar** |
| fr | 1 | LLM/Azure/BGE-v2m3 | 0.873–1.000 | 0.747 | **n=1, nicht interpretierbar** |
| it | 1 | alle gleich | 1.000 | 1.000 | **n=1, nicht interpretierbar** |
| tr | 1 | LLM/Azure/BGE-v2m3 | 1.000 | 0.830 | **n=1, nicht interpretierbar** |

Nur de/en sind mit n=23/41 groß genug, um Aussagen daraus abzuleiten. Die vier übrigen Sprachen bestehen aus je einer einzelnen Frage — jede Zeile in der Konsolenausgabe für es/fr/it/tr ist ein Einzelfall, keine Verteilung.

### 3.3 Nach Fragetyp — ausgewählte, belastbare Befunde (Pool=10)

| kind | n | Baseline | Azure Semantic | BGE (v2-m3) | LLM Reranker | Einordnung |
|---|---|---|---|---|---|---|
| course_level | 8 | 0.849 | **0.987** | 0.949 | 0.794 | Azure klar vorn, brauchbare Stichprobe |
| module_level | 8 | 0.899 | **0.968** | 0.918 | 0.800 | dito |
| drupal_level | 8 | 0.771 | **0.963** | 0.906 | 0.744 | dito |
| comparison | 7 | 0.838 | **0.930** | 0.905 | 0.721 | dito |
| colloquial | 3 | **0.913** | 0.752 | 0.910 | 0.829 | Azure fällt hier klar hinter Baseline zurück — konsistent auch bei Pool=30 (0.752), aber n=3, als Hypothese behandeln, nicht als Beweis |
| negative | 10 | 0.329 | 0.500 | 0.523 | 0.061 | **Vorsicht bei Interpretation — s. Kap. 4.6** |

Die übrigen kinds (discovery, platform, technical, short, domain, concept) haben n≤8 und zeigen keine über beide Pool-Größen konsistenten, klar interpretierbaren Muster über das oben Gesagte hinaus — volle Matrix im Anhang (Kap. 7).

---

## 4. Analyse — Kernbefunde

### 4.1 Der Produktions-Reranker (LLM/Gemma4) ist schlechter als kein Reranking

Bei beiden Pool-Größen liegt „LLM Reranker" **unter** der No-Rerank-Baseline (0.724 vs. 0.771 bei Pool 10; 0.738 vs. 0.744 bei Pool 30) — und das trotz eines leichten Bonus zugunsten des LLM-Rerankers durch die Fehler-Exklusion (s. Kap. 4.3). Das steht in scharfem Kontrast zu den alten explorativen Läufen in [ANALYSE_FEAT_IMPROVEMENTS.md](ANALYSE_FEAT_IMPROVEMENTS.md) Kap. 4.10, wo der LLM-Reranker (dort mit `Azure-Fallback`/GPT-4o statt `Gemma4`) die Baseline noch klar schlug (0.780–0.791 vs. 0.749–0.666). Der einzige Unterschied zwischen „LLM Reranker gewinnt" und „LLM Reranker verliert" ist **welches Modell rerankt** — nicht die Methode an sich. GPT-4o scheint als Reranker deutlich kompetenter zu sein als das aktuell produktiv genutzte `Gemma4` (GWDG). Das ist die wichtigste Einzelerkenntnis dieses Laufs: **die aktuelle Produktionskonfiguration reranked aktiv schlechter, als würde man gar nicht reranken.**

### 4.2 Azure Semantic und BGE (v2-m3) liegen vorn, dicht beieinander

Beide liegen bei beiden Pool-Größen ca. 6-9 NDCG@5-Punkte vor der Baseline und 10-14 Punkte vor dem LLM-Reranker. Der Abstand zwischen den beiden (0.875 vs. 0.862 bei Pool 10, 0.841 vs. 0.835 bei Pool 30) ist bei n=68 nicht als gesichert zu werten (Kap. 5) — beide sind ernstzunehmende Kandidaten für den Produktions-Default.

### 4.3 Der LLM-Reranker ist unter Produktionsmodell unzuverlässig

Bei Pool=30 schlugen **6 von 68** LLM-Reranker-Calls trotz 2-Sekunden-Cooldown mit `429 Rate limit exceeded` fehl (GWDG-Endpoint) — jeweils nach einem blockierenden 120-Sekunden-Retry-Versuch. Das ist bereits bei einem **sequenziellen** Benchmark-Lauf ohne echte Nutzerlast der Fall; unter realer, paralleler Last wäre eine höhere Fehlerrate zu erwarten. Wichtig: Der Strict-Mode zählt diese 6 Queries als Fehler und **schließt sie aus der Durchschnittsberechnung aus** (`errors=6`, kein Fallback-Score) — sie werden also nicht als „schlechtestes Ergebnis" (0) gewertet, sondern gar nicht mitgezählt. Der ausgewiesene NDCG@5-Wert des LLM-Rerankers (0.738 bei Pool 30) ist damit über nur 62 der 68 Queries gemittelt und tendenziell **zu gut**, nicht zu schlecht — der reale Befund (schlechter als Baseline) ist also, wenn überhaupt, konservativ zu seinen Gunsten verzerrt.

### 4.4 BGE (v2-m3) — Latenz-Anomalie bei Pool 30

Die Latenz des großen BGE-Modells steigt von Pool 10 auf Pool 30 (3× mehr Chunks) nicht etwa um den Faktor 3, sondern um den **Faktor ~15** (2.925 ms → 45.333 ms p50; p95 sogar auf 161.805 ms). Das ist nicht die erwartete lineare Skalierung eines Cross-Encoders und sollte **nicht unkommentiert als „BGE ist bei größeren Pools unbrauchbar" gewertet werden**, ohne die Ursache zu klären. Plausible Erklärungen, keine davon in diesem Lauf verifiziert:
- Einzelne sehr lange Chunks im 30er-Pool, die (ohne Trunkierung) quadratisch in die Attention-Kosten des Cross-Encoders eingehen und den p95/p50 durch Ausreißer verzerren.
- Der Lauf fand auf einem Laptop über >2 Stunden am Stück statt, parallel zu wiederholten 120-Sekunden-Blockier-Wartezeiten des LLM-Rerankers — thermisches Throttling oder Hintergrundlast sind nicht ausgeschlossen.
- Fehlende Batch-Größen-Abstimmung des `sentence-transformers`-CrossEncoders für 30 statt 10 Paare auf CPU.

Für die Praxis ist das ohnehin nachrangig: Produktiv wird typischerweise mit deutlich kleineren Kandidatenmengen (top_n Retrieval, nicht 30) gerankt — bei Pool 10 ist BGE (v2-m3) mit 2.925 ms p50 nutzbar, wenn auch langsamer als Azure Semantic (419 ms). **Vor einer Entscheidung für BGE bei größeren Pools sollte die Latenz-Anomalie auf dedizierter (Server-)Hardware isoliert nachgemessen werden.**

### 4.5 Die „0 € Kosten" des LLM-Rerankers sind eine Momentaufnahme, kein struktureller Vorteil

`Gemma4` (GWDG) ist im Kostenmodell als Flatrate mit 0 €/Token hinterlegt — deshalb zeigt die Tabelle 0,0000 € für den LLM-Reranker, identisch zu den echt kostenlosen lokalen Optionen (BGE, Baseline). Das verschleiert, dass die Systemarchitektur selbst einen **Fallback auf `Azure-Fallback` (GPT-4o, echte Kosten)** vorsieht, sobald GWDG nicht verfügbar ist (`LLM.gwdg_unavailable`-Mechanismus) — und genau dieses Szenario (Rate-Limiting/Nichtverfügbarkeit des GWDG-Endpoints) ist in Kap. 4.3 in diesem Lauf real aufgetreten. Die alten explorativen Läufe mit `Azure-Fallback` als Reranker-Modell zeigten 8–27 €/1.000 Queries. Kurz: **Der LLM-Reranker ist nur so lange „kostenlos", wie GWDG verfügbar ist — genau die Bedingung, die in diesem Lauf mehrfach verletzt wurde.**

### 4.6 Der „negative"-Fragetyp misst nicht direkt No-Answer-Qualität

Der Fragetyp `negative` (10 Queries) soll Kalibrierungsfragen enthalten, die die Wissensbasis „sehr wahrscheinlich nicht" abdeckt. Tatsächliche Prüfung der Ground-Truth-Labels zeigt: **6 der 10** Negative-Queries haben mindestens einen vom LLM-Judge als „teilweise relevant" (relevance=1) eingestuften Chunk im Kandidatenpool — nur 4 sind im Sinne der Metrik echte Nullpools (kein einziger relevanter Chunk, dort ist NDCG@5 für **alle** Systeme zwangsläufig 0.0, da der ideale DCG selbst 0 ist). Die NDCG-Unterschiede zwischen den Systemen in der `negative`-Zeile (Baseline 0.329, LLM 0.061, Azure 0.500, BGE-v2m3 0.523) spiegeln also größtenteils, **wie gut die Systeme die vom Judge als grenzwertig-relevant eingestuften Chunks in den Top-5 platzieren** — nicht, ob das System korrekt „keine Antwort" signalisiert. Die eigentliche No-Answer-/`min_score`-Logik wird an anderer Stelle (Integrationstests, `EmptyModule`-Kurzschluss) getestet, nicht durch diese Benchmark-Metrik.

---

## 5. Wie belastbar sind diese Ergebnisse?

Der Lauf ist methodisch solide aufgesetzt (echter Scope, TREC-Union-Pooling, strict-mode Fehlerbehandlung, konsistenter Judge zwischen Datensatz-Erstellung und Nachlabelung) — trotzdem sollten die Zahlen mit folgenden Einschränkungen gelesen werden:

- **Stichprobengröße insgesamt (n=68) ist klein bis moderat** für ein IR-Benchmark. Unterschiede von wenigen NDCG-Punkten zwischen den Top-2-Systemen (Azure Semantic, BGE v2-m3) liegen im Bereich statistischen Rauschens — es wurde kein Signifikanztest (z. B. paarweiser Bootstrap/Permutationstest über die Queries) durchgeführt. Die klaren, großen Abstände (Top-2 vs. Baseline vs. LLM-Reranker) sind dagegen robust genug, um handlungsleitend zu sein.
- **Subgruppen sind oft noch kleiner:** Pro-Sprache-Werte für es/fr/it/tr basieren auf **je einer** Frage; viele `kind`-Kategorien haben n≤8. Diese Zeilen sind beschreibend, nicht statistisch belastbar (einzeln vermerkt in Kap. 3.2/3.3).
- **Ein einziger automatisierter Judge, keine menschliche Validierung.** Die Ground-Truth-Relevanzlabels stammen von einem einzelnen LLM (`Azure-Fallback`/GPT-4o), es gibt keine zweite Bewertungsinstanz und kein Inter-Rater-Agreement. Relative Vergleiche zwischen Rerankern bleiben davon weitgehend unberührt (derselbe Judge, derselbe Pool für alle Systeme), aber die absoluten NDCG-Werte tragen die systematischen Verzerrungen dieses einen Modells (z. B. Neigung zu bestimmten Formulierungsstilen).
- **Nur 1 Durchlauf (`--runs 1`).** Frühere Läufe nutzten 3 Wiederholungen für stabilere Latenz-Schätzungen; hier gibt es keine Varianzangabe für Latenz oder für mögliche Nichtdeterminismus der LLM-basierten Systeme (Azure Semantic/LLM Reranker können bei wiederholtem Aufruf leicht andere Scores liefern).
- **Survivorship-Bias beim LLM-Reranker** (Kap. 4.3): 6 fehlgeschlagene Queries bei Pool 30 fließen nicht als Negativ-Ergebnis in den Schnitt ein, sondern werden ausgeschlossen — der ausgewiesene Wert ist also eher zu gut als zu schlecht für den LLM-Reranker.
- **Laufbedingungen:** Pool=30 lief über gut 2 Stunden auf einer lokalen Maschine, mit wiederholten minutenlangen Blockier-Wartezeiten durch Rate-Limiting. Das ist unwahrscheinlich, die Ranking-Metriken der anderen Systeme zu verzerren (jedes Timing ist um den jeweiligen Reranker-Call selbst gekapselt), aber die BGE-Latenz-Anomalie (Kap. 4.4) sollte trotzdem unter saubereren Bedingungen reproduziert werden, bevor sie als Tatsache gilt.
- **Kein Vergleich mit den alten Läufen auf identischer Basis möglich:** Reranker-Modell (`Gemma4` vs. `Azure-Fallback`), Datensatzversion und Strict-Mode unterscheiden sich gleichzeitig — der auffällige Rollentausch des LLM-Rerankers (Kap. 4.1) lässt sich mit diesem Lauf allein nicht eindeutig auf „Modellwechsel" vs. „anderer Datensatz" vs. „Strict-Mode" zurückführen, auch wenn das Modell die naheliegendste Erklärung ist.

**Zusammengefasst:** Für die grobe Richtungsentscheidung „LLM-Reranker mit Gemma4 raus, Azure Semantic oder BGE-v2-m3 rein" ist der Lauf ausreichend belastbar — die Effektgrößen sind groß genug, um Rauschen und die genannten Einschränkungen zu überstehen. Für die Feinentscheidung „Azure Semantic **oder** BGE-v2-m3" liefert dieser einzelne Lauf keine gesicherte Antwort.

---

## 6. Empfehlungen

1. **`RERANKER_TYPE` vom Default `llm` weg umstellen.** Der Produktions-Reranker mit dem aktuellen Modell verschlechtert die Ergebnisqualität aktiv und ist der langsamste, unzuverlässigste Kandidat. Das ist der klarste Befund dieses Laufs.
2. **Kandidat 1: Azure Semantic** — beste Gesamtqualität, niedrigste Latenz unter den Nicht-Baseline-Optionen (419 ms, marginal ~79 ms gegenüber ohnehin nötigem Retrieval), planbare Pauschalkosten (0,09 €/1.000 Queries). Schwäche: fällt bei `colloquial`-Fragen unter die Baseline (Kap. 3.3) — mit n=3 nicht bewiesen, aber beobachten.
3. **Kandidat 2: BGE (bge-reranker-v2-m3)** — nahezu gleichwertige Qualität, 0 € Betriebskosten, aber vor einer Entscheidung die Latenz-Anomalie bei größeren Pools (Kap. 4.4) auf Zielhardware klären. Für die aktuelle top-n-Größenordnung (~10) unproblematisch.
4. **Falls der LLM-Reranker weiterverfolgt wird:** nur mit einem stärkeren Modell (in den alten Läufen war GPT-4o klar besser als Baseline) — dann aber bewusst mit den damit verbundenen echten Kosten (8-27 €/1.000 Queries in den alten Läufen) und ohne den GWDG-Zuverlässigkeitsvorteil kalkulieren.
5. **Vor einer endgültigen Festlegung zwischen Azure Semantic und BGE-v2-m3:** einen zweiten, saubereren Lauf mit `--runs 3` (Varianzschätzung) auf einem größeren/aktualisierten Datensatz (`--generate --regenerate` nach dem nächsten Re-Ingest) fahren, um die knappe Differenz zwischen beiden abzusichern.
6. **Aufräumen vor dem nächsten offiziellen Lauf** (siehe [ANALYSE_FEAT_IMPROVEMENTS.md](ANALYSE_FEAT_IMPROVEMENTS.md) Kap. 7.10): Die unversionierten Zwischenstände in `evaluation/dataset/` (`first_try/`, `weg/`, `queries copy.jsonl`, alte `results_*.json` im Repo-Root) sollten entfernt oder klar als Archiv markiert werden, damit zukünftige Läufe nicht versehentlich mit einem veralteten Query-Set verglichen werden.

---

## 7. Anhang: Vollständige Rohdaten

### 7.1 Pool=10, nach Sprache

| Sprache | Reranker | NDCG@5 | Recall@5 | MRR | P@5 |
|---|---|---|---|---|---|
| de (n=41) | No Rerank | 0.803 | 0.536 | 0.879 | 0.771 |
| de | LLM Reranker | 0.759 | 0.410 | 0.878 | 0.702 |
| de | Azure Semantic | 0.887 | 0.585 | 0.939 | 0.815 |
| de | BGE (v2-m3) | 0.873 | 0.565 | 0.951 | 0.805 |
| de | BGE (base) | 0.856 | 0.553 | 0.939 | 0.785 |
| en (n=23) | No Rerank | 0.692 | 0.570 | 0.790 | 0.617 |
| en | LLM Reranker | 0.626 | 0.417 | 0.783 | 0.461 |
| en | Azure Semantic | 0.839 | 0.686 | 0.891 | 0.704 |
| en | BGE (v2-m3) | 0.829 | 0.641 | 0.891 | 0.661 |
| en | BGE (base) | 0.746 | 0.631 | 0.808 | 0.652 |

(es/fr/it/tr: je n=1, siehe Kap. 3.2 — Rohwerte in `results_fair.json`.)

### 7.2 NDCG@5 nach Fragetyp — vollständige Matrix

**Pool=10:**

| Reranker | colloquial | comparison | concept | course_level | discovery | domain | drupal_level | module_level | negative | platform | short | technical |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| No Rerank | 0.913 | 0.838 | 0.846 | 0.849 | 0.837 | 0.680 | 0.771 | 0.899 | 0.329 | 0.781 | 0.978 | 0.940 |
| LLM Reranker | 0.829 | 0.721 | 0.923 | 0.794 | 0.938 | 0.828 | 0.744 | 0.800 | 0.061 | 1.000 | 0.932 | 0.940 |
| Azure Semantic | 0.752 | 0.930 | 0.939 | 0.987 | 0.870 | 0.825 | 0.963 | 0.968 | 0.500 | 0.954 | 1.000 | 0.990 |
| BGE (v2-m3) | 0.910 | 0.905 | 0.932 | 0.949 | 0.892 | 0.698 | 0.906 | 0.918 | 0.523 | 0.949 | 0.978 | 0.990 |
| BGE (base) | 0.749 | 0.905 | 0.892 | 0.882 | 0.862 | 0.744 | 0.841 | 0.930 | 0.438 | 0.945 | 0.978 | 0.931 |

**Pool=30:**

| Reranker | colloquial | comparison | concept | course_level | discovery | domain | drupal_level | module_level | negative | platform | short | technical |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| No Rerank | 0.913 | 0.838 | 0.809 | 0.779 | 0.827 | 0.660 | 0.736 | 0.899 | 0.290 | 0.737 | 0.978 | 0.940 |
| LLM Reranker | 0.911 | 0.884 | 0.968 | 0.766 | 0.926 | 0.920 | 0.809 | 0.800 | 0.061 | 0.961 | 0.978 | 0.911 |
| Azure Semantic | 0.752 | 0.912 | 0.897 | 0.914 | 0.857 | 0.803 | 0.923 | 0.968 | 0.429 | 0.901 | 1.000 | 0.990 |
| BGE (v2-m3) | 0.887 | 0.941 | 0.919 | 0.890 | 0.811 | 0.660 | 0.917 | 0.918 | 0.419 | 0.917 | 1.000 | 0.990 |
| BGE (base) | 0.791 | 0.885 | 0.894 | 0.810 | 0.794 | 0.713 | 0.830 | 0.930 | 0.243 | 0.847 | 1.000 | 0.931 |

### 7.3 Fehler & Datenqualität dieses Laufs

- Fehlgeschlagene Rerank-Calls: **6**, alle beim LLM Reranker, alle bei Pool=30 (Rate-Limit `429` auf dem GWDG-Endpoint) — Pool=10 lief fehlerfrei durch.
- Betroffene Queries (aus dem Konsolen-Log): „Erkläre mir was ein neuronales Netz ist.", „Was ist ein Large Language Model?", „Was versteht man unter Bias in KI-Systemen?", „What is overfitting and how do you prevent it?", „Wie funktioniert Backpropagation beim Deep Learning Training", „Brauche ich Vorkenntnisse, um etwas über KI im Führungsalltag …".
- Nachlabelung unlabelter Chunks: 0 neu generiert, 68 aus `judge_cache.jsonl` bedient — keine Ground-Truth-Lücken in diesem Lauf.
- `unlabeled_as_zero`: 0 für alle Reranker/Pool-Kombinationen (keine Benachteiligung re-suchender Backends durch fehlende Urteile).
