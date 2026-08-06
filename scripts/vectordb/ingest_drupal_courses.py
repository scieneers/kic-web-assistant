"""
Lädt einen Drupal-PageType in den Azure Search Index.
Ausführen: uv run python ingest_drupal_courses.py

PAGE_TYPE Optionen:
  PageTypes.COURSE    – Kurse
  PageTypes.BLOGPOST  – Blogposts
  PageTypes.PAGE      – Seiten
  PageTypes.ABOUT_US  – Über uns
  PageTypes.SPEZIAL   – Spezialseiten (Stadt Land Datenfluss etc.)
"""
import logging

from src.env import env
from src.loaders.drupal import Drupal, PageTypes
from src.loaders.get_data import Fetch_Data
from src.loaders.helper import iter_nodes_from_document_hierarchical

# ── Hier anpassen ────────────────────────────────────────────
# PAGE_TYPE = PageTypes.ABOUT_US
PAGE_TYPE = PageTypes.COURSE
# PAGE_TYPE = PageTypes.BLOGPOST
# PAGE_TYPE = PageTypes.PAGE
# PAGE_TYPE = PageTypes.SPEZIAL
# COURSE, BLOGPOST, PAGE, ABOUT_US, SPEZIAL

# ─────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.DEBUG, format="{asctime} - {levelname} - {message}", style="{")

drupal = Drupal(
    base_url=env.DRUPAL_URL,
    username=env.DRUPAL_USERNAME,
    client_id=env.DRUPAL_CLIENT_ID,
    client_secret=env.DRUPAL_CLIENT_SECRET,
    grant_type=env.DRUPAL_GRANT_TYPE,
)
docs = drupal.get_page_type(PAGE_TYPE)
print(f"\n{len(docs)} Dokumente vom Typ '{PAGE_TYPE.value[0]}' geladen.")

fetcher = Fetch_Data()
fetcher.search_store.delete_by_filter(
    env.AZURE_SEARCH_INDEX,
    f"source eq 'Drupal' and type eq '{PAGE_TYPE.value[0]}'",
)
print(f"Alte '{PAGE_TYPE.value[0]}'-Dokumente aus dem Index gelöscht.")

nodes = []
for doc in docs:
    nodes.extend(iter_nodes_from_document_hierarchical(doc))

uploaded = fetcher._embed_and_upsert_nodes(nodes, stage=f"DRUPAL_{PAGE_TYPE.value[0].upper()}")
print(f"{uploaded} Chunks hochgeladen.")
