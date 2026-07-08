import json
import logging
import re
from types import SimpleNamespace
from typing import Any, Iterator

from azure.core.exceptions import HttpResponseError, ResourceNotFoundError, ServiceRequestError
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from src.env import env

# Azure AI Search hard limits for the indexing API. We stay just under them.
# https://learn.microsoft.com/azure/search/search-limits-quotas-capacity
_MAX_DOCS_PER_BATCH = 1000
_MAX_BATCH_BYTES = 14 * 1024 * 1024  # 16 MB request cap; target 14 MB for headroom

# Azure document keys may only contain letters, digits, dash, underscore and
# equals. LlamaIndex node ids are UUIDs (already valid), but we sanitize
# defensively so an unexpected id never makes a whole batch fail.
_KEY_INVALID = re.compile(r"[^A-Za-z0-9_\-=]")

_VECTOR_PROFILE = "hnsw-cosine"
_HNSW_CONFIG = "hnsw-config"

# Azure's semantic ranker rescores at most the top 50 of the initial results —
# when semantic ranking is on, the dense leg should propose at least that many.
# https://learn.microsoft.com/azure/search/semantic-search-overview
_SEMANTIC_RERANK_WINDOW = 50


def sanitize_key(raw: str) -> str:
    """Coerce an arbitrary id into a valid Azure AI Search document key."""
    return _KEY_INVALID.sub("_", raw)


class VectorDBAzureSearch:
    """Azure AI Search vector database abstraction.

    Auth is keyless: the deployed service rejects API keys
    (``local_authentication_enabled = false``), so we always use
    ``DefaultAzureCredential``. In production it resolves the user-assigned
    managed identity (bound via the ``AZURE_CLIENT_ID`` app setting); locally it
    falls back to the developer's ``az login`` credentials, whose IP must be in
    the service's ``allowed_ip_ranges``.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("loader")
        self.endpoint = env.AZURE_SEARCH_ENDPOINT
        self.index_name = env.AZURE_SEARCH_INDEX
        self.credential = DefaultAzureCredential()
        self.index_client = SearchIndexClient(endpoint=self.endpoint, credential=self.credential)
        self._search_clients: dict[str, SearchClient] = {}
        self.logger.info("Azure AI Search endpoint=%s default_index=%s", self.endpoint, self.index_name)

    # ------------------------------------------------------------------
    # Clients
    # ------------------------------------------------------------------
    def _client(self, index_name: str | None = None) -> SearchClient:
        name = index_name or self.index_name
        if name not in self._search_clients:
            self._search_clients[name] = SearchClient(
                endpoint=self.endpoint, index_name=name, credential=self.credential
            )
        return self._search_clients[name]

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------
    def create_index(self, index_name: str, vector_size: int) -> None:
        """Create or update the hybrid (vector + BM25) index.

        Uses create_or_update_index so new fields (e.g. source_doc_key,
        content_hash) are added to existing indexes without data loss.
        """
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            # German analyzer: the corpus (KI-Campus) is predominantly German.
            SearchableField(name="text", type=SearchFieldDataType.String, analyzer_name="de.microsoft"),
            SearchableField(name="fullname", type=SearchFieldDataType.String, analyzer_name="de.microsoft"),
            SearchableField(name="title", type=SearchFieldDataType.String, analyzer_name="de.microsoft"),
            SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="type", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="course_id", type=SearchFieldDataType.Int64, filterable=True),
            SimpleField(name="module_id", type=SearchFieldDataType.Int64, filterable=True),
            # url is filterable so the loader's sanity check can find empty urls.
            SimpleField(name="url", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="is_important", type=SearchFieldDataType.Boolean, filterable=True),
            SimpleField(name="date_created", type=SearchFieldDataType.String, filterable=True),
            # Lossless metadata round-trip; retrieval-only, never searched/filtered.
            SimpleField(name="metadata_json", type=SearchFieldDataType.String),
            # Change-detection fields: one stable key per source document, one hash
            # per content+chunking-params. Stored on every chunk so the hash lives
            # and dies with the chunks (self-healing on partial failures).
            SimpleField(name="source_doc_key", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="content_hash", type=SearchFieldDataType.String),
            SearchField(
                name="dense",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=vector_size,
                vector_search_profile_name=_VECTOR_PROFILE,
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name=_HNSW_CONFIG,
                    parameters=HnswParameters(metric=VectorSearchAlgorithmMetric.COSINE),
                )
            ],
            profiles=[VectorSearchProfile(name=_VECTOR_PROFILE, algorithm_configuration_name=_HNSW_CONFIG)],
        )

        semantic_search = SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name="default",
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=SemanticField(field_name="title"),
                        content_fields=[SemanticField(field_name="text")],
                        keywords_fields=[SemanticField(field_name="fullname")],
                    ),
                )
            ]
        )

        index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search, semantic_search=semantic_search)
        self.index_client.create_or_update_index(index)
        self.logger.info(
            "Azure AI Search index '%s' schema ensured (dense dim=%s, hybrid vector+BM25).",
            index_name,
            vector_size,
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    @staticmethod
    def _doc_bytes(doc: dict) -> int:
        return len(json.dumps(doc, default=str).encode("utf-8"))

    def _batches(self, docs: list[dict]) -> Iterator[list[dict]]:
        """Yield batches bounded by both Azure limits: <=1000 docs and <=~14 MB."""
        current: list[dict] = []
        current_bytes = 0
        for doc in docs:
            size = self._doc_bytes(doc)
            too_many = len(current) >= _MAX_DOCS_PER_BATCH
            too_big = current and current_bytes + size > _MAX_BATCH_BYTES
            if too_many or too_big:
                yield current
                current, current_bytes = [], 0
            current.append(doc)
            current_bytes += size
        if current:
            yield current

    def upload_documents(self, index_name: str, documents: list[dict]) -> int:
        """Upsert (merge-or-upload) documents, chunked to Azure batch limits.

        A failed batch is logged and skipped (not raised) so a transient hiccup
        does not abort an entire ingestion run, mirroring the previous behavior.
        """
        if not documents:
            return 0
        uploaded = 0
        for batch in self._batches(documents):
            results = None
            for attempt in range(2):
                try:
                    results = self._client(index_name).merge_or_upload_documents(documents=batch)
                    break
                except ServiceRequestError as e:
                    self._search_clients.pop(index_name, None)
                    if attempt == 0:
                        self.logger.warning(
                            "Azure AI Search connection error (stale connection?), retrying with fresh client: index=%s exc=%s",
                            index_name,
                            e,
                        )
                        continue
                    self.logger.error(
                        "Azure AI Search connection error after retry, skipping batch: index=%s docs=%s exc=%s",
                        index_name,
                        len(batch),
                        e,
                    )
                except HttpResponseError as e:
                    self.logger.warning(
                        "Azure AI Search upload failed: index=%s docs=%s exc=%s",
                        index_name,
                        len(batch),
                        e,
                    )
                    break
            if results is None:
                continue
            failed = [r for r in results if not r.succeeded]
            if failed:
                self.logger.warning(
                    "Azure AI Search upload: %s/%s docs failed in batch (index=%s). First error: key=%s status=%s msg=%s",
                    len(failed),
                    len(batch),
                    index_name,
                    failed[0].key,
                    failed[0].status_code,
                    failed[0].error_message,
                )
            uploaded += len(batch) - len(failed)
        self.logger.debug("Uploaded %s documents into '%s'.", uploaded, index_name)
        return uploaded

    # ------------------------------------------------------------------
    # Deletion (delete-by-filter emulation)
    # ------------------------------------------------------------------
    def _iter_keys(self, index_name: str, odata_filter: str) -> Iterator[str]:
        """Stream all document keys matching an OData filter.

        Azure AI Search has no delete-by-query, so we page the matching keys
        (the SDK follows continuation links transparently) and the caller
        deletes them by key.

        No `top` is passed: it caps the *total* results returned across the
        whole iteration (not a page size), so any filter matching more than
        the cap would silently yield only a partial key set.
        """
        client = self._client(index_name)
        results = client.search(search_text="*", filter=odata_filter, select=["id"])
        for item in results:
            yield item["id"]

    def _true_count(self, index_name: str | None, odata_filter: str) -> int | None:
        """Exact count of documents matching a filter (top=0, no documents transferred).

        Used as a cheap cross-check that a full scan actually saw everything.
        """
        try:
            results = self._client(index_name).search(
                search_text="*", filter=odata_filter, top=0, include_total_count=True
            )
            return results.get_count()
        except HttpResponseError:
            return None

    def delete_by_filter(self, index_name: str, odata_filter: str) -> int:
        """Delete every document matching an OData filter.

        Azure AI Search has no single-call delete-by-filter: search the matching
        keys, then delete them in key batches. Preserves the per-course
        (Moodle) and per-source (Drupal, Moochup) deletion semantics.
        """
        self.logger.info("Azure AI Search delete_by_filter: index=%s filter=%s", index_name, odata_filter)
        client = self._client(index_name)
        deleted = 0
        try:
            # Materialize all matching keys BEFORE deleting. Deleting while the
            # search iterator is still paging would shift skip-based pagination
            # and could let some matching documents slip through undeleted.
            keys = list(self._iter_keys(index_name, odata_filter))
            true_count = self._true_count(index_name, odata_filter)
            if true_count is not None and len(keys) < true_count:
                self.logger.error(
                    "Azure AI Search delete_by_filter: scanned %s keys but %s documents match filter"
                    " (index=%s filter=%s) — some matches were not collected for deletion.",
                    len(keys),
                    true_count,
                    index_name,
                    odata_filter,
                )
            for start in range(0, len(keys), _MAX_DOCS_PER_BATCH):
                batch = [{"id": k} for k in keys[start : start + _MAX_DOCS_PER_BATCH]]
                client.delete_documents(documents=batch)
                deleted += len(batch)
        except HttpResponseError as e:
            self.logger.warning("Azure AI Search delete_by_filter failed (index=%s): %s", index_name, e)
            return deleted
        self.logger.info("Azure AI Search delete_by_filter removed %s documents (index=%s).", deleted, index_name)
        return deleted

    def load_content_hashes(self, source: str, index_name: str | None = None) -> dict[str, str]:
        """Return {source_doc_key: content_hash} for all existing chunks of a source.

        Multiple chunks share the same source_doc_key; only the first seen is kept
        (they all carry an identical hash). Used by the loader to skip unchanged
        documents and detect stale ones.
        """
        client = self._client(index_name)
        existing: dict[str, str] = {}
        odata_filter = f"source eq '{source}'"
        try:
            results = client.search(
                search_text="*",
                filter=odata_filter,
                select=["source_doc_key", "content_hash"],
            )
            rows_seen = 0
            for item in results:
                rows_seen += 1
                key = item.get("source_doc_key")
                h = item.get("content_hash")
                if key and h and key not in existing:
                    existing[key] = h
            true_count = self._true_count(index_name, odata_filter)
            if true_count is not None and rows_seen < true_count:
                self.logger.error(
                    "load_content_hashes: scanned %s rows but %s documents match source=%s"
                    " (index=%s) — existing-hash map is incomplete, change-detection will misfire.",
                    rows_seen,
                    true_count,
                    source,
                    index_name or self.index_name,
                )
        except HttpResponseError as e:
            self.logger.warning("load_content_hashes failed (source=%s): %s", source, e)
        self.logger.info(
            "Loaded %s existing source keys (source=%s index=%s)",
            len(existing),
            source,
            index_name or self.index_name,
        )
        return existing

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        *,
        index_name: str | None = None,
        odata_filter: str | None = None,
        top: int = 10,
        candidate_factor: int = 3,
        use_semantic: bool = False,
        semantic_config_name: str = "default",
    ) -> list[dict]:
        """Hybrid search: BM25 over ``text`` + vector over ``dense``.

        Azure AI Search fuses the two result sets with Reciprocal Rank Fusion
        automatically when both ``search_text`` and a vector query are supplied.
        Returns raw result dicts (each carries ``@search.score``, or
        ``@search.reranker_score`` when use_semantic=True).

        use_semantic: enable Azure AI Search semantic reranking (requires
            Standard tier and a semantic configuration named ``semantic_config_name``
            on the index).
        """
        client = self._client(index_name)
        k_nearest = max(top * candidate_factor, top)
        if use_semantic:
            # Azure's semantic ranker rescores the top 50 of the initial result
            # set. With a small `top` (e.g. 5) the dense leg would only propose
            # top*factor candidates and starve the reranker — feed it the full
            # rerank window instead.
            k_nearest = max(k_nearest, _SEMANTIC_RERANK_WINDOW)
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=k_nearest,
            fields="dense",
        )
        extra = {}
        if use_semantic:
            extra["query_type"] = "semantic"
            extra["semantic_configuration_name"] = semantic_config_name
        results = client.search(
            search_text=query_text,
            vector_queries=[vector_query],
            filter=odata_filter,
            top=top,
            select=["id", "text", "metadata_json"],
            **extra,
        )
        return list(results)

    # ------------------------------------------------------------------
    # Read-back helpers
    # ------------------------------------------------------------------
    def get_course_module_records(self, index_name: str | None = None):
        """Return (course_records, module_records) for the frontend tree.

        Each record exposes a ``.payload`` dict.
        Course records have a ``course_id`` but no ``module_id``; module records
        carry a ``module_id``.
        """
        client = self._client(index_name)
        # Only docs that participate in the tree carry a course_id. No `top`:
        # every chunk of every module carries course_id/module_id, so this can
        # easily exceed 1000 rows — see load_content_hashes for why a `top`
        # cap here would silently truncate the result instead of paging.
        results = client.search(
            search_text="*",
            filter="course_id ne null",
            select=["course_id", "module_id", "fullname"],
        )

        courses: list[SimpleNamespace] = []
        modules: list[SimpleNamespace] = []
        for item in results:
            course_id = item.get("course_id")
            module_id = item.get("module_id")
            payload: dict[str, Any] = {"course_id": course_id, "fullname": item.get("fullname")}
            if module_id is not None:
                payload["module_id"] = module_id
                modules.append(SimpleNamespace(payload=payload))
            elif isinstance(course_id, int):
                courses.append(SimpleNamespace(payload=payload))

        courses.sort(key=lambda r: r.payload["course_id"])
        modules.sort(key=lambda r: r.payload["module_id"])
        return courses, modules

    def any_match(self, odata_filter: str, *, index_name: str | None = None) -> bool:
        """Return True if at least one document matches the OData filter."""
        results = self._client(index_name).search(
            search_text="*", filter=odata_filter, select=["id"], top=1
        )
        return any(True for _ in results)

    def check_if_course_exists(self, course_id: int) -> bool:
        """Check whether any document for the given course exists."""
        return self.any_match(f"course_id eq {int(course_id)}")

    def check_if_module_exists(self, module_id: int) -> bool:
        """Check whether any document for the given module exists."""
        return self.any_match(f"module_id eq {int(module_id)}")


if __name__ == "__main__":
    # Smoke test: requires `az login` and your IP in allowed_ip_ranges.
    db = VectorDBAzureSearch()
    print("Connected. Course exists(79)?", db.check_if_course_exists(79))
