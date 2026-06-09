import logging
import os
import sys
import time
import warnings
from typing import List

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.http.models import Distance, PointStruct, VectorParams, SparseVectorParams

from src.env import env

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

_DEPRECATED_MODES = {
    "dev_remote": 120,
    "prod_remote": 30,
}


class VectorDBQdrant:
    def __init__(self, mode: str = "auto"):
        self.logger = logging.getLogger("loader")

        # Backwards-compat: accept legacy version strings with a deprecation warning
        if mode in _DEPRECATED_MODES:
            warnings.warn(
                f"VectorDBQdrant(mode='{mode}') is deprecated. Use mode='auto' and set ENVIRONMENT instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            timeout = _DEPRECATED_MODES[mode]
            self.logger.info("Connecting to Qdrant at %s (legacy mode='%s', timeout=%ss)", env.QDRANT_URL, mode, timeout)
            self.client = QdrantClient(url=env.QDRANT_URL, port=443, https=True, timeout=timeout, api_key=env.QDRANT_API_KEY)
            self._probe_connection()
        elif mode == "auto":
            is_dev = env.ENVIRONMENT == "DEV"
            timeout = 120 if is_dev else 30
            # Local dev: a plain-http or localhost URL points at a container (e.g. `task qdrant`),
            # so connect over the URL's own port/scheme without TLS or an api key.
            is_local = env.QDRANT_URL.startswith("http://") or any(
                host in env.QDRANT_URL for host in ("localhost", "127.0.0.1")
            )
            if is_local:
                self.logger.info(
                    "Connecting to local Qdrant at %s (ENVIRONMENT=%s, timeout=%ss)", env.QDRANT_URL, env.ENVIRONMENT, timeout
                )
                self.client = QdrantClient(url=env.QDRANT_URL, timeout=timeout)
            else:
                self.logger.info(
                    "Connecting to Qdrant at %s (ENVIRONMENT=%s, timeout=%ss)", env.QDRANT_URL, env.ENVIRONMENT, timeout
                )
                self.client = QdrantClient(
                    url=env.QDRANT_URL, port=443, https=True, timeout=timeout, api_key=env.QDRANT_API_KEY
                )
            self._probe_connection()
        elif mode == "memory":
            self.client = QdrantClient(":memory:")
        elif mode == "disk":
            self.client = QdrantClient("localhost", port=6333)
            try:
                _ = self.client.get_collections()
            except ResponseHandlingException as e:
                self.logger.error(
                    "Qdrant container not running? For local dev you can run: %s",
                    "docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant:v1.6.1",
                )
                raise e
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'auto', 'memory', or 'disk'.")

    def _probe_connection(self, attempts: int = 4, base_delay: float = 1.0) -> None:
        """Verify connectivity with a lightweight get_collections() call.

        `get_collections()` is one of Qdrant's cheapest endpoints, so a failure
        here is almost always a transient transport blip (server restarting,
        momentary network hiccup) rather than a slow query. We retry such
        failures with exponential backoff so a brief outage doesn't abort an
        entire ingestion run at construction time. A non-transient error
        (e.g. UnexpectedResponse from a bad api key) is not caught here and
        surfaces immediately.
        """
        last_err: ResponseHandlingException | None = None
        for attempt in range(1, attempts + 1):
            try:
                _ = self.client.get_collections()
                return
            except ResponseHandlingException as e:
                last_err = e
                if attempt == attempts:
                    break
                delay = base_delay * (2 ** (attempt - 1))
                self.logger.warning(
                    "Qdrant connectivity probe failed (attempt %s/%s): %s. Retrying in %.1fs...",
                    attempt,
                    attempts,
                    e,
                    delay,
                )
                time.sleep(delay)
        self.logger.error("Qdrant connectivity probe failed after %s attempts.", attempts)
        raise last_err

    def as_llama_vector_store(self, collection_name) -> QdrantVectorStore:
        return QdrantVectorStore(client=self.client, collection_name=collection_name, max_retries=10)

    def create_collection(self, collection_name, vector_size, enable_sparse: bool = True) -> None:
        """Create a Qdrant collection with optional sparse vector support.
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of dense vectors
            enable_sparse: If True, enables sparse vectors for hybrid search
        """
        if self.client.collection_exists(collection_name=collection_name):
            self.logger.info("Qdrant collection '%s' already exists.", collection_name)
        else:
            self.logger.info(
                "Qdrant create_collection %s",
                {
                    "collection": collection_name,
                    "vector_size": vector_size,
                    "enable_sparse": enable_sparse,
                },
            )
            if enable_sparse:
                # Create collection with both dense and sparse vectors for hybrid retrieval
                _ = self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=vector_size,
                            # Normalized vectors are expected for cosine similiarity
                            # Ensure embeddings are normalized before upsert
                            distance=Distance.COSINE),
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            index=models.SparseIndexParams(
                                on_disk=True,
                            ),
                            # IMPORTANT: Use IDF weighting for sparse vectors to improve relevance in hybrid search
                            modifier=models.Modifier.IDF
                        ),
                    },
                    # quantize vectors to reduce storage and improve search speed, with minimal impact on relevance
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True
                        )
                    ),
                    on_disk_payload=True
                )
                # Create payload indexes for efficient filtering on 'source', 'course_id', and 'module_id'
                _ = self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="source",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

                _ = self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="course_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

                _ = self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="module_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

                print(f"Created hybrid collection '{collection_name}' with dense (size={vector_size}) and sparse vectors.")
            else:
                # Legacy: Dense-only collection
                _ = self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.DOT),
                )
                self.logger.info(
                    "Created dense-only collection '%s' with size=%s.",
                    collection_name,
                    vector_size,
                )

    def upsert(self, collection_name, points: list[dict]) -> None:
        """Upsert points into Qdrant collection.
        
        Supports both dense-only and hybrid (dense + sparse) vectors.
        
        Args:
            collection_name: Target collection name
            points: List of point dicts with 'id', 'vector', and 'payload'
                   'vector' can be:
                   - list[float] for dense-only collections
                   - dict with 'dense' and 'sparse' keys for hybrid collections
        """
        qdrant_points = [PointStruct(**point) for point in points]
        try:
            # Keep real multiprocessing (parallel=4), but pick a fork-safe start
            # method per platform. On macOS, both the default "fork" and Qdrant's
            # preferred "forkserver" fork from a parent that has already initialized
            # the Objective-C runtime, which crashes the workers
            # ("+[NSCharacterSet initialize] ... fork()"); they then never populate
            # the result queue and the parent raises _queue.Empty. "spawn" execs a
            # fresh interpreter and inherits no Obj-C state, so it is safe. On Linux
            # we let Qdrant default (method=None -> "forkserver"), which is faster.
            start_method = "spawn" if sys.platform == "darwin" else None
            operation_info = self.client.upload_points(
                collection_name=collection_name,
                points=qdrant_points,
                parallel=10,
                method=start_method,
                max_retries=3,
            )
        except ResponseHandlingException as e:
            # Z.B. httpx.RemoteProtocolError: "Server disconnected without sending a response".
            # Nur diesen Batch überspringen und weitermachen, statt den gesamten Lauf abzubrechen.
            self.logger.warning(
                "Qdrant upsert failed: collection=%s points=%s exc=%s",
                collection_name,
                len(points),
                e,
            )
            return

        self.logger.debug(
            "Upserted %s points into '%s' (operation=%s)",
            len(points),
            collection_name,
            operation_info,
        )

    def search(self, collection_name, query_vector, query_filter=None, with_payload=True, limit=10) -> list[dict]:
        """Search in Qdrant collection.
        
        Supports both dense-only and hybrid (dense + sparse) search.
        
        Args:
            collection_name: Collection to search in
            query_vector: Query vector (list[float] for dense-only, dict for hybrid)
            query_filter: Optional Qdrant filter
            with_payload: Include payload in results
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            with_payload=with_payload,
            limit=limit,
        )
        return search_result

    def get_course_module_records(self, collection_name):
        all_records = []

        next_page_offset = "first"
        offset = None

        while next_page_offset:
            if next_page_offset != "first":
                offset = next_page_offset

            try:
                records = self.client.scroll(
                    collection_name=collection_name,
                with_payload=True,
                    with_vectors=False,
                    limit=10,
                    offset=offset,
                )
            
            except ResponseHandlingException as e:
                self.logger.warning("Qdrant scroll ResponseHandlingException: %s", e)
                return [], []
            except Exception as e:
                self.logger.exception("Qdrant scroll unexpected exception: %s", e)
                return [], []

            next_page_offset = records[1]

            all_records.extend(records[0])

        courses_records = sorted(
            [
                record
                for record in all_records
                if "module_id" not in record.payload
                and "course_id" in record.payload
                and isinstance(record.payload["course_id"], int)
            ],
            key=lambda x: x.payload["course_id"],
        )
        modules_records = sorted(
            [record for record in all_records if "module_id" in record.payload], key=lambda x: x.payload["module_id"]
        )

        return courses_records, modules_records

    def check_if_course_exists(self, course_id: int) -> bool:
        """Check if a course exists in the database."""

        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="course_id",
                    match=models.MatchValue(value=course_id),
                ),
            ],
        )
        return bool(self.query_with_filter("web_assistant", scroll_filter))

    def check_if_module_exists(self, module_id: int) -> bool:
        """Check if a module exists in the database."""

        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="course_id",
                    match=models.MatchValue(value=module_id),
                ),
            ],
        )
        return bool(self.query_with_filter("web_assistant", scroll_filter))

    def query_with_filter(self, collection_name, scroll_filter) -> List:
        self.logger.debug("Qdrant scroll query on '%s' with filter=%s", collection_name, scroll_filter)
        records = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            with_payload=True,
            with_vectors=False,
            limit=10,
        )

        return records


    def delete_by_filter(self, collection_name: str, qdrant_filter: models.Filter, *, wait: bool = True) -> None:
        """Delete all points matching a payload filter.

        Note: For large deletes, Qdrant runs this as an internal operation. We still
        set wait=True by default so ingestion ordering is deterministic.
        """
        try:
            self.logger.info(
                "Qdrant delete_by_filter: collection=%s filter=%s",
                collection_name,
                qdrant_filter,
            )
            _ = self.client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(filter=qdrant_filter),
                wait=wait,
            )
        except ResponseHandlingException as e:
            self.logger.warning(
                "Qdrant delete_by_filter failed (collection=%s): %s",
                collection_name,
                e,
            )
        except Exception as e:
            self.logger.exception(
                "Qdrant delete_by_filter unexpected error (collection=%s): %s",
                collection_name,
                e,
            )
            raise


if __name__ == "__main__":
    test_connection = VectorDBQdrant(mode="disk")  # For local testing only
