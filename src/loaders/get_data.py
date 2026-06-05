import logging
import os
import time
from datetime import datetime
from typing import List
import uuid
import gc

from qdrant_client.http import models
from tqdm import tqdm

from src.env import env
from src.llm.objects.LLMs import LLM
from fastembed import SparseTextEmbedding
from src.loaders.drupal import Drupal
from src.loaders.moochup import Moochup
from src.loaders.moodle import Moodle
from src.loaders.helper import iter_nodes_from_document_hierarchical
from src.vectordb.qdrant import VectorDBQdrant
from src.loaders.run_logger import Heartbeat, RunContext, RunLogger, StageTimer, Watchdog, format_kv

DEFAULT_COLLECTION = "web_assistant_hybrid_v2"
SNAPSHOTS_TO_KEEP = 3


# A full run takes about 2,5 hours (2025-02-11)
class Fetch_Data:
    def _embed_and_upsert_nodes(
        self,
        nodes: list,
        *,
        stage: str,
        calculate_point_size,
        batch_by_size,
        max_batch_bytes: int,
    ) -> int:
        """Embed a small batch of nodes and upsert them.

        Designed to be called with *small* `nodes` batches (e.g. 32-64) to keep
        memory bounded.
        """
        if not nodes:
            return 0

        texts_to_embed = [n.get_content() for n in nodes]
        dense_embeddings = self.embedder.get_text_embedding_batch(texts_to_embed)

        hybrid_points: list[dict] = []
        sparse_embeddings_batch = list(self.sparse_encoder.embed(texts_to_embed))
        
        for index, (node, dense_vec) in enumerate(zip(nodes, dense_embeddings)):
            text = node.get_content()
            sparse_result = sparse_embeddings_batch[index]
            sparse_dict = {
                "indices": sparse_result.indices.tolist(),
                "values": sparse_result.values.tolist()
            }
            hybrid_points.append(
                {
                    "id": node.node_id or str(uuid.uuid4()),
                    "vector": {"dense": dense_vec, "sparse": sparse_dict},
                    "payload": {"text": text, **node.metadata},
                }
            )

        points_upserted = 0
        for size_batch_idx, size_batch in enumerate(batch_by_size(hybrid_points, max_batch_bytes), start=1):
            batch_size_mb = sum(calculate_point_size(p) for p in size_batch) / (1024 * 1024)
            self.logger.info(
                "QDRANT_UPSERT_BEGIN %s",
                format_kv(
                    RUN_ID=self.run_id,
                    STAGE=stage,
                    COLLECTION=DEFAULT_COLLECTION,
                    BATCH_POINT_GROUP=size_batch_idx,
                    BATCH_POINTS=len(size_batch),
                    BATCH_MB=round(batch_size_mb, 2),
                ),
            )
            t_upsert = time.time()
            self.dev_vector_store.upsert(DEFAULT_COLLECTION, size_batch)
            self.logger.info(
                "QDRANT_UPSERT_END %s",
                format_kv(
                    RUN_ID=self.run_id,
                    STAGE=stage,
                    COLLECTION=DEFAULT_COLLECTION,
                    BATCH_POINT_GROUP=size_batch_idx,
                    BATCH_POINTS=len(size_batch),
                    ELAPSED_MS=int((time.time() - t_upsert) * 1000),
                ),
            )
            points_upserted += len(size_batch)

        # Help GC by dropping large structures promptly
        del hybrid_points
        del dense_embeddings
        del texts_to_embed
        return points_upserted

    def sanity_check(self):
        # Check if URLs are missing in metadata,
        # every point needs a non-empty url field in the metadata
        query_filter = models.Filter(must=[models.IsEmptyCondition(is_empty=models.PayloadField(key="url"))])

        if self.dev_vector_store.query_with_filter(DEFAULT_COLLECTION, query_filter) != ([], None):
            self.logger.error("Missing URLs in Metadata, linking to content not possible in all cases")

    def __init__(self, run_id: str | None = None, preset_log_url: str | None = None):
        self.DATA_PATH = "./data"
        self.embedder = LLM().get_embedder()
        self.sparse_encoder = SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")  # NEW: FastEmbed BM42 sparse encoder
        self.logger = logging.getLogger("loader")
        self.logger.propagate = False
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "{asctime} - {levelname:<8} - {message}",
                style="{",
                datefmt="%d-%b-%y %H:%M:%S",
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.DEBUG if env.DEBUG_MODE else logging.INFO)

        # Create a per-run ID so we can correlate stdout, blob log, and Qdrant state.
        self.run_id = run_id or RunLogger.new_run_id()
        self.run_logger = RunLogger(run_id=self.run_id, logger_name="loader")
        # Optional: append to Azure Append Blob if RUN_LOGS_BLOB_CONNECTION_STRING is set
        self.run_logger.attach_append_blob_handler()
        self.logger.info("Ingestion run_id=%s", self.run_id)

        # Shared run context for checkpoints/heartbeat.
        self.ctx = RunContext(self.run_id)
        self.heartbeat = Heartbeat(self.logger, self.ctx)
        # If start_ingest already created a SAS URL for this run, keep it.
        self.preset_log_url = preset_log_url

        self.dev_vector_store = VectorDBQdrant(mode="dev_remote")
        self.prod_vector_store = VectorDBQdrant(mode="prod_remote")

        self.logger.info("Starting data extraction...")

    def extract(
        self,
    ):
        # Fast diagnostic mode: restrict which sources are ingested.
        # Example: RUN_SOURCES=MOOCHUP,DRUPAL
        run_sources_raw = (os.getenv("RUN_SOURCES") or "").strip()
        run_sources = {s.strip().upper() for s in run_sources_raw.split(",") if s.strip()} if run_sources_raw else None
        if run_sources is not None:
            self.logger.warning(
                "RUN %s",
                format_kv(RUN_ID=self.run_id, EVENT="SOURCE_FILTER_ENABLED", RUN_SOURCES=",".join(sorted(run_sources))),
            )

        def chunk_list(lst, chunk_size):
            """Yield successive chunk_size-sized chunks from lst."""
            for i in range(0, len(lst), chunk_size):
                yield lst[i : i + chunk_size]

        def _doc_source(doc) -> str | None:
            try:
                md = getattr(doc, "metadata", None) or {}
                src = md.get("source")
                return str(src) if src is not None else None
            except Exception:
                return None

        def _doc_url(doc) -> str | None:
            try:
                md = getattr(doc, "metadata", None) or {}
                url = md.get("url")
                return str(url) if url is not None else None
            except Exception:
                return None

        def _batch_source_counts(docs: list) -> dict[str, int]:
            counts: dict[str, int] = {}
            for d in docs:
                s = _doc_source(d) or "UNKNOWN"
                counts[s] = counts.get(s, 0) + 1
            return counts

        # Qdrant payload size limit: 32MB, we target 30MB to be safe
        MAX_BATCH_SIZE_BYTES = 30 * 1024 * 1024  # 30 MB

        # Chunking defaults (token-aware). Tune via env vars.
        chunk_size_tokens = int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
        chunk_overlap_tokens = int(os.getenv("CHUNK_OVERLAP_TOKENS", "83"))
        if chunk_overlap_tokens >= chunk_size_tokens:
            self.logger.warning(
                "Invalid overlap >= size; adjusting overlap. %s",
                format_kv(SIZE=chunk_size_tokens, OVERLAP=chunk_overlap_tokens),
            )
            chunk_overlap_tokens = max(0, chunk_size_tokens // 6)

        # Chunking implementation lives in src/loaders/helper.py so we can reuse/test it.

        def calculate_point_size(point: dict) -> int:
            """Calculate the exact JSON payload size of a single point in bytes."""
            import json

            return len(json.dumps(point, default=str).encode("utf-8"))

        def batch_by_size(points: list, max_size_bytes: int):
            """Yield batches of points that fit within the size limit."""
            current_batch = []
            current_size = 0

            for point in points:
                point_size = calculate_point_size(point)

                # If a single point exceeds the limit, log warning and send it alone
                if point_size > max_size_bytes:
                    if current_batch:
                        yield current_batch
                        current_batch = []
                        current_size = 0
                    yield [point]  # Send oversized point alone
                    continue

                # Check if adding this point would exceed the limit
                if current_size + point_size > max_size_bytes:
                    yield current_batch
                    current_batch = [point]
                    current_size = point_size
                else:
                    current_batch.append(point)
                    current_size += point_size

            # Don't forget the last batch
            if current_batch:
                yield current_batch

        # Provide a read-only link to the run log (if blob logging is enabled)
        log_url = self.preset_log_url or self.run_logger.get_readonly_sas_url(expiry_hours=24)
        if log_url:
            self.logger.info("RUN %s", format_kv(RUN_ID=self.run_id, EVENT="LOG_URL", LOG_URL=log_url))

        # Start heartbeat ASAP, so we have liveness even if we hang early.
        self.heartbeat.start()
        self.logger.info(
            "RUN %s",
            format_kv(
                RUN_ID=self.run_id,
                EVENT="STARTED",
                DEFAULT_COLLECTION=DEFAULT_COLLECTION,
                DEBUG_MODE=getattr(env, "DEBUG_MODE", False),
            ),
        )
        self.ctx.checkpoint()

        started_ts = time.time()
        watchdog: Watchdog | None = None

        # Initialize for return values (ensures we can reference them in exception paths)
        total_points_upserted = 0
        moodle_courses_total = 0
        moodle_courses_done = 0
        moodle_documents = 0
        drupal_documents = 0
        moochup_documents = 0

        try:
            # Optional snapshots (can be slow/large). Controlled via env.
            if os.getenv("QDRANT_SNAPSHOTS_ENABLED", "true").lower() in {"1", "true", "yes"}:
                with StageTimer(self.logger, self.ctx, "SNAPSHOT"):
                    self.logger.info("Create Snapshot of previous data collection...")
                    if self.dev_vector_store.client.collection_exists(DEFAULT_COLLECTION):
                        _ = self.dev_vector_store.client.create_snapshot(collection_name=DEFAULT_COLLECTION, wait=False)

                        # There will likely be one additional snapshot because the snapshot created in the previous step has not yet been added to the list.
                        all_snapshots: List[models.SnapshotDescription] = self.dev_vector_store.client.list_snapshots(
                            collection_name=DEFAULT_COLLECTION
                        )
                        sorted_snapshots = self.sort_snapshots_by_creation_time(all_snapshots)
                        if len(all_snapshots) >= SNAPSHOTS_TO_KEEP:
                            for snapshot in sorted_snapshots[SNAPSHOTS_TO_KEEP:]:
                                self.logger.debug("Deleting snapshot %s", snapshot.name)
                                self.dev_vector_store.client.delete_snapshot(
                                    collection_name=DEFAULT_COLLECTION, snapshot_name=snapshot.name
                                )
            else:
                self.logger.info(
                    "QDRANT_SNAPSHOTS %s",
                    format_kv(RUN_ID=self.run_id, EVENT="SKIPPED", REASON="QDRANT_SNAPSHOTS_ENABLED=false"),
                )

            # Ensure collection exists (no more delete/recreate each run)
            with StageTimer(self.logger, self.ctx, "QDRANT_ENSURE_COLLECTION"):
                sample_embedding = self.embedder.get_text_embedding("test")
                embedding_dim = len(sample_embedding)
                self.logger.info("Detected embedding dimension: %s", embedding_dim)
                # Keep a cheap zero-vector around for metadata-only points (e.g., ModuleFingerprint)
                zero_dense_vec = [0.0] * embedding_dim
                self.dev_vector_store.create_collection(
                    collection_name=DEFAULT_COLLECTION,
                    vector_size=embedding_dim,
                    enable_sparse=True,
                )

            # Memory controls (important for Azure Functions ~2.5GB cap)
            embed_nodes_batch = int(os.getenv("EMBED_NODES_BATCH", "48"))
            gc_every = int(os.getenv("GC_EVERY_N_BATCHES", "10"))
            malloc_trim_enabled = os.getenv("MALLOC_TRIM_ENABLED", "true").lower() in {"1", "true", "yes"}
            log_rss_every = int(os.getenv("LOG_RSS_EVERY_N_BATCHES", "20"))

            self.logger.info(
                "MEMORY_TUNING %s",
                format_kv(
                    RUN_ID=self.run_id,
                    EMBED_NODES_BATCH=embed_nodes_batch,
                    GC_EVERY_N_BATCHES=gc_every,
                    MALLOC_TRIM_ENABLED=malloc_trim_enabled,
                    LOG_RSS_EVERY_N_BATCHES=log_rss_every,
                ),
            )

            # Track one course document per course (small; used for discoverability)
            ingested_course_summaries: set[int] = set()

            def _rss_mb() -> float | None:
                try:
                    import resource

                    # ru_maxrss is KB on Linux, bytes on macOS. Azure Functions is Linux.
                    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    return kb / 1024.0
                except Exception:
                    return None

            def _malloc_trim() -> None:
                if not malloc_trim_enabled:
                    return
                try:
                    import ctypes

                    libc = ctypes.CDLL("libc.so.6")
                    libc.malloc_trim(0)
                except Exception:
                    # best-effort; ignore
                    return

            def _maybe_cleanup(batch_counter: int) -> None:
                if gc_every > 0 and batch_counter % gc_every == 0:
                    gc.collect()
                    _malloc_trim()
                if log_rss_every > 0 and batch_counter % log_rss_every == 0:
                    rss = _rss_mb()
                    if rss is not None:
                        self.logger.info(
                            "MEMORY %s",
                            format_kv(RUN_ID=self.run_id, RSS_MB=round(rss, 1), BATCH=batch_counter),
                        )

            def _upsert_document_streaming(doc, *, stage: str) -> int:
                """Chunk/embed/upsert a single Document with strict bounded memory.

                Returns points upserted.
                """
                nonlocal total_points_upserted
                points_upserted = 0
                batch_counter = 0

                node_batch = []
                for node in iter_nodes_from_document_hierarchical(
                    doc,
                    chunk_size_tokens=chunk_size_tokens,
                    chunk_overlap_tokens=chunk_overlap_tokens,
                ):
                    node_batch.append(node)
                    if len(node_batch) < embed_nodes_batch:
                        continue

                    batch_counter += 1
                    points_upserted += self._embed_and_upsert_nodes(
                        node_batch,
                        stage=stage,
                        calculate_point_size=calculate_point_size,
                        batch_by_size=batch_by_size,
                        max_batch_bytes=MAX_BATCH_SIZE_BYTES,
                    )
                    node_batch = []
                    _maybe_cleanup(batch_counter)

                if node_batch:
                    batch_counter += 1
                    points_upserted += self._embed_and_upsert_nodes(
                        node_batch,
                        stage=stage,
                        calculate_point_size=calculate_point_size,
                        batch_by_size=batch_by_size,
                        max_batch_bytes=MAX_BATCH_SIZE_BYTES,
                    )
                    node_batch = []
                    _maybe_cleanup(batch_counter)

                total_points_upserted += points_upserted
                self.ctx.set_counter("points_upserted", total_points_upserted)
                return points_upserted

            def _upsert_documents(docs: list, *, stage: str, batch_docs: int = 50) -> int:
                """Chunk/embed/upsert docs. (legacy helper for non-Moodle sources)."""
                nonlocal total_points_upserted
                points_upserted = 0

                for batch_idx, batch in enumerate(tqdm(chunk_list(docs, batch_docs), desc=f"{stage} batches"), start=1):
                    self.ctx.checkpoint()
                    source_counts = _batch_source_counts(batch)
                    first_url = _doc_url(batch[0]) if batch else None
                    last_url = _doc_url(batch[-1]) if batch else None
                    self.logger.info(
                        "DOC_BATCH %s",
                        format_kv(
                            RUN_ID=self.run_id,
                            STAGE=stage,
                            EVENT="DOC_BATCH",
                            BATCH_DOC_INDEX=batch_idx,
                            DOCS_IN_BATCH=len(batch),
                            SOURCES=str(source_counts),
                            FIRST_URL=first_url,
                            LAST_URL=last_url,
                        ),
                    )
                    if batch:
                        self.ctx.set_last(url=first_url)

                    for d in batch:
                        points_upserted += _upsert_document_streaming(d, stage=stage)

                return points_upserted

            # Moochup: delete by source and upsert
            with StageTimer(self.logger, self.ctx, "MOOCHUP"):
                if run_sources is None or "MOOCHUP" in run_sources:
                    self.logger.info("Loading Moodle data from Moochup API...")
                    moochup_docs = Moochup(env.DATA_SOURCE_MOOCHUP_MOODLE_URL).get_course_documents()
                    moochup_documents = len(moochup_docs)
                    self.ctx.set_counter("moochup_documents", moochup_documents)
                    self.ctx.set_counter("moochup_courses", moochup_documents)

                    # Delete existing Moochup points
                    self.dev_vector_store.delete_by_filter(
                        DEFAULT_COLLECTION,
                        models.Filter(
                            must=[models.FieldCondition(key="source", match=models.MatchValue(value="Moochup"))]
                        ),
                    )
                    _upsert_documents(moochup_docs, stage="MOOCHUP_UPSERT")
                else:
                    self.logger.warning("Skipping MOOCHUP due to RUN_SOURCES filter")

            # Moodle: stream per course; delete+upsert per course_id
            moodle_watchdog_s = int(os.getenv("RUN_MOODLE_WATCHDOG_SECONDS", "1800"))
            watchdog = Watchdog(self.logger, self.ctx, "MOODLE", threshold_seconds=moodle_watchdog_s)
            if run_sources is None or "MOODLE" in run_sources:
                watchdog.start()
                with StageTimer(self.logger, self.ctx, "MOODLE"):
                    self.logger.info("Streaming Moodle data (per course)...")
                    moodle = Moodle(run_ctx=self.ctx)

                    # Best-effort: count total courses early for progress
                    try:
                        moodle_courses_total = len(moodle.get_courses())
                        self.ctx.set_counter("moodle_courses_total", moodle_courses_total)
                    except Exception:
                        moodle_courses_total = 0

                    prev_course_id: int | None = None
                    # Track per-course state to avoid duplicate deletes within a run
                    deleted_course_summary: set[int] = set()

                    for course, doc in moodle.iter_course_documents_stream():
                        course_id = int(getattr(course, "id", 0))

                        # Determine whether this is the per-course summary doc or a module doc.
                        module_id = None
                        if getattr(doc, "metadata", None):
                            module_id = doc.metadata.get("module_id")

                        # Course boundary cleanup (before starting the next course)
                        if prev_course_id is not None and course_id != prev_course_id:
                            gc.collect()
                            _malloc_trim()
                        prev_course_id = course_id

                        # Maintain a stable single course summary point per course by deleting old
                        # summaries and chunks before inserting the new ones.
                        if module_id is None and course_id not in deleted_course_summary:
                            deleted_course_summary.add(course_id)
                            moodle_courses_done += 1
                            self.ctx.set_counter("moodle_courses_done", moodle_courses_done)
                            self.dev_vector_store.delete_by_filter(
                                DEFAULT_COLLECTION,
                                models.Filter(
                                    must=[
                                        models.FieldCondition(key="source", match=models.MatchValue(value="Moodle")),
                                        models.FieldCondition(key="course_id", match=models.MatchValue(value=course_id)),
                                    ]
                                ),
                            )

                        # Skip per-course "Kurs" docs after first insert (avoids duplicates)
                        if module_id is None:
                            if course_id in ingested_course_summaries:
                                continue
                            ingested_course_summaries.add(course_id)

                        moodle_documents += 1
                        self.ctx.set_counter("moodle_documents", moodle_documents)
                        self.logger.info(
                            "MOODLE_DOC %s",
                            format_kv(
                                RUN_ID=self.run_id,
                                COURSE_ID=course_id,
                                MODULE_ID=module_id,
                                EVENT="UPSERT_DOC_BEGIN",
                            ),
                        )
                        _upsert_document_streaming(doc, stage="MOODLE_UPSERT")
                        self.logger.info(
                            "MOODLE_DOC %s",
                            format_kv(
                                RUN_ID=self.run_id,
                                COURSE_ID=course_id,
                                MODULE_ID=module_id,
                                EVENT="UPSERT_DOC_END",
                            ),
                        )

                        # Additional periodic cleanup handled by _maybe_cleanup and course-boundary cleanup above

                watchdog.stop()
                watchdog = None
            else:
                self.logger.warning("Skipping MOODLE due to RUN_SOURCES filter")

            # Drupal: delete by source and upsert
            with StageTimer(self.logger, self.ctx, "DRUPAL"):
                self.logger.info("Loading Drupal data from Drupal API...")
                if run_sources is None or "DRUPAL" in run_sources:
                    drupal_docs = Drupal(
                        base_url=env.DRUPAL_URL,
                        username=env.DRUPAL_USERNAME,
                        client_id=env.DRUPAL_CLIENT_ID,
                        client_secret=env.DRUPAL_CLIENT_SECRET,
                        grant_type=env.DRUPAL_GRANT_TYPE,
                        run_ctx=self.ctx,
                    ).extract()
                    drupal_documents = len(drupal_docs)
                    self.ctx.set_counter("drupal_documents", drupal_documents)
                    if drupal_documents == 0:
                        self.logger.warning("Drupal extraction returned 0 documents")

                    self.dev_vector_store.delete_by_filter(
                        DEFAULT_COLLECTION,
                        models.Filter(
                            must=[models.FieldCondition(key="source", match=models.MatchValue(value="Drupal"))]
                        ),
                    )
                    _upsert_documents(drupal_docs, stage="DRUPAL_UPSERT")
                else:
                    self.logger.warning("Skipping DRUPAL due to RUN_SOURCES filter")

            # NOTE: We no longer migrate DEV->PROD automatically in streaming mode.
            # Running two endpoints with migrate+recreate can wipe data. Handle promotion separately.
            self.logger.info("Finished incremental delete+upsert into Dev Qdrant.")

            with StageTimer(self.logger, self.ctx, "SANITY_CHECK"):
                self.sanity_check()

            elapsed_s = int(time.time() - started_ts)
            self.logger.info(
                "RUN %s",
                format_kv(
                    RUN_ID=self.run_id,
                    EVENT="COMPLETED",
                    ELAPSED_S=elapsed_s,
                    TOTAL_DOCUMENTS=(moochup_documents + moodle_documents + drupal_documents),
                    POINTS_UPSERTED=total_points_upserted,
                ),
            )

            return {
                "run_id": self.run_id,
                "log_url": log_url,
                "counts": {
                    "moochup_documents": moochup_documents,
                    "moodle_documents": moodle_documents,
                    "moodle_courses_total": moodle_courses_total,
                    "moodle_courses_done": moodle_courses_done,
                    "drupal_documents": drupal_documents,
                    "total_documents": (moochup_documents + moodle_documents + drupal_documents),
                    "points_upserted": total_points_upserted,
                },
                "status": "completed",
            }

        except Exception:
            elapsed_s = int(time.time() - started_ts)
            self.logger.exception(
                "RUN %s",
                format_kv(
                    RUN_ID=self.run_id,
                    EVENT="FAILED",
                    ELAPSED_S=elapsed_s,
                    STAGE=self.ctx.snapshot().get("stage"),
                ),
            )
            raise

        finally:
            try:
                if watchdog:
                    watchdog.stop()
            except Exception:
                pass

            try:
                self.heartbeat.stop()
            except Exception:
                pass

            try:
                self.run_logger.detach()
            except Exception:
                pass

    def sort_snapshots_by_creation_time(
        self, snapshots: List[models.SnapshotDescription]
    ) -> List[models.SnapshotDescription]:
        return sorted(
            snapshots,
            key=lambda snapshot: datetime.fromisoformat(snapshot.creation_time)
            if snapshot.creation_time
            else datetime.min,
            reverse=True,
        )


if __name__ == "__main__":
    Fetch_Data().extract()
