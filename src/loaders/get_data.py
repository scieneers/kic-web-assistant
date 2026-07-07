import hashlib
import json
import logging
import os
import time
import uuid
import gc


from src.env import env
from src.llm.objects.LLMs import LLM
from src.loaders.drupal import Drupal
from src.loaders.moochup import Moochup
from src.loaders.moodle import Moodle
from src.loaders.helper import iter_nodes_from_document_hierarchical
from src.vectordb.azure_search import VectorDBAzureSearch, sanitize_key
from src.loaders.run_logger import Heartbeat, RunContext, RunLogger, StageTimer, Watchdog, format_kv


def _source_doc_key(doc) -> str:
    """Stable identifier for a source document (pre-chunking)."""
    md = getattr(doc, "metadata", None) or {}
    source = md.get("source", "unknown")
    if source == "Drupal":
        return md.get("url") or f"drupal:{getattr(doc, 'doc_id', '')}"
    if source == "Moodle":
        course_id = md.get("course_id", "")
        module_id = md.get("module_id")
        suffix = str(module_id) if module_id is not None else "summary"
        return f"moodle:{course_id}:{suffix}"
    if source == "Moochup":
        return md.get("url") or f"moochup:{md.get('course_id', getattr(doc, 'doc_id', ''))}"
    return md.get("url") or f"{source}:{getattr(doc, 'doc_id', '')}"


def _content_hash(doc, chunk_size: int, chunk_overlap: int) -> str:
    """SHA-256 fingerprint of text + key metadata + chunking params (truncated to 16 hex chars).

    Including chunk_size/chunk_overlap means a parameter change automatically
    invalidates all existing hashes and triggers a full re-embed on the next run.
    """
    md = getattr(doc, "metadata", None) or {}
    stable = (
        (getattr(doc, "text", None) or "")
        + (md.get("title") or "")
        + (md.get("url") or "")
        + f"|cs={chunk_size}|co={chunk_overlap}"
    )
    return hashlib.sha256(stable.encode()).hexdigest()[:16]


def _odata_escape(value: str) -> str:
    """Escape a string value for embedding in an OData filter expression."""
    return value.replace("'", "''")


def _as_int(value) -> int | None:
    """Coerce a metadata value to int for an Edm.Int64 field, or None.

    Loaders normally provide int course/module ids, but a stray string would
    otherwise fail the whole upload batch. Unconvertible values fall back to
    None at the top level (the original is still preserved in metadata_json).
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# A full run takes about 2,5 hours (2025-02-11)
class Fetch_Data:
    @staticmethod
    def _build_document(node, text: str, md: dict, dense_vec: list[float]) -> dict:
        """Map a chunked node into an Azure AI Search document.

        Explicit, typed fields cover everything we filter or display on; the
        full original metadata is preserved verbatim in `metadata_json` so the
        retriever can reconstruct the node losslessly. Only declared index
        fields are included - Azure rejects unknown fields.
        """
        raw_id = node.node_id or str(uuid.uuid4())
        return {
            "id": sanitize_key(raw_id),
            "text": text,
            "source": md.get("source"),
            "type": md.get("type"),
            "course_id": _as_int(md.get("course_id")),
            "module_id": _as_int(md.get("module_id")),
            "url": md.get("url"),
            "fullname": md.get("fullname"),
            "title": md.get("title"),
            "is_important": md.get("is_important"),
            "date_created": md.get("date_created"),
            "metadata_json": json.dumps(md, default=str, ensure_ascii=False),
            "dense": dense_vec,
            "source_doc_key": md.get("source_doc_key"),
            "content_hash": md.get("content_hash"),
        }

    def _embed_and_upsert_nodes(self, nodes: list, *, stage: str) -> int:
        """Embed a small batch of nodes and upsert them into Azure AI Search.

        Designed to be called with *small* `nodes` batches (e.g. 32-64) to keep
        memory bounded. The store internally chunks the upload to Azure's batch
        limits (<=1000 docs and <=16 MB per request).
        """
        if not nodes:
            return 0

        texts_to_embed = [n.get_content() for n in nodes]
        t_embed = time.time()
        dense_embeddings = self.embedder.get_text_embedding_batch(texts_to_embed)
        self.logger.info(
            "EMBED_BATCH %s",
            format_kv(
                RUN_ID=self.run_id,
                STAGE=stage,
                NODES=len(texts_to_embed),
                ELAPSED_MS=int((time.time() - t_embed) * 1000),
            ),
        )

        documents: list[dict] = [
            self._build_document(node, node.get_content(), node.metadata or {}, dense_vec)
            for node, dense_vec in zip(nodes, dense_embeddings)
        ]

        self.logger.info(
            "AZURE_SEARCH_UPSERT_BEGIN %s",
            format_kv(RUN_ID=self.run_id, STAGE=stage, INDEX=self.index_name, DOCS=len(documents)),
        )
        t_upsert = time.time()
        uploaded = self.search_store.upload_documents(self.index_name, documents)
        self.logger.info(
            "AZURE_SEARCH_UPSERT_END %s",
            format_kv(
                RUN_ID=self.run_id,
                STAGE=stage,
                INDEX=self.index_name,
                DOCS=uploaded,
                ELAPSED_MS=int((time.time() - t_upsert) * 1000),
            ),
        )

        # Help GC by dropping large structures promptly
        del documents
        del dense_embeddings
        del texts_to_embed
        return uploaded

    def sanity_check(self):
        # Every document needs a non-empty url so we can link back to content.
        if self.search_store.any_match("url eq null or url eq ''", index_name=self.index_name):
            self.logger.error("Missing URLs in Metadata, linking to content not possible in all cases")

    def __init__(self, run_id: str | None = None, preset_log_url: str | None = None, index_name: str | None = None):
        self.DATA_PATH = "./data"
        self.embedder = LLM().get_embedder()
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

        # Create a per-run ID so we can correlate stdout, blob log, and index state.
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

        self.search_store = VectorDBAzureSearch()
        self.index_name = index_name or env.AZURE_SEARCH_INDEX

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
        # Azure AI Search batch limits (<=1000 docs and <=16 MB/request) are
        # enforced inside VectorDBAzureSearch.upload_documents, so the loader no
        # longer needs its own size-based batching.

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
                INDEX=self.index_name,
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
        moodle_skipped = 0
        moodle_stale_deleted = 0
        drupal_documents = 0
        drupal_skipped = 0
        drupal_stale_deleted = 0
        moochup_documents = 0
        moochup_skipped = 0
        moochup_stale_deleted = 0

        try:
            # Ensure the index exists. It is created once with a fixed vector
            # dimension; thereafter we only delete+upsert (no recreate per run).
            # Backups are an infra concern, not part of the loader.
            with StageTimer(self.logger, self.ctx, "AZURE_SEARCH_ENSURE_INDEX"):
                sample_embedding = self.embedder.get_text_embedding("test")
                embedding_dim = len(sample_embedding)
                self.logger.info("Detected embedding dimension: %s", embedding_dim)
                self.search_store.create_index(
                    index_name=self.index_name,
                    vector_size=embedding_dim,
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

            # Shared node buffer: chunked nodes are accumulated ACROSS documents and
            # only embedded/upserted once embed_nodes_batch nodes are queued. This keeps
            # embedding batches full even when individual documents are tiny (common for
            # Moodle modules), while staying bounded in memory: the buffer never holds
            # more than one batch worth of nodes.
            node_buffer: list = []
            batch_state = {"counter": 0}

            def _flush_node_buffer(*, stage: str) -> int:
                """Embed + upsert everything currently queued in node_buffer.

                Returns points upserted; no-op (returns 0) when the buffer is empty.
                """
                nonlocal total_points_upserted
                if not node_buffer:
                    return 0
                batch_state["counter"] += 1
                points_upserted = self._embed_and_upsert_nodes(node_buffer, stage=stage)
                node_buffer.clear()
                _maybe_cleanup(batch_state["counter"])
                total_points_upserted += points_upserted
                self.ctx.set_counter("points_upserted", total_points_upserted)
                return points_upserted

            def _queue_document(doc, *, stage: str) -> None:
                """Chunk a document and append its nodes to the shared buffer.

                Flushes whenever the buffer reaches embed_nodes_batch, but does NOT
                force a flush at the document boundary - so a single batch can span
                multiple documents. Callers must invoke _flush_node_buffer() at the end
                of a stage to drain the remainder.
                """
                for node in iter_nodes_from_document_hierarchical(
                    doc,
                    chunk_size_tokens=chunk_size_tokens,
                    chunk_overlap_tokens=chunk_overlap_tokens,
                ):
                    node_buffer.append(node)
                    if len(node_buffer) >= embed_nodes_batch:
                        _flush_node_buffer(stage=stage)

            # Moochup: hash-based delta upsert
            with StageTimer(self.logger, self.ctx, "MOOCHUP"):
                if run_sources is None or "MOOCHUP" in run_sources:
                    self.logger.info("Loading Moochup data...")
                    moochup_docs = Moochup(env.DATA_SOURCE_MOOCHUP_MOODLE_URL).get_course_documents()
                    moochup_documents = len(moochup_docs)
                    self.ctx.set_counter("moochup_documents", moochup_documents)
                    self.ctx.set_counter("moochup_courses", moochup_documents)

                    moochup_existing = self.search_store.load_content_hashes("Moochup", self.index_name)
                    # Migration guard: old chunks (pre-hash) have no source_doc_key and are
                    # invisible to the stale cleanup. Delete them once so the index starts clean.
                    if not moochup_existing and self.search_store.any_match(
                        "source eq 'Moochup' and source_doc_key eq null", index_name=self.index_name
                    ):
                        self.logger.info(
                            "MIGRATION_GUARD %s",
                            format_kv(RUN_ID=self.run_id, SOURCE="Moochup", EVENT="DELETE_OLD_SCHEMA"),
                        )
                        self.search_store.delete_by_filter(self.index_name, "source eq 'Moochup'")
                    moochup_seen: set[str] = set()

                    for doc in moochup_docs:
                        self.ctx.checkpoint()
                        key = _source_doc_key(doc)
                        new_hash = _content_hash(doc, chunk_size_tokens, chunk_overlap_tokens)
                        moochup_seen.add(key)
                        doc.metadata["source_doc_key"] = key
                        doc.metadata["content_hash"] = new_hash

                        if key not in moochup_existing:
                            _queue_document(doc, stage="MOOCHUP_UPSERT")
                        elif moochup_existing[key] != new_hash:
                            self.search_store.delete_by_filter(
                                self.index_name, f"source_doc_key eq '{_odata_escape(key)}'"
                            )
                            _queue_document(doc, stage="MOOCHUP_UPSERT")
                        else:
                            moochup_skipped += 1

                    _flush_node_buffer(stage="MOOCHUP_UPSERT")

                    stale = moochup_existing.keys() - moochup_seen
                    for key in stale:
                        self.search_store.delete_by_filter(
                            self.index_name, f"source_doc_key eq '{_odata_escape(key)}'"
                        )
                    moochup_stale_deleted = len(stale)

                    self.logger.info(
                        "MOOCHUP_DELTA %s",
                        format_kv(
                            RUN_ID=self.run_id,
                            TOTAL=moochup_documents,
                            SKIPPED=moochup_skipped,
                            STALE_DELETED=moochup_stale_deleted,
                        ),
                    )
                else:
                    self.logger.warning("Skipping MOOCHUP due to RUN_SOURCES filter")

            # Moodle: stream per course; hash-based delta per module
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

                    moodle_existing = self.search_store.load_content_hashes("Moodle", self.index_name)
                    if not moodle_existing and self.search_store.any_match(
                        "source eq 'Moodle' and source_doc_key eq null", index_name=self.index_name
                    ):
                        self.logger.info(
                            "MIGRATION_GUARD %s",
                            format_kv(RUN_ID=self.run_id, SOURCE="Moodle", EVENT="DELETE_OLD_SCHEMA"),
                        )
                        self.search_store.delete_by_filter(self.index_name, "source eq 'Moodle'")
                    moodle_seen: set[str] = set()
                    prev_course_id: int | None = None

                    for course, doc in moodle.iter_course_documents_stream():
                        course_id = int(getattr(course, "id", 0))
                        module_id = doc.metadata.get("module_id") if doc.metadata else None

                        # GC at course boundaries to keep memory bounded
                        if prev_course_id is not None and course_id != prev_course_id:
                            gc.collect()
                            _malloc_trim()
                        prev_course_id = course_id

                        key = _source_doc_key(doc)
                        new_hash = _content_hash(doc, chunk_size_tokens, chunk_overlap_tokens)

                        # Skip duplicates within the same run (course summary may appear multiple times)
                        if key in moodle_seen:
                            continue
                        moodle_seen.add(key)

                        if module_id is None:
                            moodle_courses_done += 1
                            self.ctx.set_counter("moodle_courses_done", moodle_courses_done)

                        doc.metadata["source_doc_key"] = key
                        doc.metadata["content_hash"] = new_hash

                        if key not in moodle_existing:
                            moodle_documents += 1
                            self.ctx.set_counter("moodle_documents", moodle_documents)
                            _queue_document(doc, stage="MOODLE_UPSERT")
                            self.logger.info(
                                "MOODLE_DOC %s",
                                format_kv(RUN_ID=self.run_id, COURSE_ID=course_id, MODULE_ID=module_id, EVENT="DOC_NEW"),
                            )
                        elif moodle_existing[key] != new_hash:
                            moodle_documents += 1
                            self.ctx.set_counter("moodle_documents", moodle_documents)
                            self.search_store.delete_by_filter(
                                self.index_name, f"source_doc_key eq '{_odata_escape(key)}'"
                            )
                            _queue_document(doc, stage="MOODLE_UPSERT")
                            self.logger.info(
                                "MOODLE_DOC %s",
                                format_kv(RUN_ID=self.run_id, COURSE_ID=course_id, MODULE_ID=module_id, EVENT="DOC_UPDATED"),
                            )
                        else:
                            moodle_skipped += 1
                            self.logger.debug(
                                "MOODLE_DOC %s",
                                format_kv(RUN_ID=self.run_id, COURSE_ID=course_id, MODULE_ID=module_id, EVENT="DOC_SKIPPED"),
                            )

                    # Drain any nodes still buffered from the last course(s).
                    _flush_node_buffer(stage="MOODLE_UPSERT")

                    # Clean up modules/courses that no longer exist in Moodle.
                    # Skip when MOODLE_COURSE_OFFSET is set: skipped courses are absent
                    # from moodle_seen but still valid in the index — deleting them would
                    # wipe data we intentionally did not re-process this run.
                    moodle_course_offset = int(os.getenv("MOODLE_COURSE_OFFSET", "0"))
                    moodle_stale = moodle_existing.keys() - moodle_seen
                    if moodle_course_offset > 0:
                        self.logger.warning(
                            "MOODLE_STALE_SKIP %s",
                            format_kv(
                                RUN_ID=self.run_id,
                                REASON="MOODLE_COURSE_OFFSET set",
                                WOULD_DELETE=len(moodle_stale),
                            ),
                        )
                        moodle_stale_deleted = 0
                    else:
                        for key in moodle_stale:
                            self.search_store.delete_by_filter(
                                self.index_name, f"source_doc_key eq '{_odata_escape(key)}'"
                            )
                        moodle_stale_deleted = len(moodle_stale)

                    self.logger.info(
                        "MOODLE_DELTA %s",
                        format_kv(
                            RUN_ID=self.run_id,
                            TOTAL_SEEN=len(moodle_seen),
                            SKIPPED=moodle_skipped,
                            STALE_DELETED=moodle_stale_deleted,
                        ),
                    )

                watchdog.stop()
                watchdog = None
            else:
                self.logger.warning("Skipping MOODLE due to RUN_SOURCES filter")

            # Drupal: hash-based delta upsert
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

                    drupal_existing = self.search_store.load_content_hashes("Drupal", self.index_name)
                    if not drupal_existing and self.search_store.any_match(
                        "source eq 'Drupal' and source_doc_key eq null", index_name=self.index_name
                    ):
                        self.logger.info(
                            "MIGRATION_GUARD %s",
                            format_kv(RUN_ID=self.run_id, SOURCE="Drupal", EVENT="DELETE_OLD_SCHEMA"),
                        )
                        self.search_store.delete_by_filter(self.index_name, "source eq 'Drupal'")
                    drupal_seen: set[str] = set()

                    for doc in drupal_docs:
                        self.ctx.checkpoint()
                        key = _source_doc_key(doc)
                        new_hash = _content_hash(doc, chunk_size_tokens, chunk_overlap_tokens)
                        drupal_seen.add(key)
                        doc.metadata["source_doc_key"] = key
                        doc.metadata["content_hash"] = new_hash

                        if key not in drupal_existing:
                            _queue_document(doc, stage="DRUPAL_UPSERT")
                        elif drupal_existing[key] != new_hash:
                            self.search_store.delete_by_filter(
                                self.index_name, f"source_doc_key eq '{_odata_escape(key)}'"
                            )
                            _queue_document(doc, stage="DRUPAL_UPSERT")
                        else:
                            drupal_skipped += 1

                    _flush_node_buffer(stage="DRUPAL_UPSERT")

                    stale = drupal_existing.keys() - drupal_seen
                    for key in stale:
                        self.search_store.delete_by_filter(
                            self.index_name, f"source_doc_key eq '{_odata_escape(key)}'"
                        )
                    drupal_stale_deleted = len(stale)

                    self.logger.info(
                        "DRUPAL_DELTA %s",
                        format_kv(
                            RUN_ID=self.run_id,
                            TOTAL=drupal_documents,
                            SKIPPED=drupal_skipped,
                            STALE_DELETED=drupal_stale_deleted,
                        ),
                    )
                else:
                    self.logger.warning("Skipping DRUPAL due to RUN_SOURCES filter")

            # NOTE: We no longer migrate DEV->PROD automatically in streaming mode.
            # Running two endpoints with migrate+recreate can wipe data. Handle promotion separately.
            self.logger.info("Finished incremental delete+upsert into Azure AI Search.")

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
                    TOTAL_SKIPPED=(moochup_skipped + moodle_skipped + drupal_skipped),
                    TOTAL_STALE_DELETED=(moochup_stale_deleted + moodle_stale_deleted + drupal_stale_deleted),
                    POINTS_UPSERTED=total_points_upserted,
                ),
            )

            return {
                "run_id": self.run_id,
                "log_url": log_url,
                "counts": {
                    "moochup_documents": moochup_documents,
                    "moochup_skipped": moochup_skipped,
                    "moochup_stale_deleted": moochup_stale_deleted,
                    "moodle_documents": moodle_documents,
                    "moodle_skipped": moodle_skipped,
                    "moodle_stale_deleted": moodle_stale_deleted,
                    "moodle_courses_total": moodle_courses_total,
                    "moodle_courses_done": moodle_courses_done,
                    "drupal_documents": drupal_documents,
                    "drupal_skipped": drupal_skipped,
                    "drupal_stale_deleted": drupal_stale_deleted,
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


if __name__ == "__main__":
    Fetch_Data().extract()
