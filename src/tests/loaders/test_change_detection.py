"""Unit tests for the hash-based change detection logic in get_data.py."""

import hashlib
import logging
import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import Document

from src.loaders.get_data import Fetch_Data, _content_hash, _odata_escape, _source_doc_key, _stale_deletion_allowed
from src.vectordb.azure_search import VectorDBAzureSearch


# ---------------------------------------------------------------------------
# _source_doc_key
# ---------------------------------------------------------------------------


class TestSourceDocKey:
    def test_drupal_uses_url(self):
        doc = Document(text="x", metadata={"source": "Drupal", "url": "https://ki-campus.org/node/42"})
        assert _source_doc_key(doc) == "https://ki-campus.org/node/42"

    def test_drupal_fallback_without_url(self):
        doc = Document(text="x", metadata={"source": "Drupal"})
        key = _source_doc_key(doc)
        assert key.startswith("drupal:")

    def test_moodle_module(self):
        doc = Document(text="x", metadata={"source": "Moodle", "course_id": 123, "module_id": 456})
        assert _source_doc_key(doc) == "moodle:123:456"

    def test_moodle_course_summary_no_module_id(self):
        doc = Document(text="x", metadata={"source": "Moodle", "course_id": 123})
        assert _source_doc_key(doc) == "moodle:123:summary"

    def test_moodle_course_summary_explicit_none(self):
        doc = Document(text="x", metadata={"source": "Moodle", "course_id": 99, "module_id": None})
        assert _source_doc_key(doc) == "moodle:99:summary"

    def test_moochup_uses_url(self):
        doc = Document(text="x", metadata={"source": "Moochup", "url": "https://moocub.org/courses/abc"})
        assert _source_doc_key(doc) == "https://moocub.org/courses/abc"

    def test_moochup_fallback_to_course_id(self):
        doc = Document(text="x", metadata={"source": "Moochup", "course_id": "xyz-123"})
        key = _source_doc_key(doc)
        assert "xyz-123" in key

    def test_key_is_stable_across_calls(self):
        doc = Document(text="x", metadata={"source": "Moodle", "course_id": 10, "module_id": 20})
        assert _source_doc_key(doc) == _source_doc_key(doc)

    def test_different_modules_different_keys(self):
        doc1 = Document(text="x", metadata={"source": "Moodle", "course_id": 1, "module_id": 100})
        doc2 = Document(text="x", metadata={"source": "Moodle", "course_id": 1, "module_id": 200})
        assert _source_doc_key(doc1) != _source_doc_key(doc2)


# ---------------------------------------------------------------------------
# _content_hash
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_returns_16_hex_chars(self):
        doc = Document(text="hello world", metadata={"source": "Drupal", "url": "https://example.com"})
        h = _content_hash(doc, 500, 83)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        doc = Document(text="same content", metadata={"source": "Drupal", "url": "https://x.com", "title": "T"})
        assert _content_hash(doc, 500, 83) == _content_hash(doc, 500, 83)

    def test_text_change_changes_hash(self):
        meta = {"source": "Drupal", "url": "https://x.com", "title": "T"}
        doc1 = Document(text="original", metadata=meta)
        doc2 = Document(text="changed", metadata=meta)
        assert _content_hash(doc1, 500, 83) != _content_hash(doc2, 500, 83)

    def test_title_change_changes_hash(self):
        doc1 = Document(text="body", metadata={"source": "Drupal", "url": "https://x.com", "title": "Old"})
        doc2 = Document(text="body", metadata={"source": "Drupal", "url": "https://x.com", "title": "New"})
        assert _content_hash(doc1, 500, 83) != _content_hash(doc2, 500, 83)

    def test_url_change_changes_hash(self):
        doc1 = Document(text="body", metadata={"source": "Drupal", "url": "https://x.com/1", "title": "T"})
        doc2 = Document(text="body", metadata={"source": "Drupal", "url": "https://x.com/2", "title": "T"})
        assert _content_hash(doc1, 500, 83) != _content_hash(doc2, 500, 83)

    def test_chunk_size_change_changes_hash(self):
        doc = Document(text="body", metadata={"source": "Drupal", "url": "https://x.com", "title": "T"})
        assert _content_hash(doc, 500, 83) != _content_hash(doc, 250, 83)

    def test_chunk_overlap_change_changes_hash(self):
        doc = Document(text="body", metadata={"source": "Drupal", "url": "https://x.com", "title": "T"})
        assert _content_hash(doc, 500, 83) != _content_hash(doc, 500, 50)

    def test_missing_metadata_does_not_raise(self):
        doc = Document(text="body", metadata={})
        h = _content_hash(doc, 500, 83)
        assert len(h) == 16

    def test_hash_matches_manual_sha256(self):
        doc = Document(text="abc", metadata={"source": "Moodle", "url": "https://u.com", "title": "T"})
        stable = "abc" + "T" + "https://u.com" + "|cs=500|co=83"
        expected = hashlib.sha256(stable.encode()).hexdigest()[:16]
        assert _content_hash(doc, 500, 83) == expected


# ---------------------------------------------------------------------------
# _odata_escape
# ---------------------------------------------------------------------------


class TestOdataEscape:
    def test_no_quotes_unchanged(self):
        assert _odata_escape("https://ki-campus.org/node/42") == "https://ki-campus.org/node/42"

    def test_single_quote_doubled(self):
        assert _odata_escape("it's") == "it''s"

    def test_multiple_quotes(self):
        assert _odata_escape("a'b'c") == "a''b''c"

    def test_empty_string(self):
        assert _odata_escape("") == ""

    def test_moodle_key_unchanged(self):
        assert _odata_escape("moodle:123:456") == "moodle:123:456"


# ---------------------------------------------------------------------------
# VectorDBAzureSearch.load_content_hashes
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_search_store():
    """VectorDBAzureSearch with Azure connections patched out."""
    with patch("src.vectordb.azure_search.DefaultAzureCredential"), \
         patch("src.vectordb.azure_search.SearchIndexClient"), \
         patch.object(VectorDBAzureSearch, "__init__", lambda self: None):
        store = VectorDBAzureSearch.__new__(VectorDBAzureSearch)
        store.logger = MagicMock()
        store.endpoint = "https://fake.search.windows.net"
        store.index_name = "test-index"
        store.credential = MagicMock()
        store._search_clients = {}
        return store


class _FakeSearchResults:
    """Mimics the subset of Azure's SearchItemPaged our code relies on."""

    def __init__(self, items, true_total):
        self._items = items
        self._true_total = true_total

    def __iter__(self):
        return iter(self._items)

    def get_count(self):
        return self._true_total


def _make_capped_client(items):
    """Fake SearchClient enforcing Azure's real `top` semantics.

    `top` caps the TOTAL rows returned across the whole call (not a page
    size) — a `top` below `len(items)` must truncate. This is what makes a
    test using this fixture actually catch a `top=1000`-style regression
    instead of silently ignoring the kwarg like a naive fixed-return mock.
    `get_count()` always reports the true (uncapped) total, matching a real
    `top=0, include_total_count=True` count query.
    """
    true_total = len(items)

    def _search(*, top=None, **kwargs):
        if top is not None and top < len(items):
            return _FakeSearchResults(items[:top], true_total)
        return _FakeSearchResults(items, true_total)

    client = MagicMock()
    client.search.side_effect = _search
    return client


class TestLoadContentHashes:
    def _make_client(self, items):
        return _make_capped_client(items)

    def test_returns_empty_dict_for_empty_index(self, mock_search_store):
        mock_search_store._search_clients["test-index"] = self._make_client([])
        result = mock_search_store.load_content_hashes("Drupal")
        assert result == {}

    def test_deduplicates_chunks_with_same_key(self, mock_search_store):
        items = [
            {"source_doc_key": "https://x.com/1", "content_hash": "abc123"},
            {"source_doc_key": "https://x.com/1", "content_hash": "abc123"},  # same doc, second chunk
            {"source_doc_key": "https://x.com/2", "content_hash": "def456"},
        ]
        mock_search_store._search_clients["test-index"] = self._make_client(items)
        result = mock_search_store.load_content_hashes("Drupal")
        assert result == {"https://x.com/1": "abc123", "https://x.com/2": "def456"}

    def test_ignores_chunks_without_key(self, mock_search_store):
        items = [
            {"source_doc_key": None, "content_hash": "abc"},
            {"source_doc_key": "https://x.com/1", "content_hash": "abc123"},
        ]
        mock_search_store._search_clients["test-index"] = self._make_client(items)
        result = mock_search_store.load_content_hashes("Drupal")
        assert result == {"https://x.com/1": "abc123"}

    def test_ignores_chunks_without_hash(self, mock_search_store):
        items = [
            {"source_doc_key": "https://x.com/1", "content_hash": None},
            {"source_doc_key": "https://x.com/2", "content_hash": "def456"},
        ]
        mock_search_store._search_clients["test-index"] = self._make_client(items)
        result = mock_search_store.load_content_hashes("Drupal")
        assert result == {"https://x.com/2": "def456"}

    def test_filters_by_source(self, mock_search_store):
        client = self._make_client([])
        mock_search_store._search_clients["test-index"] = client
        mock_search_store.load_content_hashes("Moodle")
        call_kwargs = client.search.call_args
        assert "source eq 'Moodle'" in str(call_kwargs)

    def test_returns_empty_on_http_error(self, mock_search_store):
        from azure.core.exceptions import HttpResponseError
        client = MagicMock()
        client.search.side_effect = HttpResponseError(message="boom")
        mock_search_store._search_clients["test-index"] = client
        result = mock_search_store.load_content_hashes("Drupal")
        assert result == {}

    def test_no_top_kwarg_passed_to_scan(self, mock_search_store):
        """Regression guard: a `top` on the scan call would silently cap the
        total rows returned (see test_returns_all_entries_beyond_1000_row_cap)."""
        client = self._make_client([{"source_doc_key": "k", "content_hash": "h"}])
        mock_search_store._search_clients["test-index"] = client
        mock_search_store.load_content_hashes("Drupal")
        scan_call = client.search.call_args_list[0]
        assert scan_call.kwargs.get("top") is None

    def test_returns_all_entries_beyond_1000_row_cap(self, mock_search_store):
        items = [
            {"source_doc_key": f"key-{i}", "content_hash": f"hash-{i}"}
            for i in range(1500)
        ]
        mock_search_store._search_clients["test-index"] = self._make_client(items)
        result = mock_search_store.load_content_hashes("Moodle")
        assert len(result) == 1500

    def test_logs_error_when_scan_is_incomplete(self, mock_search_store):
        # Simulate a client that still truncates despite no `top` being passed
        # (e.g. a server-side change) — the sanity check must catch this.
        client = MagicMock()
        client.search.side_effect = [
            iter([{"source_doc_key": "key-0", "content_hash": "h"}]),
            _FakeSearchResults([], true_total=1500),
        ]
        mock_search_store._search_clients["test-index"] = client
        mock_search_store.load_content_hashes("Moodle")
        mock_search_store.logger.error.assert_called_once()
        logged = str(mock_search_store.logger.error.call_args)
        assert "1" in logged and "1500" in logged

    def test_no_error_logged_when_scan_is_complete(self, mock_search_store):
        items = [{"source_doc_key": f"key-{i}", "content_hash": "h"} for i in range(5)]
        mock_search_store._search_clients["test-index"] = self._make_client(items)
        mock_search_store.load_content_hashes("Moodle")
        mock_search_store.logger.error.assert_not_called()


# ---------------------------------------------------------------------------
# VectorDBAzureSearch.delete_by_filter / _iter_keys
# ---------------------------------------------------------------------------


class TestDeleteByFilter:
    def test_deletes_all_matching_beyond_1000_row_cap(self, mock_search_store):
        keys = [{"id": f"doc-{i}"} for i in range(1500)]
        client = _make_capped_client(keys)
        mock_search_store._search_clients["test-index"] = client
        deleted = mock_search_store.delete_by_filter("test-index", "source eq 'Moodle'")
        assert deleted == 1500
        # Batched at <=1000 docs per Azure limit: 1500 keys -> 2 delete_documents calls.
        assert client.delete_documents.call_count == 2

    def test_no_top_kwarg_passed_to_key_scan(self, mock_search_store):
        client = _make_capped_client([{"id": "doc-1"}])
        mock_search_store._search_clients["test-index"] = client
        mock_search_store.delete_by_filter("test-index", "source eq 'Drupal'")
        scan_call = client.search.call_args_list[0]
        assert scan_call.kwargs.get("top") is None

    def test_logs_error_when_key_scan_is_incomplete(self, mock_search_store):
        client = MagicMock()
        client.search.side_effect = [
            iter([{"id": "doc-0"}]),
            _FakeSearchResults([], true_total=1500),
        ]
        mock_search_store._search_clients["test-index"] = client
        mock_search_store.delete_by_filter("test-index", "source eq 'Moodle'")
        mock_search_store.logger.error.assert_called_once()

    def test_returns_partial_count_on_http_error(self, mock_search_store):
        from azure.core.exceptions import HttpResponseError
        client = MagicMock()
        client.search.side_effect = HttpResponseError(message="boom")
        mock_search_store._search_clients["test-index"] = client
        deleted = mock_search_store.delete_by_filter("test-index", "source eq 'Drupal'")
        assert deleted == 0


# ---------------------------------------------------------------------------
# _stale_deletion_allowed
# ---------------------------------------------------------------------------


class TestStaleDeletionAllowed:
    def _call(self, seen: int, existing: int, stale: int, logger=None) -> bool:
        return _stale_deletion_allowed(logger or MagicMock(), "test-run", "Drupal", seen, existing, stale)

    def test_nothing_stale_is_always_allowed(self):
        # Even a run that saw nothing may proceed if there is nothing to delete.
        assert self._call(seen=0, existing=0, stale=0) is True
        assert self._call(seen=0, existing=100, stale=0) is True

    def test_full_fetch_allows_deletion(self):
        assert self._call(seen=100, existing=100, stale=5) is True

    def test_empty_fetch_refuses_deletion(self):
        # The core scenario: API succeeds but returns nothing (e.g. Drupal
        # without credentials) — everything would be stale.
        assert self._call(seen=0, existing=100, stale=100) is False

    def test_partial_fetch_below_threshold_refuses(self):
        assert self._call(seen=40, existing=100, stale=60) is False

    def test_fetch_at_threshold_allows(self):
        assert self._call(seen=50, existing=100, stale=50) is True

    def test_growth_run_allows(self):
        # More documents seen than indexed (new content) is never suspicious.
        assert self._call(seen=150, existing=100, stale=2) is True

    def test_threshold_configurable_via_env(self, monkeypatch):
        monkeypatch.setenv("STALE_DELETE_MIN_SEEN_RATIO", "0.9")
        assert self._call(seen=80, existing=100, stale=20) is False
        monkeypatch.setenv("STALE_DELETE_MIN_SEEN_RATIO", "0.3")
        assert self._call(seen=40, existing=100, stale=60) is True

    def test_refusal_logs_error_with_counts(self):
        logger = MagicMock()
        assert self._call(seen=0, existing=100, stale=100, logger=logger) is False
        logger.error.assert_called_once()
        logged = str(logger.error.call_args)
        assert "STALE_DELETE_REFUSED" in logged
        assert "WOULD_DELETE=100" in logged

    def test_allowed_path_does_not_log_error(self):
        logger = MagicMock()
        assert self._call(seen=100, existing=100, stale=3, logger=logger) is True
        logger.error.assert_not_called()


# ---------------------------------------------------------------------------
# Fetch_Data._build_document — chunk id scheme
# ---------------------------------------------------------------------------


class TestBuildDocument:
    def _node(self, node_id="fallback-uuid"):
        return SimpleNamespace(node_id=node_id)

    def _md(self, **overrides):
        md = {"source_doc_key": "moodle:1:2", "chunk_index": 0, "source": "Moodle"}
        md.update(overrides)
        return md

    def test_deterministic_for_same_key_and_index(self):
        doc1 = Fetch_Data._build_document(self._node(), "text", self._md(), [0.0])
        doc2 = Fetch_Data._build_document(self._node(), "text", self._md(), [0.0])
        assert doc1["id"] == doc2["id"]

    def test_different_chunk_index_different_id(self):
        doc0 = Fetch_Data._build_document(self._node(), "text", self._md(chunk_index=0), [0.0])
        doc1 = Fetch_Data._build_document(self._node(), "text", self._md(chunk_index=1), [0.0])
        assert doc0["id"] != doc1["id"]

    def test_different_source_doc_key_different_id(self):
        doc1 = Fetch_Data._build_document(self._node(), "text", self._md(source_doc_key="moodle:1:2"), [0.0])
        doc2 = Fetch_Data._build_document(self._node(), "text", self._md(source_doc_key="moodle:1:3"), [0.0])
        assert doc1["id"] != doc2["id"]

    def test_ids_survive_reprocessing_run(self):
        """The scenario that caused the duplicate-chunk bug: re-processing an
        unchanged document must upsert over the same ids, not mint new ones."""
        run1 = [Fetch_Data._build_document(self._node(), "text", self._md(chunk_index=i), [0.0]) for i in range(5)]
        run2 = [Fetch_Data._build_document(self._node(), "text", self._md(chunk_index=i), [0.0]) for i in range(5)]
        assert [d["id"] for d in run1] == [d["id"] for d in run2]

    def test_missing_source_doc_key_falls_back_to_node_id(self):
        md = self._md(source_doc_key=None)
        doc = Fetch_Data._build_document(self._node("fallback-uuid"), "text", md, [0.0])
        assert doc["id"] == "fallback-uuid"

    def test_missing_source_doc_key_logs_warning(self, caplog):
        md = self._md(source_doc_key=None)
        with caplog.at_level(logging.WARNING, logger="loader"):
            Fetch_Data._build_document(self._node("fallback-uuid"), "text", md, [0.0])
        assert "non-deterministic id" in caplog.text

    def test_missing_chunk_index_falls_back(self):
        md = self._md(chunk_index=None)
        doc = Fetch_Data._build_document(self._node("fallback-uuid-2"), "text", md, [0.0])
        assert doc["id"] == "fallback-uuid-2"

    def test_id_contains_only_valid_azure_key_characters(self):
        md = self._md(source_doc_key="https://example.com/path?x=1&y=2")
        doc = Fetch_Data._build_document(self._node(), "text", md, [0.0])
        assert re.fullmatch(r"[A-Za-z0-9_\-=]+", doc["id"])
