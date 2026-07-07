"""Unit tests for the hash-based change detection logic in get_data.py."""

import hashlib
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import Document

from src.loaders.get_data import _content_hash, _odata_escape, _source_doc_key, _stale_deletion_allowed
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


class TestLoadContentHashes:
    def _make_client(self, items):
        client = MagicMock()
        client.search.return_value = iter(items)
        return client

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
        client = MagicMock()
        client.search.return_value = iter([])
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
