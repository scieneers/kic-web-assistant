"""Unit tests for VectorDBAzureSearch.fetch_all — used by the summarize feature
to pull a complete course/module scope instead of a top-k search."""

from unittest.mock import MagicMock

from src.vectordb.azure_search import VectorDBAzureSearch


def _instance(client: MagicMock) -> VectorDBAzureSearch:
    """Bypass __init__ (which needs Azure credentials) — mirrors the pattern
    used in test_reranker_node.py::TestHybridSearchSemanticWindow."""
    instance = object.__new__(VectorDBAzureSearch)
    instance.index_name = "idx"
    instance.logger = MagicMock()
    instance._client = lambda name=None: client
    return instance


class TestFetchAll:
    def test_returns_all_rows_no_top_cap(self):
        client = MagicMock()
        rows = [{"id": str(i), "text": f"t{i}"} for i in range(5)]

        def _search(**kwargs):
            if kwargs.get("include_total_count"):
                result = MagicMock()
                result.get_count.return_value = 5
                return result
            return iter(rows)

        client.search.side_effect = _search

        db = _instance(client)
        result = db.fetch_all("module_id eq 1")

        assert len(result) == 5
        search_kwargs = client.search.call_args_list[0].kwargs
        assert "top" not in search_kwargs
        assert search_kwargs["filter"] == "module_id eq 1"
        assert search_kwargs["search_text"] == "*"

    def test_logs_error_when_scan_incomplete(self):
        client = MagicMock()

        def _search(**kwargs):
            if kwargs.get("include_total_count"):
                result = MagicMock()
                result.get_count.return_value = 10
                return result
            return iter([{"id": "1", "text": "t"}])  # only 1 of 10 scanned

        client.search.side_effect = _search
        db = _instance(client)

        result = db.fetch_all("course_id eq 1")

        assert len(result) == 1
        db.logger.error.assert_called_once()

    def test_no_error_logged_when_scan_complete(self):
        client = MagicMock()

        def _search(**kwargs):
            if kwargs.get("include_total_count"):
                result = MagicMock()
                result.get_count.return_value = 2
                return result
            return iter([{"id": "1", "text": "t"}, {"id": "2", "text": "t2"}])

        client.search.side_effect = _search
        db = _instance(client)

        db.fetch_all("course_id eq 1")

        db.logger.error.assert_not_called()
