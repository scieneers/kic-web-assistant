"""Unit tests for typed retrieval (retrieve_items/lookup_glossary) and scope ordering."""

import json
from unittest.mock import MagicMock

from src.api.models.serializable_text_node import SerializableTextNode
from src.llm.objects.retriever import KiCampusRetriever, _scope_sort_key


def _retriever_with_mocks() -> KiCampusRetriever:
    """Bypass __init__ (needs Azure credentials) — mirrors test_fetch_all.py."""
    retriever = object.__new__(KiCampusRetriever)
    retriever.use_hybrid = True
    retriever.n_chunks = 10
    retriever.use_semantic = False
    retriever.embedder = MagicMock()
    retriever.embedder.get_query_embedding.return_value = [0.0] * 4
    retriever.vector_db = MagicMock()
    retriever.index_name = "test-index"
    return retriever


def _raw_row(text: str, metadata: dict) -> dict:
    return {"id": "x", "text": text, "metadata_json": json.dumps(metadata)}


def _node(metadata: dict, text: str = "t") -> SerializableTextNode:
    return SerializableTextNode(text=text, metadata=metadata)


# ---------------------------------------------------------------------------
# _scope_sort_key
# ---------------------------------------------------------------------------


class TestScopeSortKey:
    def test_module_doc_sorts_before_its_items(self):
        module_doc = _node({"source_doc_key": "moodle:79:456", "chunk_index": 0})
        chapter = _node({"source_doc_key": "moodle:79:456:chapter:2", "item_order": 0})
        assert _scope_sort_key(module_doc) < _scope_sort_key(chapter)

    def test_items_sort_by_item_order_not_key_string(self):
        # key string sort would put chapter "10" before chapter "2"
        chapter_2 = _node({"source_doc_key": "moodle:79:456:chapter:2", "item_order": 0})
        chapter_10 = _node({"source_doc_key": "moodle:79:456:chapter:10", "item_order": 1})
        assert _scope_sort_key(chapter_2) < _scope_sort_key(chapter_10)

    def test_modules_stay_grouped(self):
        module_a_item = _node({"source_doc_key": "moodle:79:100:glossary:5", "item_order": 3})
        module_b_doc = _node({"source_doc_key": "moodle:79:200", "chunk_index": 0})
        assert _scope_sort_key(module_a_item) < _scope_sort_key(module_b_doc)


# ---------------------------------------------------------------------------
# retrieve_items / lookup_glossary
# ---------------------------------------------------------------------------


class TestRetrieveItems:
    def test_query_path_uses_doc_type_filter(self):
        retriever = _retriever_with_mocks()
        retriever.vector_db.hybrid_search.return_value = [_raw_row("q", {"type": "QuizItem"})]
        results = retriever.retrieve_items("QuizItem", course_id=79, module_id=7, query="Overfitting", top=5)
        odata_filter = retriever.vector_db.hybrid_search.call_args.kwargs["odata_filter"]
        assert "type eq 'QuizItem'" in odata_filter
        assert "course_id eq 79" in odata_filter
        assert len(results) == 1

    def test_fetch_path_sorts_by_item_order_and_caps(self):
        retriever = _retriever_with_mocks()
        retriever.vector_db.fetch_all.return_value = [
            _raw_row("second", {"source_doc_key": "moodle:79:7:quiz:b", "item_order": 1}),
            _raw_row("first", {"source_doc_key": "moodle:79:7:quiz:a", "item_order": 0}),
        ]
        results = retriever.retrieve_items("QuizItem", course_id=79, module_id=7)
        assert [n.text for n in results] == ["first", "second"]
        assert len(retriever.retrieve_items("QuizItem", course_id=79, module_id=7, top=1)) == 1
        assert len(retriever.retrieve_items("QuizItem", course_id=79, module_id=7, top=0)) == 2  # 0 = no cap


class TestLookupGlossary:
    def test_exact_concept_match_wins_over_rank(self):
        retriever = _retriever_with_mocks()
        semantic_hit = _node({"payload": {"concept": "Neuronales Netz"}}, text="Neuronales Netz: ...")
        exact_hit = _node({"payload": {"concept": "Neuron"}}, text="Neuron: ...")
        retriever.retrieve_items = MagicMock(return_value=[semantic_hit, exact_hit])
        results = retriever.lookup_glossary("neuron", course_id=79)
        assert results[0] is exact_hit

    def test_falls_back_to_semantic_order_without_exact_match(self):
        retriever = _retriever_with_mocks()
        first = _node({"payload": {"concept": "Gradient"}})
        retriever.retrieve_items = MagicMock(return_value=[first])
        assert retriever.lookup_glossary("Backpropagation")[0] is first
