"""Integration tests for the Drupal content loader.

These tests hit the live ki-campus.org JSONAPI and require internet access.
No Azure credentials needed — the Drupal API is publicly readable.

Run with: pytest src/tests/loaders/test_drupal_integration.py -m integration -v
"""

import pytest
from llama_index.core import Document

from src.loaders.drupal import Drupal, PageTypes

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def drupal() -> Drupal:
    return Drupal()


@pytest.fixture(scope="module")
def course_docs(drupal: Drupal) -> list[Document]:
    """Fetched once and shared across all course tests."""
    return drupal.get_page_type(PageTypes.COURSE)


@pytest.fixture(scope="module")
def blogpost_docs(drupal: Drupal) -> list[Document]:
    """Fetched once and shared across all blogpost tests."""
    return drupal.get_page_type(PageTypes.BLOGPOST)


class TestDrupalCoursePages:
    def test_fetches_at_least_one_course(self, course_docs):
        assert len(course_docs) > 0

    def test_course_documents_are_document_objects(self, course_docs):
        for doc in course_docs[:5]:
            assert isinstance(doc, Document)

    def test_course_document_has_required_metadata(self, course_docs):
        assert len(course_docs) > 0
        doc = course_docs[0]
        assert "title" in doc.metadata
        assert "source" in doc.metadata
        assert doc.metadata["source"] == "Drupal"
        assert "type" in doc.metadata
        assert "url" in doc.metadata

    def test_course_document_has_non_empty_text(self, course_docs):
        non_empty = [d for d in course_docs if d.text and len(d.text) > 0]
        assert len(non_empty) > 0

    def test_course_url_contains_ki_campus(self, course_docs):
        for doc in course_docs[:5]:
            assert "ki-campus.org" in doc.metadata["url"]


class TestDrupalBlogPosts:
    def test_fetches_at_least_one_blogpost(self, blogpost_docs):
        assert len(blogpost_docs) > 0

    def test_blogpost_type_in_metadata(self, blogpost_docs):
        for doc in blogpost_docs[:3]:
            assert doc.metadata["type"] == "blogpost"

    def test_blogpost_has_title(self, blogpost_docs):
        for doc in blogpost_docs[:3]:
            assert len(doc.metadata.get("title", "")) > 0
