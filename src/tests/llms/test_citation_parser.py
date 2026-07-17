"""Unit tests for CitationParser and _get_display_title."""

import pytest
from llama_index.core.schema import TextNode

from src.llm.objects.citation_parser import CitationParser, _format_video_timestamp, _get_display_title, citation_suffix


def _node(title=None, fullname=None, url=None, start_seconds=None) -> TextNode:
    meta = {}
    if title is not None:
        meta["title"] = title
    if fullname is not None:
        meta["fullname"] = fullname
    if url is not None:
        meta["url"] = url
    if start_seconds is not None:
        meta["start_seconds"] = start_seconds
    return TextNode(text="content", metadata=meta)


# ---------------------------------------------------------------------------
# _get_display_title
# ---------------------------------------------------------------------------

class TestGetDisplayTitle:
    def test_uses_title_field(self):
        assert _get_display_title(_node(title="Deep Learning Basics")) == "Deep Learning Basics"

    def test_falls_back_to_fullname_when_no_title(self):
        assert _get_display_title(_node(fullname="Kurs XY")) == "Kurs XY"

    def test_title_takes_priority_over_fullname(self):
        assert _get_display_title(_node(title="Title", fullname="Fullname")) == "Title"

    def test_strips_leading_chapter_number(self):
        assert _get_display_title(_node(title="3.1. Einführung")) == "Einführung"

    def test_strips_single_digit_chapter(self):
        assert _get_display_title(_node(title="1. Basics")) == "Basics"

    def test_truncates_long_title_at_word_boundary(self):
        long = "Ein sehr langer Titel der definitiv mehr als fünfzig Zeichen umfasst und gekürzt werden sollte"
        result = _get_display_title(_node(title=long))
        assert result.endswith("...")
        assert len(result) <= 53  # 50 chars + "..."

    def test_short_title_not_truncated(self):
        short = "Kurzer Titel"
        assert _get_display_title(_node(title=short)) == short

    def test_falls_back_to_url_when_no_title(self):
        result = _get_display_title(_node(url="https://ki-campus.org/courses/intro"))
        assert "ki-campus.org" in result

    def test_truncates_long_url_path(self):
        result = _get_display_title(_node(url="https://ki-campus.org/" + "x" * 40))
        assert "..." in result

    def test_falls_back_to_quelle_when_nothing(self):
        assert _get_display_title(TextNode(text="", metadata={})) == "Quelle"


# ---------------------------------------------------------------------------
# CitationParser internals
# ---------------------------------------------------------------------------

class TestCleanUpAnswer:
    def setup_method(self):
        self.parser = CitationParser()

    def test_removes_double_spaces(self):
        assert self.parser._clean_up_answer("Hallo  Welt") == "Hallo Welt"

    def test_removes_float_doc_id(self):
        result = self.parser._clean_up_answer("Text [doc1.2] mehr")
        assert "[doc1.2]" not in result

    def test_keeps_valid_doc_id(self):
        result = self.parser._clean_up_answer("Text [doc3] mehr")
        assert "[doc3]" in result

    def test_removes_non_digit_doc_ref(self):
        result = self.parser._clean_up_answer("[docA]")
        assert "[docA]" not in result


class TestGetSourceDocsFromAnswer:
    def setup_method(self):
        self.parser = CitationParser()

    def test_extracts_single_id(self):
        assert self.parser._get_source_docs_from_answer("Text [doc2] text") == [2]

    def test_extracts_multiple_ids(self):
        assert self.parser._get_source_docs_from_answer("[doc1] und [doc3]") == [1, 3]

    def test_deduplicates_preserving_order(self):
        assert self.parser._get_source_docs_from_answer("[doc2] [doc1] [doc2]") == [2, 1]

    def test_returns_empty_for_no_refs(self):
        assert self.parser._get_source_docs_from_answer("Keine Referenzen") == []


# ---------------------------------------------------------------------------
# CitationParser.parse
# ---------------------------------------------------------------------------

class TestCitationParserParse:
    def setup_method(self):
        self.parser = CitationParser()

    def test_replaces_doc_ref_with_link(self):
        docs = [_node(title="Kurs A", url="https://ki-campus.org/a")]
        result = self.parser.parse("Siehe [doc1] für Details.", docs)
        assert "https://ki-campus.org/a" in result
        assert "Kurs A" in result
        assert "[doc1]" not in result

    def test_deduplicates_same_url(self):
        docs = [
            _node(title="Kurs A", url="https://ki-campus.org/a"),
            _node(title="Kurs A Kopie", url="https://ki-campus.org/a"),
        ]
        result = self.parser.parse("[doc1] und [doc2]", docs)
        # Second ref to same URL gets removed, not duplicated
        assert result.count("https://ki-campus.org/a") == 1

    def test_removes_hallucinated_ref(self):
        docs = [_node(title="Kurs A", url="https://ki-campus.org/a")]
        result = self.parser.parse("Text [doc5]", docs)
        assert "[doc5]" not in result

    def test_no_refs_in_answer_unchanged(self):
        docs = [_node(title="Kurs A", url="https://ki-campus.org/a")]
        result = self.parser.parse("Keine Quellen erwähnt.", docs)
        assert result == "Keine Quellen erwähnt."

    def test_removes_comma_space_before_ref(self):
        docs = [_node(title="Kurs A", url="https://ki-campus.org/a")]
        result = self.parser.parse("Text, [doc1]", docs)
        assert ", [doc1]" not in result

    def test_video_source_gets_deterministic_timestamp_suffix(self):
        """The timestamp is attached at render time — never relies on the LLM
        mentioning it in prose (small models don't follow that reliably)."""
        docs = [_node(title="Video: KI", url="https://moodle.ki-campus.org/mod/videotime/view.php?id=1", start_seconds=235.0)]
        result = self.parser.parse("Siehe [doc1].", docs)
        assert "(ab Minute 3:55)" in result
        assert "moodle.ki-campus.org" in result
        assert "vimeo.com" not in result and "youtube.com" not in result

    def test_non_video_source_gets_no_suffix(self):
        docs = [_node(title="Kurs A", url="https://ki-campus.org/a")]
        result = self.parser.parse("[doc1]", docs)
        assert "(ab Minute" not in result


# ---------------------------------------------------------------------------
# citation_suffix / _format_video_timestamp
# ---------------------------------------------------------------------------


class TestFormatVideoTimestamp:
    def test_seconds_under_a_minute(self):
        assert _format_video_timestamp(5) == "0:05"

    def test_minutes_and_seconds(self):
        assert _format_video_timestamp(235) == "3:55"

    def test_over_an_hour_includes_hours(self):
        assert _format_video_timestamp(3725) == "1:02:05"

    def test_truncates_fractional_seconds(self):
        assert _format_video_timestamp(90.9) == "1:30"


class TestCitationSuffix:
    def test_positive_start_seconds_produces_suffix(self):
        assert citation_suffix(_node(start_seconds=235.0)) == " (ab Minute 3:55)"

    def test_zero_start_seconds_produces_suffix(self):
        assert citation_suffix(_node(start_seconds=0)) == " (ab Minute 0:00)"

    def test_missing_start_seconds_produces_no_suffix(self):
        assert citation_suffix(_node()) == ""
