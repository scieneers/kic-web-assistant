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


# ---------------------------------------------------------------------------
# CitationParser.parse — numbered inline links + source list (feedback p. 17)
# ---------------------------------------------------------------------------

A = "https://ki-campus.org/a"
B = "https://ki-campus.org/b"


def _body_and_footer(result: str) -> tuple[str, str]:
    body, _, footer = result.partition("\n\n**")
    return body, footer


class TestCitationParserParse:
    def setup_method(self):
        self.parser = CitationParser()

    def test_inline_number_links_to_source(self):
        docs = [_node(title="Kurs A", url=A)]
        result = self.parser.parse("Siehe [doc1] für Details.", docs, language="German")
        body, footer = _body_and_footer(result)
        assert f'[<a href="{A}">1</a>]' in body
        assert "Kurs A" not in body  # the title lives only in the list
        assert "[doc1]" not in result

    def test_source_list_below_answer(self):
        docs = [_node(title="Kurs A", url=A), _node(title="Kurs B", url=B)]
        result = self.parser.parse("Erstens [doc2]. Zweitens [doc1].", docs, language="German")
        body, footer = _body_and_footer(result)
        assert body == f'Erstens[<a href="{B}">1</a>]. Zweitens[<a href="{A}">2</a>].'
        assert footer == (
            "Zum Weiterlesen:**\n"
            f'1. <a href="{B}">Kurs B</a>\n'
            f'2. <a href="{A}">Kurs A</a>'
        )

    def test_heading_in_answer_language(self):
        docs = [_node(title="Kurs A", url=A)]
        assert "**Further reading:**" in self.parser.parse("See [doc1].", docs, language="English")
        assert "**Zum Weiterlesen:**" in self.parser.parse("Siehe [doc1].", docs, language="Deutsch")

    def test_unknown_language_falls_back_to_english_heading(self):
        docs = [_node(title="Kurs A", url=A)]
        assert "**Further reading:**" in self.parser.parse("Voir [doc1].", docs, language="French")

    def test_same_url_reuses_number_and_is_listed_once(self):
        docs = [_node(title="Kurs A", url=A), _node(title="Kurs A Kopie", url=A)]
        result = self.parser.parse("Erst [doc1]. Später [doc2].", docs, language="German")
        body, footer = _body_and_footer(result)
        assert body.count(f'[<a href="{A}">1</a>]') == 2
        assert footer.count(A) == 1

    def test_adjacent_markers_to_same_page_render_once(self):
        docs = [_node(title="Kurs A", url=A), _node(title="Kurs A Kopie", url=A), _node(title="Kurs B", url=B)]
        body, _ = _body_and_footer(self.parser.parse("Satz [doc1][doc2][doc3].", docs, language="German"))
        assert body == f'Satz[<a href="{A}">1</a>][<a href="{B}">2</a>].'

    def test_removes_hallucinated_ref_without_list(self):
        docs = [_node(title="Kurs A", url=A)]
        result = self.parser.parse("Text [doc5]", docs, language="German")
        assert result == "Text"

    def test_no_refs_in_answer_unchanged(self):
        docs = [_node(title="Kurs A", url=A)]
        result = self.parser.parse("Keine Quellen erwähnt.", docs, language="German")
        assert result == "Keine Quellen erwähnt."

    def test_already_resolved_streamed_answer_unchanged(self):
        """Streaming already rendered numbers + list; the citation node must not touch it."""
        docs = [_node(title="Kurs A", url=A)]
        streamed = self.parser.parse("Siehe [doc1].", docs, language="German")
        assert self.parser.parse(streamed, docs, language="German") == streamed

    def test_removes_comma_space_before_ref(self):
        docs = [_node(title="Kurs A", url=A)]
        result = self.parser.parse("Text, [doc1]", docs)
        assert ", [doc1]" not in result
        assert result.startswith(f'Text[<a href="{A}">1</a>]')

    def test_video_source_gets_deterministic_timestamp_suffix(self):
        """The timestamp is attached at render time — never relies on the LLM
        mentioning it in prose (small models don't follow that reliably)."""
        docs = [_node(title="Video: KI", url="https://moodle.ki-campus.org/mod/videotime/view.php?id=1", start_seconds=235.0)]
        result = self.parser.parse("Siehe [doc1].", docs)
        _, footer = _body_and_footer(result)
        assert "Video: KI</a> (ab Minute 3:55)" in footer
        assert "moodle.ki-campus.org" in result
        assert "vimeo.com" not in result and "youtube.com" not in result

    def test_non_video_source_gets_no_suffix(self):
        docs = [_node(title="Kurs A", url=A)]
        result = self.parser.parse("[doc1]", docs)
        assert "(ab Minute" not in result


class TestSourceListStreamingParity:
    """The streaming path (CitationStreamResolver + SourceList) must produce
    exactly what the non-streaming parser produces, chunk boundaries aside."""

    def test_stream_matches_parser(self):
        from src.llm.objects.citation_parser import SourceList
        from src.llm.streaming import CitationStreamResolver

        docs = [_node(title="Kurs A", url=A), _node(title="Kurs A Kopie", url=A), _node(title="Kurs B", url=B)]
        answer = "Satz eins.[doc1][doc2] Satz zwei.[doc3] Nochmal.[doc2]"
        emitted: list[str] = []
        source_list = SourceList(docs)
        resolver = CitationStreamResolver(resolve=source_list.resolve, callback=emitted.append)
        for i in range(0, len(answer), 3):
            resolver.feed(answer[i : i + 3])
        resolver.flush()
        streamed = resolver.emitted_text + source_list.footer("German")
        assert streamed == CitationParser().parse(answer, docs, language="German")


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


class TestListTitleLength:
    def test_list_keeps_titles_up_to_100_chars(self):
        title = "Skript Video: Künstliche Intelligenz – Grundlagen, Begriffe und Geschichte"  # 75 chars
        result = CitationParser().parse("[doc1]", [_node(title=title, url="https://x/a")], language="German")
        assert f">{title}</a>" in result

    def test_very_long_list_title_still_truncated(self):
        title = "Wort " * 40
        result = CitationParser().parse("[doc1]", [_node(title=title, url="https://x/a")], language="German")
        shown = result.split('">', 2)[-1].split("</a>")[0]
        assert shown.endswith("...") and len(shown) <= 100
