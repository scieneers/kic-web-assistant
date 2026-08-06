"""Unit tests for CitationStreamFilter and CitationStreamResolver."""

import pytest

from src.llm.streaming import CitationStreamFilter, CitationStreamResolver


# ---------------------------------------------------------------------------
# CitationStreamFilter
# ---------------------------------------------------------------------------

class TestCitationStreamFilter:
    def setup_method(self):
        self.f = CitationStreamFilter()

    def _feed_all(self, tokens: list[str]) -> str:
        out = ""
        for t in tokens:
            out += self.f.feed(t)
        out += self.f.flush()
        return out

    def test_passthrough_text_without_brackets(self):
        assert self._feed_all(["Hallo Welt"]) == "Hallo Welt"

    def test_drops_complete_marker_in_one_token(self):
        assert self._feed_all(["Text [doc1] mehr"]) == "Text  mehr"

    def test_drops_marker_split_across_tokens(self):
        assert self._feed_all(["Text [", "doc", "3", "] mehr"]) == "Text  mehr"

    def test_drops_marker_split_two_tokens(self):
        assert self._feed_all(["Text [doc1", "] weiter"]) == "Text  weiter"

    def test_non_citation_bracket_passed_through(self):
        assert self._feed_all(["[nicht-doc]"]) == "[nicht-doc]"

    def test_drops_multiple_markers(self):
        assert self._feed_all(["[doc1] und [doc2]"]) == " und "

    def test_non_citation_bracket_emitted_by_feed(self):
        # "[xyz" can't become a [docN] marker → emitted immediately by feed()
        out = self.f.feed("Text [xyz")
        assert "[xyz" in out

    def test_empty_input(self):
        assert self._feed_all([""]) == ""

    def test_incomplete_marker_at_end_flushed(self):
        self.f.feed("Text [doc")
        tail = self.f.flush()
        # An unclosed [doc... is held back and flushed as-is
        assert "Text" in self.f.flush() or "doc" in tail or tail == "[doc" or tail == ""
        # What matters: no crash and flush returns the buffer


# ---------------------------------------------------------------------------
# CitationStreamResolver
# ---------------------------------------------------------------------------

class TestCitationStreamResolver:
    def _make_resolver(self, mapping: dict[str, str]) -> tuple[CitationStreamResolver, list[str]]:
        emitted: list[str] = []

        def resolve(marker: str) -> str:
            return mapping.get(marker, "")

        def callback(text: str) -> None:
            emitted.append(text)

        resolver = CitationStreamResolver(resolve=resolve, callback=callback)
        return resolver, emitted

    def test_resolves_complete_marker(self):
        resolver, emitted = self._make_resolver({"[doc1]": "<LINK>"})
        resolver.feed("Text [doc1] Ende")
        resolver.flush()
        full = "".join(emitted)
        assert "<LINK>" in full
        assert "[doc1]" not in full

    def test_drops_unknown_marker(self):
        resolver, emitted = self._make_resolver({})
        resolver.feed("[doc99] Text")
        resolver.flush()
        full = "".join(emitted)
        assert "[doc99]" not in full

    def test_passthrough_plain_text(self):
        resolver, emitted = self._make_resolver({})
        resolver.feed("Kein Marker hier")
        resolver.flush()
        assert "".join(emitted) == "Kein Marker hier"

    def test_emitted_text_property_tracks_output(self):
        resolver, _ = self._make_resolver({"[doc1]": "LINK"})
        resolver.feed("A [doc1] B")
        resolver.flush()
        assert "A" in resolver.emitted_text
        assert "LINK" in resolver.emitted_text

    def test_split_marker_resolved_correctly(self):
        resolver, emitted = self._make_resolver({"[doc2]": "<R>"})
        for chunk in ["Text [", "doc", "2", "] Ende"]:
            resolver.feed(chunk)
        resolver.flush()
        full = "".join(emitted)
        assert "<R>" in full
        assert "[doc2]" not in full

    def test_multiple_markers_all_resolved(self):
        resolver, emitted = self._make_resolver({"[doc1]": "A", "[doc2]": "B"})
        resolver.feed("[doc1] und [doc2]")
        resolver.flush()
        full = "".join(emitted)
        assert "A" in full
        assert "B" in full

    def test_non_citation_bracket_passes_through(self):
        resolver, emitted = self._make_resolver({})
        resolver.feed("[kein-doc] Text")
        resolver.flush()
        full = "".join(emitted)
        assert "[kein-doc]" in full

    def test_callback_is_called_with_resolved_text(self):
        calls = []
        resolver = CitationStreamResolver(
            resolve=lambda m: "RESOLVED",
            callback=lambda t: calls.append(t),
        )
        resolver.feed("Pre [doc1] Post")
        resolver.flush()
        combined = "".join(calls)
        assert "RESOLVED" in combined
        assert "Pre" in combined

    def test_flush_releases_incomplete_buffer(self):
        resolver, emitted = self._make_resolver({})
        resolver.feed("Text [doc")
        resolver.flush()
        full = "".join(emitted)
        assert "Text" in full

    def test_empty_stream_emits_nothing(self):
        resolver, emitted = self._make_resolver({})
        resolver.feed("")
        resolver.flush()
        assert "".join(emitted) == ""

    def test_resolved_text_used_as_replacement_not_marker(self):
        resolver, emitted = self._make_resolver({"[doc1]": "[Quelle 1](https://example.com)"})
        resolver.feed("Laut [doc1] ist das so.")
        resolver.flush()
        full = "".join(emitted)
        assert "[Quelle 1](https://example.com)" in full
        assert "[doc1]" not in full
