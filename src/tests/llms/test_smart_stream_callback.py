"""Unit tests for SmartStreamCallback sentinel detection and streaming pipeline.

SmartStreamCallback buffers the first ~20 characters to detect the "NO ANSWER FOUND"
sentinel before committing to realtime streaming.  Tests verify the peek → realtime
state machine and that finalize() handles both the sentinel and normal paths correctly.
"""

import pytest

from src.llm.streaming import CitationStreamResolver, SmartStreamCallback


class TestSmartStreamCallback:
    """State-machine tests: peeking phase, realtime switch, flush, finalize."""

    # ── helpers ──────────────────────────────────────────────────────────────

    def _setup(self, resolve=None):
        """Return (ssc, resolver, emitted) with a plain pass-through resolver."""
        emitted: list[str] = []
        def callback(text: str) -> None:
            emitted.append(text)
        resolver = CitationStreamResolver(
            resolve=resolve or (lambda m: m),
            callback=callback,
        )
        ssc = SmartStreamCallback(resolver, outer_callback=lambda t: None)
        return ssc, resolver, emitted

    # ── peeking phase ─────────────────────────────────────────────────────────

    def test_short_feed_stays_buffered(self):
        """Tokens shorter than _THRESHOLD stay in the peek buffer — nothing emitted."""
        ssc, _, emitted = self._setup()
        ssc.feed("Hello")
        assert emitted == []
        assert ssc._peeking is True

    def test_multiple_short_feeds_stay_buffered_below_threshold(self):
        """Accumulating just under _THRESHOLD still does not emit."""
        ssc, _, emitted = self._setup()
        # _THRESHOLD = len("NO ANSWER FOUND") + 5 = 20
        for _ in range(3):
            ssc.feed("abc")  # 9 chars total, below 20
        assert emitted == []
        assert ssc._peeking is True

    def test_flush_while_peeking_emits_nothing(self):
        """flush() during peeking must not forward anything — sentinel may still come."""
        ssc, _, emitted = self._setup()
        ssc.feed("NO ANSWER")  # partial sentinel, peeking
        ssc.flush()
        assert emitted == []
        assert ssc._peeking is True

    # ── realtime switch ───────────────────────────────────────────────────────

    def test_exceeding_threshold_switches_to_realtime(self):
        """Once accumulated length > _THRESHOLD the callback receives text."""
        ssc, _, emitted = self._setup()
        ssc.feed("a" * 30)  # 30 > 20 → realtime
        assert ssc._peeking is False
        assert "".join(emitted) != ""

    def test_multiple_feeds_crossing_threshold_switch_to_realtime(self):
        """Crossing the threshold across multiple feed() calls switches correctly."""
        ssc, _, emitted = self._setup()
        for _ in range(7):
            ssc.feed("abcd")  # 28 chars > 20
        assert ssc._peeking is False
        assert len(emitted) > 0

    def test_realtime_subsequent_feed_emits_immediately(self):
        """After the switch, every subsequent feed() goes directly to the callback."""
        ssc, _, emitted = self._setup()
        ssc.feed("a" * 30)  # switch
        emitted.clear()
        ssc.feed("direct text")
        assert "direct text" in "".join(emitted)

    def test_flush_in_realtime_flushes_resolver_buffer(self):
        """flush() in realtime forwards any pending citation buffer from the resolver."""
        ssc, resolver, emitted = self._setup()
        ssc.feed("a" * 30)  # realtime
        emitted.clear()
        resolver._buffer = "[doc7"  # inject a partial citation
        ssc.flush()
        assert "[doc7" in "".join(emitted)

    # ── finalize: no-answer path ──────────────────────────────────────────────

    def test_finalize_no_answer_while_peeking_discards_buffer(self):
        """finalize(True) during peeking silently discards tokens — no callback."""
        ssc, _, emitted = self._setup()
        ssc.feed("NO ANSWER FOU")  # partial sentinel
        ssc.finalize(is_no_answer=True)
        assert emitted == []
        assert ssc._peek_buffer == []

    def test_finalize_no_answer_in_realtime_emits_nothing_extra(self):
        """finalize(True) in realtime mode neither errors nor emits extra text."""
        ssc, _, emitted = self._setup()
        ssc.feed("a" * 30)  # realtime
        emitted.clear()
        ssc.finalize(is_no_answer=True)
        assert emitted == []  # no additional emission

    def test_finalize_no_answer_clears_peek_buffer(self):
        """After finalize(True), the peek buffer is empty regardless of prior state."""
        ssc, _, _ = self._setup()
        ssc.feed("some text")
        ssc.finalize(is_no_answer=True)
        assert ssc._peek_buffer == []

    # ── finalize: normal path ─────────────────────────────────────────────────

    def test_finalize_normal_while_peeking_emits_buffered_text(self):
        """finalize(False) during peeking force-flushes the peek buffer through the resolver."""
        ssc, _, emitted = self._setup()
        ssc.feed("Hello World")  # stays buffered
        ssc.finalize(is_no_answer=False)
        assert "Hello World" in "".join(emitted)

    def test_finalize_normal_in_realtime_flushes_resolver_citation_buffer(self):
        """finalize(False) in realtime flushes any partial citation still in the resolver."""
        ssc, resolver, emitted = self._setup()
        ssc.feed("a" * 30)  # realtime
        resolver._buffer = "[doc5"
        ssc.finalize(is_no_answer=False)
        assert "[doc5" in "".join(emitted)

    def test_finalize_normal_switches_peeking_to_false(self):
        """After finalize(False) during peek, peeking flag is cleared."""
        ssc, _, _ = self._setup()
        ssc.feed("short")
        ssc.finalize(is_no_answer=False)
        assert ssc._peeking is False

    # ── citation handling in realtime ─────────────────────────────────────────

    def test_citation_resolved_in_realtime(self):
        """Citations are resolved via the resolve callable once in realtime mode."""
        resolved: list[str] = []
        def resolve(marker: str) -> str:
            resolved.append(marker)
            return "[Quelle]"

        emitted: list[str] = []
        resolver = CitationStreamResolver(resolve=resolve, callback=emitted.append)
        ssc = SmartStreamCallback(resolver, outer_callback=lambda t: None)

        ssc.feed("a" * 30)  # switch to realtime
        emitted.clear()
        resolved.clear()

        ssc.feed("[doc1]")
        assert "[doc1]" in resolved
        assert "[Quelle]" in "".join(emitted)

    def test_citation_dropped_in_realtime_when_resolve_returns_empty(self):
        """When resolve returns '', the marker is silently dropped from the stream."""
        emitted: list[str] = []
        resolver = CitationStreamResolver(resolve=lambda m: "", callback=emitted.append)
        ssc = SmartStreamCallback(resolver, outer_callback=lambda t: None)

        ssc.feed("a" * 30)
        emitted.clear()
        ssc.feed("[doc2]")
        assert "[doc2]" not in "".join(emitted)
