"""Unit tests for src/loaders/helper.py text normalization utilities.

Covers:
- normalize_text_for_rag: NBSP removal, zero-width chars, newline normalization,
  whitespace collapsing, blank-line reduction, NFC composition
- process_html_summaries: HTML tag stripping, BR to space, whitespace cleanup
- chunk_text_hierarchical: basic chunking behavior (empty input, short text, long text)
"""

import pytest

from src.loaders.helper import normalize_text_for_rag, process_html_summaries, chunk_text_hierarchical


class TestNormalizeTextForRag:
    def test_empty_string_returns_empty(self):
        assert normalize_text_for_rag("") == ""

    def test_none_returns_empty(self):
        assert normalize_text_for_rag(None) == ""

    def test_strips_nbsp(self):
        text = "Hallo Welt"
        result = normalize_text_for_rag(text)
        assert " " not in result
        assert "Hallo Welt" in result

    def test_removes_zero_width_spaces(self):
        text = "Hallo​Welt‌!"
        result = normalize_text_for_rag(text)
        assert "​" not in result
        assert "‌" not in result

    def test_removes_bom(self):
        text = "﻿Inhalt"
        result = normalize_text_for_rag(text)
        assert "﻿" not in result
        assert "Inhalt" in result

    def test_normalizes_crlf_to_lf(self):
        text = "Zeile 1\r\nZeile 2\r\nZeile 3"
        result = normalize_text_for_rag(text)
        assert "\r" not in result
        assert "Zeile 1\nZeile 2\nZeile 3" in result

    def test_collapses_multiple_blank_lines(self):
        text = "Absatz 1\n\n\n\n\nAbsatz 2"
        result = normalize_text_for_rag(text)
        assert "\n\n\n" not in result
        assert "Absatz 1" in result
        assert "Absatz 2" in result

    def test_strips_leading_and_trailing_whitespace(self):
        text = "   Inhalt   "
        result = normalize_text_for_rag(text)
        assert result == "Inhalt"

    def test_collapses_internal_multi_spaces(self):
        text = "Wort1   Wort2     Wort3"
        result = normalize_text_for_rag(text)
        assert "   " not in result
        assert "Wort1" in result
        assert "Wort2" in result

    def test_preserves_single_newlines(self):
        text = "Zeile 1\nZeile 2\nZeile 3"
        result = normalize_text_for_rag(text)
        assert "Zeile 1\nZeile 2\nZeile 3" == result

    def test_converts_tabs_to_spaces(self):
        text = "Wort1\tWort2"
        result = normalize_text_for_rag(text)
        assert "\t" not in result
        assert "Wort1" in result

    def test_nfc_composition_of_umlauts(self):
        # NFKD decomposes ü -> u + combining diaeresis; NFC should recompose
        import unicodedata
        text = "für"  # ü as composed
        result = normalize_text_for_rag(text)
        assert unicodedata.is_normalized("NFC", result)
        assert "ü" in result

    def test_plain_ascii_unchanged(self):
        text = "Simple ASCII text with no special characters."
        result = normalize_text_for_rag(text)
        assert result == text


class TestProcessHtmlSummaries:
    def test_removes_html_tags(self):
        html = "<p>Hallo <strong>Welt</strong></p>"
        result = process_html_summaries(html)
        assert "<" not in result
        assert "Hallo" in result
        assert "Welt" in result

    def test_converts_br_to_space(self):
        html = "Zeile 1<br>Zeile 2"
        result = process_html_summaries(html)
        assert "<br>" not in result
        assert "Zeile 1" in result
        assert "Zeile 2" in result

    def test_plain_text_returned_unchanged(self):
        text = "Kein HTML hier."
        result = process_html_summaries(text)
        assert "Kein HTML hier." in result

    def test_strips_html_entities(self):
        html = "<p>Test &amp; Inhalt</p>"
        result = process_html_summaries(html)
        assert "<" not in result

    def test_collapses_excess_whitespace(self):
        html = "<p>A   B   C</p>"
        result = process_html_summaries(html)
        assert "   " not in result


class TestChunkTextHierarchical:
    def test_empty_string_returns_empty_list(self):
        result = chunk_text_hierarchical("", chunk_size_tokens=100, chunk_overlap_tokens=10)
        assert result == []

    def test_short_text_returns_single_chunk(self):
        text = "Das ist ein kurzer Text."
        result = chunk_text_hierarchical(text, chunk_size_tokens=200, chunk_overlap_tokens=20)
        assert len(result) == 1
        assert "kurzer Text" in result[0]

    def test_long_text_is_split_into_multiple_chunks(self):
        sentence = "Dies ist ein Satz. " * 200
        result = chunk_text_hierarchical(sentence, chunk_size_tokens=50, chunk_overlap_tokens=5)
        assert len(result) > 1

    def test_chunks_do_not_exceed_size(self):
        from src.loaders.helper import count_tokens
        text = "Kurzer Satz. " * 300
        chunks = chunk_text_hierarchical(text, chunk_size_tokens=80, chunk_overlap_tokens=10)
        for chunk in chunks:
            assert count_tokens(chunk) <= 80 + 5  # small tolerance for edge cases

    def test_each_chunk_is_non_empty(self):
        text = "Satz eins.\n\nSatz zwei.\n\nSatz drei.\n\nSatz vier."
        chunks = chunk_text_hierarchical(text, chunk_size_tokens=20, chunk_overlap_tokens=0)
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_none_input_returns_empty_list(self):
        result = chunk_text_hierarchical(None, chunk_size_tokens=100, chunk_overlap_tokens=10)
        assert result == []
