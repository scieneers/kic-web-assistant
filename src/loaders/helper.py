import re
import unicodedata
from io import StringIO
from pathlib import Path
from typing import Iterable

import webvtt
from bs4 import BeautifulSoup
from llama_index.core.schema import TextNode
from llama_index.core.utils import get_tokenizer

TMP_DIR = Path(__file__).parent.parent.parent.joinpath("tmp").resolve()


def convert_vtt_to_text(vtt_buffer: StringIO) -> str:
    try:
        vtt = webvtt.read_buffer(vtt_buffer)
    except webvtt.MalformedFileError as err:
        raise err
    transcript = ""

    lines = []
    for line in vtt:
        # Strip the newlines from the end of the text.
        # Split the string if it has a newline in the middle
        # Add the lines to an array
        lines.extend(line.text.strip().splitlines())

    # Remove repeated lines
    previous = None
    for line in lines:
        if line == previous:
            continue
        transcript += " " + line
        previous = line

    return transcript


def _vtt_timestamp_to_seconds(timestamp: str) -> float:
    """'HH:MM:SS.mmm' (or 'MM:SS.mmm') → seconds."""
    parts = timestamp.split(":")
    parts = ["0"] * (3 - len(parts)) + parts
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def convert_vtt_to_segments(vtt_buffer: StringIO) -> list[dict]:
    """Parse WebVTT into timed segments: [{"start_seconds": float, "text": str}].

    Companion to convert_vtt_to_text that PRESERVES the cue start times —
    Transcript documents carry them per chunk so citations can deep-link into
    the video ("#t=..."). Consecutive duplicate caption lines are dropped the
    same way as in the flat conversion.
    """
    vtt = webvtt.read_buffer(vtt_buffer)
    segments: list[dict] = []
    previous_line = None
    last_start = 0.0
    for cue in vtt:
        kept_lines = []
        for line in cue.text.strip().splitlines():
            line = line.strip()
            if not line or line == previous_line:
                continue
            kept_lines.append(line)
            previous_line = line
        if not kept_lines:
            continue
        try:
            start = _vtt_timestamp_to_seconds(cue.start)
            last_start = start
        except (ValueError, AttributeError):
            start = last_start
        segments.append({"start_seconds": start, "text": " ".join(kept_lines)})
    return segments


def process_html_summaries(text: str) -> str:
    """remove html tags from summaries, beautify poorly formatted texts"""
    if "<" not in text:
        new_text = text.strip()
    else:
        soup = BeautifulSoup(text, "html.parser")
        for br_tag in soup.find_all("br"):
            br_tag.replace_with(" ")
        new_text = soup.get_text().strip()
    new_text = new_text.replace("\n", " ").replace("\r", "")
    # \r = carriage return character
    new_text = re.sub(r"\s{3,}", "  ", new_text)

    # Normalize parsed text (remove \xa0 from str)
    new_text = unicodedata.normalize("NFKD", new_text)
    return new_text


_ZERO_WIDTH_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF]")


def normalize_text_for_rag(text: str | None) -> str:
    """Normalize extracted text for chunking + storage.

    Goals:
    - remove common HTML-to-text artifacts (NBSP/zero-width chars)
    - normalize newlines and whitespace while keeping paragraph boundaries
    - keep list-like formatting reasonably intact (one item per line)

    This function is intentionally conservative: it preserves single newlines
    but collapses excessive blank lines.
    """

    if not text:
        return ""

    # Unicode normalization:
    # - NFKC would normalize some compatibility characters; NFKD decomposes.
    #   The codebase historically used NFKD, so we keep it to avoid unexpected changes.
    t = unicodedata.normalize("NFKD", text)

    # Newline normalization
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # Remove NBSP and zero-width spaces
    t = t.replace("\u00A0", " ")
    t = _ZERO_WIDTH_RE.sub("", t)

    # Normalize tabs/other spacing
    t = t.replace("\t", " ")

    # Trim each line and collapse internal multi-space runs (but keep intentional newlines)
    lines = []
    for line in t.split("\n"):
        # collapse multiple spaces inside a line
        line = re.sub(r"[ ]{2,}", " ", line.strip())
        lines.append(line)

    t = "\n".join(lines)

    # Collapse excessive blank lines (3+ -> 2)
    t = re.sub(r"\n{3,}", "\n\n", t)

    # Strip leading/trailing whitespace and blank lines
    t = t.strip()

    # IMPORTANT: Re-compose to NFC for stable storage/display.
    #
    # The pipeline historically used NFKD above, which decomposes characters like:
    #   "ü" (U+00FC) -> "u" (U+0075) + COMBINING DIAERESIS (U+0308)
    #
    # Some UIs (including payload preview layers) can mis-render these combining marks
    # and show mojibake such as "fuÌr". Re-normalizing to NFC keeps the same textual
    # content but prefers composed codepoints where available.
    t = unicodedata.normalize("NFC", t)

    return t


def _split_paragraphs(text: str) -> list[str]:
    """Split by blank lines into paragraphs (after normalization)."""
    if not text:
        return []
    # One or more blank lines indicates paragraph boundary
    paras = re.split(r"\n\s*\n+", text)
    paras = [p.strip() for p in paras if p and p.strip()]
    return paras


def iter_sentence_like_units(text: str) -> Iterable[str]:
    """Yield sentence-like units.

    We avoid heavy NLP deps; this is a pragmatic splitter:
    - split on punctuation followed by whitespace/newline
    - also split on line breaks for list-like fragments

    If it fails, downstream packing will still avoid mid-sentence splits when possible.
    """

    if not text:
        return

    # First split by line breaks to preserve list items; then split each line into sentences.
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Split after . ! ? … : ; when followed by whitespace
        parts = re.split(r"(?<=[\.!\?…:;])\s+", line)
        for p in parts:
            p = p.strip()
            if p:
                yield p


def count_tokens(text: str) -> int:
    """Token count based on LlamaIndex tokenizer (best effort)."""
    if not text:
        return 0
    try:
        tok = get_tokenizer()
        return len(tok(text))
    except Exception:
        return max(1, len(text) // 4)


def _split_oversized_unit(unit: str, max_tokens: int) -> list[str]:
    """Fallback splitter when a single unit exceeds max_tokens.

    Tries to split at whitespace boundaries to avoid breaking words.
    """
    words = unit.split(" ")
    out: list[str] = []
    buf: list[str] = []
    for w in words:
        candidate = (" ".join(buf + [w])).strip()
        if not candidate:
            continue
        if count_tokens(candidate) <= max_tokens or not buf:
            buf.append(w)
        else:
            out.append(" ".join(buf).strip())
            buf = [w]
    if buf:
        out.append(" ".join(buf).strip())
    return [o for o in out if o]


def chunk_text_hierarchical(
    text: str,
    *,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    normalize: bool = True,
) -> list[str]:
    """Chunk text along semantic boundaries: paragraphs -> sentences -> token packing."""

    if not text:
        return []

    clean_text = normalize_text_for_rag(text) if normalize else text
    if not clean_text:
        return []

    paragraphs = [p for p in clean_text.split("\n\n") if p.strip()]

    units: list[str] = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        if count_tokens(p) <= chunk_size_tokens:
            units.append(p)
            continue

        for s in iter_sentence_like_units(p):
            if count_tokens(s) <= chunk_size_tokens:
                units.append(s)
            else:
                units.extend(_split_oversized_unit(s, chunk_size_tokens))

    chunks: list[str] = []
    current: list[str] = []

    def flush():
        nonlocal current
        if not current:
            return
        chunk_text = "\n".join(current).strip()
        if chunk_text:
            chunks.append(chunk_text)
        current = []

    for u in units:
        u = u.strip()
        if not u:
            continue

        if not current:
            current = [u]
            continue

        candidate = "\n".join(current + [u])
        if count_tokens(candidate) <= chunk_size_tokens:
            current.append(u)
            continue

        flush()

        # overlap from previous chunk by tail units
        if chunk_overlap_tokens > 0 and chunks:
            prev_units = chunks[-1].split("\n")
            overlap_units: list[str] = []
            for ou in reversed(prev_units):
                ou = ou.strip()
                if not ou:
                    continue
                test = "\n".join([ou] + overlap_units)
                if count_tokens(test) <= chunk_overlap_tokens:
                    overlap_units.insert(0, ou)
                else:
                    break
            if overlap_units:
                current = overlap_units

        current.append(u)

    flush()
    return chunks


def build_nodes_from_documents_hierarchical(
    docs: list,
    *,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    chunk_method: str = "paragraph_sentence_v1",
) -> list[TextNode]:
    """Create LlamaIndex TextNodes from docs, preserving metadata and adding chunk metadata."""

    nodes: list[TextNode] = []
    for doc in docs:
        raw_text = getattr(doc, "text", None) or ""
        md = getattr(doc, "metadata", None) or {}

        chunks = chunk_text_hierarchical(
            raw_text,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            normalize=True,
        )

        chunk_count = len(chunks)
        for idx, chunk_text in enumerate(chunks):
            node_md = dict(md)
            node_md.update(
                {
                    "chunk_index": idx,
                    "chunk_count": chunk_count,
                    "chunk_method": chunk_method,
                    "chunk_size_tokens": chunk_size_tokens,
                    "chunk_overlap_tokens": chunk_overlap_tokens,
                }
            )
            nodes.append(TextNode(text=chunk_text, metadata=node_md))

    return nodes


def _iter_transcript_nodes(doc, *, chunk_size_tokens: int) -> Iterable[TextNode]:
    """Cue-aware chunking for Transcript documents.

    Chunk boundaries fall on cue boundaries (no mid-sentence overlap needed —
    cues are already speech-aligned), and every chunk carries the
    ``start_seconds`` of its first cue so citations can deep-link into the
    video. The bulky ``segments`` list is dropped from the chunk metadata —
    only the per-chunk start time survives into the index.
    """
    md = getattr(doc, "metadata", None) or {}
    segments = md.get("segments") or []

    chunks: list[tuple[str, float]] = []  # (text, start_seconds)
    current_texts: list[str] = []
    current_start = 0.0
    for segment in segments:
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        if not current_texts:
            current_start = segment.get("start_seconds") or 0.0
        candidate = " ".join(current_texts + [text])
        if current_texts and count_tokens(candidate) > chunk_size_tokens:
            chunks.append((" ".join(current_texts), current_start))
            current_texts = [text]
            current_start = segment.get("start_seconds") or 0.0
        else:
            current_texts.append(text)
    if current_texts:
        chunks.append((" ".join(current_texts), current_start))

    base_md = {k: v for k, v in md.items() if k != "segments"}
    chunk_count = len(chunks)
    for idx, (chunk_text, start_seconds) in enumerate(chunks):
        node_md = dict(base_md)
        node_md.update(
            {
                "chunk_index": idx,
                "chunk_count": chunk_count,
                "chunk_method": "transcript_cues_v1",
                "chunk_size_tokens": chunk_size_tokens,
                "start_seconds": round(start_seconds, 1),
            }
        )
        yield TextNode(text=normalize_text_for_rag(chunk_text), metadata=node_md)


def iter_nodes_from_document_hierarchical(
    doc,
    *,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    chunk_method: str = "paragraph_sentence_v1",
) -> Iterable[TextNode]:
    """Yield TextNodes for a single document without accumulating them all.

    This is the bounded-memory alternative to `build_nodes_from_documents_hierarchical`.
    It keeps payload metadata consistent and is designed for huge module documents
    (PDFs, books).

    Transcript documents with timed segments take a cue-aware path instead of
    the generic paragraph/sentence chunking (see _iter_transcript_nodes).
    """

    raw_text = getattr(doc, "text", None) or ""
    md = getattr(doc, "metadata", None) or {}

    from src.vectordb.doc_types import TRANSCRIPT

    if md.get("type") == TRANSCRIPT and md.get("segments"):
        yield from _iter_transcript_nodes(doc, chunk_size_tokens=chunk_size_tokens)
        return

    chunks = chunk_text_hierarchical(
        raw_text,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        normalize=True,
    )

    chunk_count = len(chunks)
    for idx, chunk_text in enumerate(chunks):
        node_md = dict(md)
        node_md.update(
            {
                "chunk_index": idx,
                "chunk_count": chunk_count,
                "chunk_method": chunk_method,
                "chunk_size_tokens": chunk_size_tokens,
                "chunk_overlap_tokens": chunk_overlap_tokens,
            }
        )
        yield TextNode(text=chunk_text, metadata=node_md)
