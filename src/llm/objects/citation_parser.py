import re
from urllib.parse import urlparse

from langfuse.decorators import observe
from llama_index.core.schema import TextNode

from src.llm.state.socratic_v2_language import canonical_language

# Inline: a short clickable number — long page titles in the running text made
# answers hard to read in the narrow chat panel (testing feedback p. 17).
INLINE_CITATION = '[<a href="{url}">{number}</a>]'
# Footer: the full, descriptive source list below the answer.
SOURCE_ENTRY = '{number}. <a href="{url}">{title}</a>{suffix}'
# Heading doubles as the call-to-action towards further course content.
SOURCES_HEADING = {
    "German": "Zum Weiterlesen:",
    "English": "Further reading:",
}
# The list has room for full titles; the old 50-char cap was meant for running text.
LIST_TITLE_MAX_LENGTH = 100
_DOC_MARKER_RE = re.compile(r"\[doc(\d+)\]")
# A run of adjacent markers like "[doc1][doc2]" or ", [doc1] [doc3]".
_MARKER_GROUP_RE = re.compile(r"(?:[, ]*\[doc\d+\])+")


def _format_video_timestamp(seconds: float) -> str:
    """Seconds → 'M:SS' (or 'H:MM:SS' for videos over an hour)."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def citation_suffix(doc: TextNode) -> str:
    """Deterministic ' (ab Minute M:SS)' suffix for video-transcript sources.

    The citation link always points at the KI-Campus module page, never the
    external video — so the exact video position can't be conveyed via the
    link itself. Attaching it here (at render time) guarantees it always
    appears, rather than relying on an LLM instruction to mention it in
    prose, which smaller models don't follow reliably.
    """
    start = doc.metadata.get("start_seconds")
    if isinstance(start, (int, float)) and start >= 0:
        return f" (ab Minute {_format_video_timestamp(start)})"
    return ""


def _get_display_title(doc: TextNode, max_length: int = 50) -> str:
    """Get display title from document metadata, with fallback to shortened URL."""
    title = doc.metadata.get("title") or doc.metadata.get("fullname")
    if title:
        # Entferne führende Kapitelnummern (z.B. "3.1. Titel" → "Titel")
        title = re.sub(r'^\d+(\.\d+)*\.?\s*', '', title)
        
        # Kürze sehr lange Titel (smart truncate am Wortende)
        if len(title) > max_length:
            truncated = title[: max_length - 3]
            # Finde letztes Leerzeichen
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.6:  # Nur wenn noch genug Text übrig bleibt
                truncated = truncated[:last_space]
            return truncated + "..."
        return title
    
    # Fallback: URL kürzen
    url = doc.metadata.get("url", "")
    if url:
        parsed = urlparse(url)
        # Zeige Host + gekürzte Pfad
        path = parsed.path
        if len(path) > 30:
            path = path[:27] + "..."
        return f"{parsed.netloc}{path}"
    
    return "Quelle"


class SourceList:
    """Numbers cited sources in order of first citation and renders the list.

    Shared by the streaming resolver and the non-streaming parser so both
    produce identical output. Sources are deduplicated by URL: citing the same
    page twice reuses its number.
    """

    def __init__(self, sources: list[TextNode]) -> None:
        self._sources = sources
        self._number_by_url: dict[str, int] = {}
        self._entries: list[str] = []

    def resolve(self, marker: str) -> str:
        """'[docN]' → inline numbered link; '' for unknown/hallucinated N."""
        m = _DOC_MARKER_RE.fullmatch(marker)
        if not m:
            return ""
        idx = int(m.group(1)) - 1
        if idx < 0 or idx >= len(self._sources):
            return ""
        doc = self._sources[idx]
        url = doc.metadata.get("url") or ""
        if url not in self._number_by_url:
            number = len(self._entries) + 1
            self._number_by_url[url] = number
            self._entries.append(
                SOURCE_ENTRY.format(number=number, url=url, title=_get_display_title(doc, LIST_TITLE_MAX_LENGTH), suffix=citation_suffix(doc))
            )
        return INLINE_CITATION.format(url=url, number=self._number_by_url[url])

    def footer(self, language: str | None) -> str:
        """Source list to append below the answer ('' if nothing was cited)."""
        if not self._entries:
            return ""
        heading = SOURCES_HEADING.get(canonical_language(language), SOURCES_HEADING["English"])
        return f"\n\n**{heading}**\n" + "\n".join(self._entries)


class CitationParser:
    """This class is responsible for parsing the document references from the return string."""

    def __init__(self) -> None:
        self.name = "OutputParser"

    def _clean_up_answer(self, answer: str) -> str:
        """Remove double spaces and remove doc references in false format.
        Keep only references with allowed format [docN] where N=integer. E.g. [doc5] instead of [doc10.2] or doc[2.3.1]
        """
        answer = answer.replace("  ", " ")
        # Anything where N is a float
        answer = re.sub(r"\[doc\d+(?:\.\d+)+\]", "", answer)
        # Anything where N is not a digit
        answer = re.sub(r"\[doc\D\]", "", answer)
        return answer

    @observe()
    def parse(
        self,
        answer: str,
        source_documents: list[TextNode],
        language: str | None = None,
    ) -> str:
        answer = self._clean_up_answer(answer)
        sources = SourceList(source_documents)

        def _resolve_group(match: re.Match) -> str:
            # "[doc1][doc2]" pointing at the same page must not render "[1][1]".
            rendered: list[str] = []
            for number in _DOC_MARKER_RE.findall(match.group()):
                link = sources.resolve(f"[doc{number}]")
                if link and link not in rendered:
                    rendered.append(link)
            return "".join(rendered)

        answer = _MARKER_GROUP_RE.sub(_resolve_group, answer)
        return answer + sources.footer(language)
