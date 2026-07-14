import re
from urllib.parse import urlparse

from langfuse.decorators import observe
from llama_index.core.schema import TextNode

CITATION_TEXT = '[<a href="{url}">{title}</a>]{suffix}'


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
    if isinstance(start, (int, float)) and start > 0:
        return f" (ab Minute {_format_video_timestamp(start)})"
    return ""


def _get_display_title(doc: TextNode) -> str:
    """Get display title from document metadata, with fallback to shortened URL."""
    title = doc.metadata.get("title") or doc.metadata.get("fullname")
    if title:
        # Entferne führende Kapitelnummern (z.B. "3.1. Titel" → "Titel")
        title = re.sub(r'^\d+(\.\d+)*\.?\s*', '', title)
        
        # Kürze sehr lange Titel (smart truncate am Wortende)
        if len(title) > 50:
            truncated = title[:47]
            # Finde letztes Leerzeichen
            last_space = truncated.rfind(' ')
            if last_space > 30:  # Nur wenn noch genug Text übrig bleibt
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

    def _get_source_docs_from_answer(self, answer: str) -> list[int]:
        """Extract all [docN] from answer and extract N, and just return the N's as a list of ints"""
        results = re.findall(r"\[doc(\d+)\]", answer)
        doc_ids = [int(i) for i in results]
        # remove duplicates while preserving order
        doc_ids = list(dict.fromkeys(doc_ids))
        return doc_ids

    def _remove_doc_ids_from_answer(self, answer: str, doc_ids: list[int]) -> str:
        """Remove all [docN] from answer where N is in doc_ids list. Used for hallucinated references."""
        for idx in doc_ids:
            answer = answer.replace(f"[doc{idx}]", "")
        return answer

    @observe()
    def parse(
        self,
        answer: str,
        source_documents: list[TextNode],
    ) -> str:
        answer = self._clean_up_answer(answer)
        doc_ids = self._get_source_docs_from_answer(answer)

        # we track doc ids which exist in case of hallucination
        fake_doc_ids: list[int] = []
        seen_urls = set()

        for i in doc_ids:
            idx = i - 1
            try:
                doc = source_documents[idx]
                url = doc.metadata.get("url") or ""
                if url not in seen_urls:
                    title = _get_display_title(doc)
                    replacement_text = CITATION_TEXT.format(url=url, title=title, suffix=citation_suffix(doc))
                    seen_urls.add(url)
                    answer = re.sub(rf"[, ]*\[doc{i}\]", lambda m: replacement_text, answer)
                else:
                    answer = re.sub(rf"[, ]*\[doc{i}\]", "", answer)
            except IndexError:
                print(f"Could not find doc{i} in source documents")
                fake_doc_ids.append(i)

        answer = self._remove_doc_ids_from_answer(answer, fake_doc_ids)

        return answer