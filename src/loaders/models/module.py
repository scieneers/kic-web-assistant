import hashlib
import json
from enum import StrEnum
from typing import Any, Optional

from llama_index.core import Document
from pydantic import BaseModel, HttpUrl, computed_field

from src.vectordb.doc_types import BOOK_CHAPTER, FLASHCARD, GLOSSARY_ENTRY, QUIZ_ITEM, TRANSCRIPT

from src.loaders.models.book import Book
from src.loaders.models.downloadablecontent import DownloadableContent
from src.loaders.models.folder import Folder
from src.loaders.models.glossary import Glossary
from src.loaders.models.resource import Resource
from src.loaders.models.texttrack import TextTrack
from src.loaders.models.url import UrlModule
from src.loaders.models.videotime import Video


class ModuleTypes(StrEnum):
    VIDEOTIME = "videotime"
    PAGE = "page"
    H5P = "h5pactivity"
    GLOSSARY = "glossary"
    RESOURCE = "resource"
    FOLDER = "folder"
    BOOK = "book"
    URL = "url"


# Friendly German labels for modnames we deliberately do NOT extract, so the
# chat fallback can name what's missing instead of implying the module is
# empty. "quiz" (native Moodle Quiz) was investigated concretely: there is no
# read-only web-service path to question+answer content — the only way is a
# real (though "preview") quiz attempt with a per-question-type form replay,
# which writes state into Moodle on every ingest run (see
# documentation/PLAN_INHALTSTYPEN.md). Any OTHER modname with no ModuleTypes
# entry falls back to a generic label naming the raw modname (see
# to_document()) so future gaps are equally honest without needing an entry
# here first.
UNSUPPORTED_MODNAME_LABELS: dict[str, str] = {
    "quiz": "natives Moodle-Quiz",
}


class Module(BaseModel):
    """Lowest level content block of a course. Can be a file, video, hp5, etc."""

    id: int
    visible: int
    name: str
    url: HttpUrl | None = None
    modname: str  # content type
    instance: int | None = None  # ID of the specific resource (glossary_id, videotime_id, etc.)
    h5p_content_type: str | None = None  # H5P library name from content.json
    text: str | None = None
    intro: str | None = None  # HTML intro text from API (available for resources, activities, etc.)
    contents: list[DownloadableContent] | None = None
    # Raw metadata from core_course_get_contents. We keep it as a dict so we can compute
    # stable fingerprints without changing extraction logic.
    contentsinfo: dict[str, Any] | None = None
    videotime: Video | None = None
    transcripts: list[TextTrack] = []
    interactive_video: dict | None = None  # H5P Interactive Video data (als dict, nicht typisiert)
    glossary: Glossary | None = None  # Glossary entries
    resource: Resource | None = None  # Resource file (PDF, DOCX, etc.)
    folder: Folder | None = None  # Folder with multiple files (PDF, Audio, etc.)
    book: Book | None = None  # Book with multiple chapters (HTML, videos, attachments)
    url_module: UrlModule | None = None  # URL module (external link or downloadable file)
    # Structured item payloads (collected via h5p_payloads.collect_item_payloads,
    # independent of the flattened text pipeline). Feed to_item_documents().
    quiz_items: list[dict] = []  # {kind, question, correct_answers, incorrect_answers, ...}
    flashcards: list[dict] = []  # {front, back}

    @computed_field  # type: ignore[misc]
    @property
    def type(self) -> Optional[ModuleTypes]:
        match self.modname:
            case "videotime":
                return ModuleTypes.VIDEOTIME
            case "page":
                return ModuleTypes.PAGE
            case "h5pactivity":
                return ModuleTypes.H5P
            case "glossary":
                return ModuleTypes.GLOSSARY
            case "resource":
                return ModuleTypes.RESOURCE
            case "folder":
                return ModuleTypes.FOLDER
            case "book":
                return ModuleTypes.BOOK
            case "url":
                return ModuleTypes.URL
            case _:
                return None

    def to_document(self, course_id) -> Document:
        text_parts = []
        
        # Name
        text_parts.append(f"Module Name: {self.name}")
        
        # Intro
        if self.intro:
            from src.loaders.models.hp5activities import strip_html
            intro_clean = strip_html(self.intro)
            if intro_clean:
                text_parts.append(f"\nBeschreibung: {intro_clean}")
        
        # Text-Content
        if self.text is not None:
            text_parts.append(f"\nText: {self.text}")
        
        # Video-Transkripte werden als eigene Transcript-Dokumente indexiert
        # (to_item_documents, cue-bewusst gechunkt mit start_seconds für
        # Video-Deep-Links) — hier nur die Inventarzeile (s. u.).
        
        # H5P Inhalte (Interactive Video, QuestionSet, etc.)
        if self.interactive_video:
            # Dynamischer Header basierend auf H5P-Typ
            if self.h5p_content_type:
                if "InteractiveVideo" in self.h5p_content_type:
                    header = "\n--- Interaktive Inhalte im Video ---"
                elif "Accordion" in self.h5p_content_type:
                    header = "\n--- Accordion-Inhalte ---"
                elif "Column" in self.h5p_content_type:
                    header = "\n--- Spalten-Inhalte ---"
                elif "QuestionSet" in self.h5p_content_type:
                    header = "\n--- Fragen im QuestionSet ---"
                elif "CoursePresentation" in self.h5p_content_type:
                    header = "\n--- Course Presentation ---"
                elif "MultiChoice" in self.h5p_content_type or "SingleChoiceSet" in self.h5p_content_type:
                    header = "\n--- Quiz-Frage ---"
                elif "TrueFalse" in self.h5p_content_type:
                    header = "\n--- Wahr/Falsch-Frage ---"
                elif "Blanks" in self.h5p_content_type:
                    header = "\n--- Lückentext ---"
                elif "DragText" in self.h5p_content_type:
                    header = "\n--- Drag-Text-Aufgabe ---"
                elif "DragQuestion" in self.h5p_content_type:
                    header = "\n--- Drag-&-Drop-Aufgabe ---"
                elif "Text" in self.h5p_content_type:
                    header = "\n--- H5P Text-Inhalt ---"
                elif "Video" in self.h5p_content_type:
                    header = "\n--- Video-Inhalt ---"
                elif "Dialogcards" in self.h5p_content_type:
                    header = "\n--- Dialog-Karten ---"
                elif "Flashcards" in self.h5p_content_type:
                    header = "\n--- Karteikarten ---"
                elif "ImageHotspot" in self.h5p_content_type:
                    header = "\n--- Bild-Hotspot-Aufgabe ---"
                elif "Timeline" in self.h5p_content_type:
                    header = "\n--- Timeline ---"
                elif "Gamemap" in self.h5p_content_type or "GameMap" in self.h5p_content_type:
                    header = "\n--- Interaktive Karte (Gamemap) ---"
                elif "Crossword" in self.h5p_content_type:
                    header = "\n--- Kreuzworträtsel ---"
                else:
                    header = f"\n--- H5P Inhalt ({self.h5p_content_type}) ---"
            else:
                header = "\n--- H5P Inhalte ---"
            
            # Reine Karten-Module: die Karten werden als eigene Flashcard-
            # Dokumente indexiert (to_item_documents) — im Modultext bleibt
            # nur die Inventarzeile, sonst wären sie doppelt im Index.
            is_pure_card_module = self.h5p_content_type and (
                "Dialogcards" in self.h5p_content_type or "Flashcards" in self.h5p_content_type
            )
            if not is_pure_card_module:
                text_parts.append(header)
                interactions = self.interactive_video.get("interactions", [])
                for interaction_text in interactions:
                    text_parts.append(interaction_text)

        # Glossar-Einträge werden als eigene GlossaryEntry-Dokumente indexiert
        # (to_item_documents) — hier nur die Inventarzeile (s. u.), sonst wäre
        # jeder Eintrag doppelt im Index.
        
        # Resource Inhalte (PDF, DOCX, etc.)
        if self.resource and self.resource.extracted_text:
            text_parts.append(f"\n--- Dokument ({self.resource.filename}) ---")
            text_parts.append(self.resource.extracted_text)
        
        # Folder Inhalte (mehrere Dateien)
        if self.folder and self.folder.total_files > 0:
            text_parts.append(f"\n--- Ordner ({self.folder.total_files} Datei(en)) ---")
            text_parts.append(str(self.folder))
        
        # Buch-Kapitel werden als eigene BookChapter-Dokumente indexiert
        # (to_item_documents, mit Kapitel-Deep-Link) — im Modultext bleiben
        # nur die Einführung und die Inventarzeile (s. u.).
        if self.book and self.book.intro:
            text_parts.append(f"\n=== Einführung ===\n{self.book.intro}")
        
        # URL Module (externe Links oder verarbeitbare Dateien)
        if self.url_module:
            text_parts.append("\n--- Externe Ressource ---")
            text_parts.append(str(self.url_module))

        # Inventarzeile: benennt die als eigene Dokumente ausgelagerten Inhalte
        # (Quiz, Glossar, Karteikarten). Hält Kurs-Discovery am Leben ("hat
        # Kurs X ein Glossar?") und verhindert, dass ein Modul, dessen Inhalt
        # komplett in Item-Dokumente gewandert ist, als EmptyModule endet.
        inventory = self._inventory_parts()
        if inventory:
            text_parts.append("\nEnthält: " + "; ".join(inventory))

        text = "\n".join(text_parts)

        # Leere Module: minimaler Marker-Text damit check_if_module_exists() in der API
        # funktioniert. type="EmptyModule" signalisiert dem Retriever, diesen Chunk in
        # generellen Suchen auszufiltern; im Modul-Kontext (module_id-Filter gesetzt)
        # ist er auffindbar und liefert den fullname für den Fallback-Text.
        name_only = f"Module Name: {self.name}"
        is_empty = text.strip() in ("", name_only)
        doc_type = "EmptyModule" if is_empty else "module"

        metadata = {
            "course_id": course_id,
            "module_id": self.id,
            "fullname": self.name,
            "type": doc_type,
            "source": "Moodle",
            "url": str(self.url),
            "modname": self.modname,
            **({"h5p_content_type": self.h5p_content_type} if self.h5p_content_type else {}),
        }

        # self.type is None exactly when modname has no ModuleTypes entry —
        # i.e. no extraction handler ever ran (see get_module_contents() in
        # moodle.py), which is why is_empty is always true in that case. Tag
        # it distinctly from a genuinely-empty-but-supported module so the
        # chat fallback can name the real reason instead of "kein Inhalt".
        if is_empty and self.type is None:
            metadata["unsupported_modname"] = self.modname
            metadata["unsupported_label"] = UNSUPPORTED_MODNAME_LABELS.get(
                self.modname, f"Inhaltstyp „{self.modname}“"
            )

        return Document(text=name_only if is_empty else text, metadata=metadata)

    def _inventory_parts(self) -> list[str]:
        """Short German phrases naming the content that lives in item documents."""
        parts: list[str] = []
        if self.quiz_items:
            parts.append(f"{len(self.quiz_items)} Quizfrage(n)")
        if self.glossary and self.glossary.total_entries > 0:
            concepts = [e.concept for e in self.glossary.entries[:30] if e.concept]
            listing = f": {', '.join(concepts)}" if concepts else ""
            parts.append(f"Glossar mit {self.glossary.total_entries} Einträgen{listing}")
        if self.flashcards:
            parts.append(f"{len(self.flashcards)} Karteikarte(n)")
        transcript_count = sum(1 for t in self.transcripts if t is not None and (t.transcript or t.segments))
        if transcript_count:
            parts.append(f"{transcript_count} Video-Transkript(e)")
        if self.book and self.book.total_chapters > 0:
            titles = [c.title for c in self.book.chapters[:20] if c.title]
            listing = f": {', '.join(titles)}" if titles else ""
            parts.append(f"Book mit {self.book.total_chapters} Kapiteln{listing}")
        return parts

    def _item_metadata(
        self, course_id, doc_type: str, key: str, title: str, order: int, payload: dict, url: str | None = None
    ) -> dict:
        """Shared metadata for one structured item document.

        fullname stays the MODULE name — the frontend course/module tree labels
        modules by fullname; title carries the item-specific label instead.
        payload_fingerprint participates in the loader's content hash so a
        changed solution re-ingests even when the visible text is unchanged.
        """
        return {
            "course_id": course_id,
            "module_id": self.id,
            "fullname": self.name,
            "title": title,
            "type": doc_type,
            "source": "Moodle",
            "url": url or str(self.url),
            "modname": self.modname,
            **({"h5p_content_type": self.h5p_content_type} if self.h5p_content_type else {}),
            "source_doc_key": key,
            "item_order": order,
            "payload": payload,
            "payload_fingerprint": json.dumps(payload, sort_keys=True, ensure_ascii=False),
        }

    @staticmethod
    def _dedup_key(key: str, used_keys: set[str]) -> str:
        """Deterministic ordinal suffix for the degenerate exact-duplicate case."""
        candidate, ordinal = key, 2
        while candidate in used_keys:
            candidate = f"{key}-{ordinal}"
            ordinal += 1
        used_keys.add(candidate)
        return candidate

    @staticmethod
    def _quiz_item_text(payload: dict) -> str:
        """Answer-free embedding text for a quiz item (defense in depth: even
        though QuizItem docs are excluded from normal retrieval, their text
        never carries the solution — only the metadata payload does)."""
        lines = [payload.get("question") or ""]
        options = sorted((payload.get("correct_answers") or []) + (payload.get("incorrect_answers") or []))
        if options:
            lines.append("Antwortoptionen: " + ", ".join(options))
        for group in payload.get("statement_groups") or []:
            lines.append("Aussagen: " + ", ".join(sorted(group)))
        if payload.get("text_with_blanks"):
            import re

            lines.append(re.sub(r"\*[^*]+\*", "___", payload["text_with_blanks"]))
        return "\n".join(line for line in lines if line).strip()

    def to_item_documents(self, course_id) -> list[Document]:
        """One standalone index Document per structured item of this module.

        Emits QuizItem (question + solutions in payload), GlossaryEntry
        (concept/definition, deep link) and Flashcard (front/back) documents.
        Each carries its own stable source_doc_key — the loader's
        _source_doc_key() honors it, so items get independent change detection
        and stale deletion alongside the module document.
        """
        documents: list[Document] = []
        used_keys: set[str] = set()
        base_key = f"moodle:{course_id}:{self.id}"

        for order, payload in enumerate(self.quiz_items):
            stable_id = payload.get("sub_content_id") or hashlib.sha256(
                (
                    (payload.get("question") or "")
                    + "|"
                    + "|".join(sorted((payload.get("correct_answers") or []) + (payload.get("incorrect_answers") or [])))
                ).encode()
            ).hexdigest()[:12]
            key = self._dedup_key(f"{base_key}:quiz:{stable_id}", used_keys)
            title = (payload.get("question") or "Quizfrage")[:80]
            text = self._quiz_item_text(payload)
            if not text:
                continue
            documents.append(
                Document(text=text, metadata=self._item_metadata(course_id, QUIZ_ITEM, key, title, order, payload))
            )

        if self.glossary:
            for order, entry in enumerate(self.glossary.entries):
                if not entry.concept:
                    continue
                key = self._dedup_key(f"{base_key}:glossary:{entry.id}", used_keys)
                separator = "&" if "?" in str(self.url) else "?"
                # Moodle in-course entry deep link (see PLAN_INHALTSTYPEN.md —
                # verify once against the live Moodle before the first ingest).
                entry_url = f"{self.url}{separator}mode=entry&hook={entry.id}"
                payload = {"concept": entry.concept, "definition": entry.definition}
                documents.append(
                    Document(
                        text=f"{entry.concept}: {entry.definition}",
                        metadata=self._item_metadata(
                            course_id, GLOSSARY_ENTRY, key, entry.concept[:80], order, payload, url=entry_url
                        ),
                    )
                )

        for order, card in enumerate(self.flashcards):
            front, back = card.get("front") or "", card.get("back") or ""
            if not front or not back:
                continue
            stable_id = hashlib.sha256(f"{front}\x1f{back}".encode()).hexdigest()[:12]
            key = self._dedup_key(f"{base_key}:card:{stable_id}", used_keys)
            documents.append(
                Document(
                    text=f"{front}: {back}",
                    metadata=self._item_metadata(course_id, FLASHCARD, key, front[:80], order, card),
                )
            )

        for order, track in enumerate(self.transcripts):
            if track is None or not (track.transcript or track.segments):
                continue
            # Stable, non-positional key (removing one of several videos must
            # not re-key the others): video URL, else the track id, else a
            # content fingerprint.
            if getattr(track, "video_url", None):
                stable_id = hashlib.sha256(track.video_url.encode()).hexdigest()[:12]
            elif getattr(track, "id", 0):
                stable_id = str(track.id)
            else:
                stable_id = hashlib.sha256((track.transcript or "")[:500].encode()).hexdigest()[:12]
            key = self._dedup_key(f"{base_key}:transcript:{stable_id}", used_keys)

            text = track.transcript or " ".join(s.get("text", "") for s in track.segments or [])
            if not text.strip():
                continue
            metadata = self._item_metadata(
                course_id,
                TRANSCRIPT,
                key,
                f"Video: {self.name}"[:80],
                order,
                # No solution payload — the transcript IS the text, and there
                # is nothing else worth storing: the citable link is always
                # the KI-Campus module page (citations never leave the
                # platform), and start_seconds (below) already carries
                # everything the citation renderer needs for the timestamp.
                {},
                url=str(self.url),
            )
            # Timed segments ride along for the cue-aware chunker (helper.py
            # strips them from the chunk metadata again — only the per-chunk
            # start_seconds reaches the index).
            if track.segments:
                metadata["segments"] = track.segments
            documents.append(Document(text=text, metadata=metadata))

        if self.book:
            for order, chapter in enumerate(self.book.chapters):
                chapter_text = str(chapter).strip()
                if not chapter_text:
                    continue
                key = self._dedup_key(f"{base_key}:chapter:{chapter.chapter_id}", used_keys)
                separator = "&" if "?" in str(self.url) else "?"
                chapter_url = f"{self.url}{separator}chapterid={chapter.chapter_id}"
                documents.append(
                    Document(
                        text=chapter_text,
                        metadata=self._item_metadata(
                            course_id,
                            BOOK_CHAPTER,
                            key,
                            chapter.title[:80] if chapter.title else f"Kapitel {chapter.chapter_id}",
                            order,
                            {"chapter_id": chapter.chapter_id, "title": chapter.title},
                            url=chapter_url,
                        ),
                    )
                )

        return documents
