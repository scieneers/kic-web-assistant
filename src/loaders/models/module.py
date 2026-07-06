from enum import StrEnum
from typing import Any, Optional

from llama_index.core import Document
from pydantic import BaseModel, HttpUrl, computed_field

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
        
        # Transkript
        if len(self.transcripts) > 0:
            text_parts.append("\nTranscript:")
            text_parts.extend([str(transcript) for transcript in self.transcripts])
        
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
            
            text_parts.append(header)
            interactions = self.interactive_video.get("interactions", [])
            for interaction_text in interactions:
                text_parts.append(interaction_text)
        
        # Glossary Einträge
        if self.glossary and self.glossary.total_entries > 0:
            text_parts.append(f"\n--- Glossar ({self.glossary.total_entries} Einträge) ---")
            text_parts.append(str(self.glossary))
        
        # Resource Inhalte (PDF, DOCX, etc.)
        if self.resource and self.resource.extracted_text:
            text_parts.append(f"\n--- Dokument ({self.resource.filename}) ---")
            text_parts.append(self.resource.extracted_text)
        
        # Folder Inhalte (mehrere Dateien)
        if self.folder and self.folder.total_files > 0:
            text_parts.append(f"\n--- Ordner ({self.folder.total_files} Datei(en)) ---")
            text_parts.append(str(self.folder))
        
        # Book Inhalte (Kapitel mit HTML, Videos, Anhänge)
        if self.book and self.book.total_chapters > 0:
            text_parts.append(f"\n--- Book ({self.book.total_chapters} Kapitel) ---")
            text_parts.append(str(self.book))
        
        # URL Module (externe Links oder verarbeitbare Dateien)
        if self.url_module:
            text_parts.append("\n--- Externe Ressource ---")
            text_parts.append(str(self.url_module))
        
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

        return Document(text=name_only if is_empty else text, metadata=metadata)
