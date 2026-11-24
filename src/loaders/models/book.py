"""
Datenmodell für Moodle Book Module.

Ein Book-Modul repräsentiert ein mehrseitiges Dokument mit Kapiteln.
Jedes Kapitel kann HTML-Inhalt, Videos und Dateianhänge (PDFs, ZIPs, etc.) enthalten.
"""

import re

from pydantic import BaseModel

from src.loaders.models.resource import Resource
from src.loaders.models.texttrack import TextTrack


def chapter_order_from_structure(structure) -> dict[str, int]:
    """Kapitel-ID → Position aus dem Moodle-structure-JSON (autoritative
    Reihenfolge inkl. Unterkapitel-Hierarchie).

    Die chapter_id ist Moodles interne DB-ID — weder ihre String- noch ihre
    numerische Sortierung entspricht zwingend der pädagogischen Reihenfolge;
    nur die structure-Datei kennt sie. Fehlt sie, bleibt als Fallback die
    numerische Sortierung (siehe extract_book).
    """
    order: dict[str, int] = {}
    counter = 0

    def _walk(nodes) -> None:
        nonlocal counter
        if not isinstance(nodes, list):
            return
        for node in nodes:
            if not isinstance(node, dict):
                continue
            match = re.search(r"(\d+)/index\.html", node.get("href") or "")
            if match:
                order[match.group(1)] = counter
                counter += 1
            _walk(node.get("subitems") or [])

    if isinstance(structure, dict):
        structure = [structure]
    _walk(structure)
    return order


class BookChapter(BaseModel):
    """
    Repräsentiert ein einzelnes Kapitel in einem Book-Modul.
    
    Jedes Kapitel besteht aus:
    - Titel
    - HTML-Inhalt (Text)
    - Optional: Videos (Transkripte)
    - Optional: Dateianhänge (PDFs, ZIPs, etc.)
    
    Attributes:
        chapter_id: ID des Kapitels (aus filepath, z.B. "287" aus "/287/")
        title: Titel des Kapitels
        html_text: Extrahierter Text aus HTML
        transcripts: Video-Transkripte (falls Videos im HTML eingebettet sind)
        attachments: Dateianhänge (PDFs, ZIPs, etc.) als Resource-Objekte
    """
    
    chapter_id: str
    title: str
    html_text: str | None = None
    transcripts: list[TextTrack] = []
    attachments: list[Resource] = []
    
    def __str__(self) -> str:
        """
        Formatiert das Kapitel für Document-Generierung.
        
        Format:
        === Kapitel: [Titel] ===
        [HTML-Text]
        
        [Video-Transkripte]
        
        [Dateianhänge]
        """
        parts = [f"=== Kapitel: {self.title} ==="]
        
        # HTML-Text
        if self.html_text:
            parts.append(self.html_text)
        
        # Video-Transkripte
        if self.transcripts:
            parts.append("\n--- Videos im Kapitel ---")
            for transcript in self.transcripts:
                if transcript:
                    parts.append(str(transcript))
        
        # Dateianhänge
        if self.attachments:
            parts.append("\n--- Dateianhänge ---")
            for attachment in self.attachments:
                if attachment.extracted_text:
                    parts.append(f"\nDatei: {attachment.filename}")
                    parts.append(attachment.extracted_text)
        
        return "\n".join(parts)


class Book(BaseModel):
    """
    Repräsentiert ein Book-Modul mit mehreren Kapiteln.
    
    Books sind strukturierte Dokumente mit hierarchischen Kapiteln.
    Jedes Kapitel kann eigene Inhalte, Videos und Anhänge haben.
    
    Attributes:
        book_id: ID des Books (aus module.instance)
        module_id: ID des Course-Moduls
        chapters: Liste der Kapitel (BookChapter-Objekte)
        structure: JSON-Struktur aus dem structure-Content (optional)
    """
    
    book_id: int
    module_id: int
    intro: str | None = None  # Intro-Text aus mod_book_get_books_by_courses
    chapters: list[BookChapter] = []
    # Rohes structure-JSON aus dem Moodle-Content — je nach Moodle-Version ein
    # Objekt oder eine Liste von Kapitel-Knoten.
    structure: dict | list | None = None
    
    def __str__(self) -> str:
        """
        Formatiert alle Kapitel für Document-Generierung.
        
        Jedes Kapitel wird mit Header versehen und sequentiell aufgelistet.
        """
        if not self.chapters:
            return f"Book (leer, {self.book_id})"
        
        parts = []
        
        # Intro-Text (falls vorhanden)
        if self.intro:
            parts.append(f"=== Einführung ===\n{self.intro}")
        
        # Kapitel
        chapter_texts = [str(chapter) for chapter in self.chapters]
        parts.extend(chapter_texts)
        
        return "\n\n".join(parts)
    
    @property
    def total_chapters(self) -> int:
        """Gibt die Anzahl der Kapitel zurück."""
        return len(self.chapters)
    
    @property
    def total_attachments(self) -> int:
        """Gibt die Gesamtanzahl aller Anhänge zurück."""
        return sum(len(chapter.attachments) for chapter in self.chapters)
    
    @property
    def total_videos(self) -> int:
        """Gibt die Gesamtanzahl aller Video-Transkripte zurück."""
        return sum(len(chapter.transcripts) for chapter in self.chapters)
