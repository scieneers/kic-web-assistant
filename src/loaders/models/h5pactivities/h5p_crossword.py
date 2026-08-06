"""H5P.Crossword - Kreuzworträtsel-Handler."""
from dataclasses import dataclass, field
from typing import Optional
from src.loaders.models.hp5activities import strip_html, extract_library_from_h5p
from src.loaders.models.h5pactivities.h5p_base import H5PLeaf


@dataclass
class CrosswordEntry:
    """Ein einzelner Kreuzworträtsel-Eintrag."""
    clue: str  # Hinweis/Frage
    answer: str  # Antwort
    orientation: str = "across"  # "across" oder "down" (nicht relevant für Textausgabe)
    
    def to_text(self) -> str:
        """Answer-free rendering: word length shown as placeholder,
        matching how the crossword itself reveals it, without the solution."""
        clue_clean = strip_html(self.clue).strip()
        placeholder = "_" * len(self.answer.strip())
        return f"Frage: {clue_clean}. Antwort: {placeholder}"


@dataclass
class Crossword(H5PLeaf):
    """H5P.Crossword - Kreuzworträtsel mit Hinweisen und Antworten."""
    type: str = "H5P.Crossword"
    task_description: str = ""
    entries: list[CrosswordEntry] = field(default_factory=list)
    
    @classmethod
    def from_h5p_package(cls, module, content: dict, h5p_zip_path: str, **kwargs) -> Optional[str]:
        """
        Handler für standalone H5P.Crossword.
        Befüllt module.interactive_video mit Kreuzworträtsel-Einträgen.
        
        Args:
            module: Module-Objekt zum Befüllen
            content: Geladenes content.json dict
            h5p_zip_path: Pfad zum H5P ZIP-File
            **kwargs: Zusätzliche Services (nicht verwendet)
            
        Returns:
            Optional[str]: Fehlermeldung oder None
        """
        library = extract_library_from_h5p(h5p_zip_path) or "H5P.Crossword"
        
        crossword = cls.from_h5p_params(library, content)
        
        if crossword and crossword.entries:
            # Speichere als dict (Dependency Inversion)
            module.interactive_video = {
                "video_url": "",
                "vimeo_id": None,
                "interactions": [crossword.to_text()]
            }
            return None
        
        return "Konnte Kreuzworträtsel nicht extrahieren oder keine Einträge gefunden"
    
    @classmethod
    def from_h5p_params(cls, library: str, params: dict) -> Optional['Crossword']:
        """
        Extrahiert Crossword aus H5P params.
        
        Args:
            library: H5P library name (z.B. "H5P.Crossword 0.5")
            params: Parameter-Dict aus content.json
            
        Returns:
            Crossword-Objekt oder None
        """
        words = params.get("words", [])
        
        if not words:
            return None
        
        # Extrahiere task_description (optional)
        task_description = params.get("taskDescription", "").strip()
        
        # Erstelle CrosswordEntry-Objekte
        entries = []
        for word_data in words:
            clue = word_data.get("clue", "").strip()
            answer = word_data.get("answer", "").strip()
            orientation = word_data.get("orientation", "across")
            
            if clue and answer:
                entries.append(CrosswordEntry(
                    clue=clue,
                    answer=answer,
                    orientation=orientation
                ))
        
        if entries:
            return cls(
                type=library,
                task_description=task_description,
                entries=entries
            )
        
        return None
    
    def to_text(self) -> str:
        """
        Konvertiert Kreuzworträtsel zu Text.
        Format: [Kreuzworträtsel] <task_description>\n<clue>. Antwort: <answer>
        """
        parts = ["[Kreuzworträtsel]"]
        
        # Task-Description hinzufügen (falls vorhanden)
        if self.task_description:
            task_clean = strip_html(self.task_description).strip()
            if task_clean:
                parts.append(task_clean)
        
        # Alle Einträge hinzufügen
        for entry in self.entries:
            parts.append(entry.to_text())
        
        return "\n".join(parts)
