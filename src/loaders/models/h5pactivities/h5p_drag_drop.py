from dataclasses import dataclass, field
from typing import Optional
from src.loaders.models.hp5activities import strip_html, extract_library_from_h5p
from src.loaders.models.h5pactivities.h5p_base import H5PLeaf


@dataclass
class DragDropText(H5PLeaf):
    """Drag Text - Wörter in Lücken ziehen (H5P.DragText)"""
    type: str  # "H5P.DragText"
    task_description: str
    text_field: str  # Text mit *Wort*-Markierungen für Lücken
    hint: str = "Wörter in Asterisken (*...*) müssen in die richtige Lücke gezogen werden."
    
    @classmethod
    def from_h5p_package(cls, module, content: dict, h5p_zip_path: str, **kwargs) -> Optional[str]:
        """
        Handler für standalone H5P.DragText.
        Befüllt module.interactive_video mit einer Drag-Text-Aufgabe.
        """
        library = extract_library_from_h5p(h5p_zip_path) or "H5P.DragText"
        params = content
        
        drag_text = cls.from_h5p_params(library, params)
        
        if drag_text:
            # Speichere als dict (Dependency Inversion)
            module.interactive_video = {
                "video_url": "",
                "vimeo_id": None,
                "interactions": [drag_text.to_text()]
            }
            return None
        
        return "Konnte Drag-Text-Aufgabe nicht extrahieren"
    
    @classmethod
    def from_h5p_params(cls, library: str, params: dict) -> Optional['DragDropText']:
        """Extrahiert DragDropText aus H5P params."""
        task_description = params.get("taskDescription", "").strip()
        text_field = params.get("textField", "").strip()
        
        if text_field:
            # Fallback für task_description, falls leer
            if not task_description:
                task_description = "Ziehen Sie die Wörter in die richtigen Lücken."
            
            return cls(
                type=library,
                task_description=task_description,
                text_field=text_field
            )
        
        return None
    
    def to_text(self) -> str:
        task_clean = strip_html(self.task_description)
        text_clean = strip_html(self.text_field)
        return f"[Drag Text] {task_clean}\n{self.hint}\n{text_clean}"


@dataclass
class DragDropQuestion(H5PLeaf):
    """Drag & Drop-Frage"""
    type: str  # "H5P.DragQuestion"
    question: str
    categories: list[str]  # Dropzones/Kategorien
    draggable_items: list[str]  # Elemente zum Ziehen
    correct_mappings: dict[str, list[str]] = field(default_factory=dict)  # Kategorie -> Liste von Elementen
    
    @classmethod
    def from_h5p_package(cls, module, content: dict, h5p_zip_path: str, **kwargs) -> Optional[str]:
        """
        Handler für standalone H5P.DragQuestion.
        Befüllt module.interactive_video mit einer Drag&Drop-Aufgabe.
        """
        library = extract_library_from_h5p(h5p_zip_path) or "H5P.DragQuestion"
        params = content
        
        drag_drop = cls.from_h5p_params(library, params)
        
        if drag_drop:
            # Speichere als dict (Dependency Inversion)
            module.interactive_video = {
                "video_url": "",
                "vimeo_id": None,
                "interactions": [drag_drop.to_text()]
            }
            return None
        
        return "Konnte Drag&Drop-Aufgabe nicht extrahieren"
    
    @classmethod
    def from_h5p_params(cls, library: str, params: dict) -> Optional['DragDropQuestion']:
        """Extrahiert DragDropQuestion aus H5P params."""
        task = params.get("question", {}).get("task", {})
        dropzones = task.get("dropZones", [])
        elements = task.get("elements", [])
        
        question_text = "Ordne die Elemente den Kategorien zu:"
        
        # Extrahiere Kategorien (Dropzones) mit Index
        categories = []
        category_map = {}  # Index -> Label
        for idx, dz in enumerate(dropzones):
            label_html = dz.get("label", "").strip()
            label_clean = strip_html(label_html).strip()
            if label_clean:
                categories.append(label_html)  # Original für to_text() wo nochmal gestrippt wird
                category_map[str(idx)] = label_clean
        
        # Extrahiere ziehbare Elemente mit Index
        draggable_items = []
        element_map = {}  # Index -> Text (bereits cleaned)
        for idx, elem in enumerate(elements):
            text_html = elem.get("type", {}).get("params", {}).get("text", "").strip()
            text_clean = strip_html(text_html).strip()
            if text_clean:
                draggable_items.append(text_html)  # Original für to_text()
                element_map[str(idx)] = text_clean
        
        # Erstelle korrekte Zuordnungen
        correct_mappings = {}
        for idx, dz in enumerate(dropzones):
            label_clean = category_map.get(str(idx))
            correct_elem_ids = dz.get("correctElements", [])
            
            if label_clean and correct_elem_ids:
                correct_items = []
                for elem_id in correct_elem_ids:
                    elem_text = element_map.get(str(elem_id))
                    if elem_text:
                        correct_items.append(elem_text)
                
                if correct_items:
                    correct_mappings[label_clean] = correct_items
        
        if categories and draggable_items and correct_mappings:
            return cls(
                type=library,
                question=question_text,
                categories=categories,
                draggable_items=draggable_items,
                correct_mappings=correct_mappings
            )
        
        return None
    
    def to_text(self) -> str:
        question_clean = strip_html(self.question)
        categories_clean = [strip_html(c) for c in self.categories]
        items_clean = [strip_html(i) for i in self.draggable_items]
        
        result = (
            f"[Drag & Drop] {question_clean}\n"
            f"Kategorien: {', '.join(categories_clean)}\n"
            f"Elemente: {', '.join(items_clean)}\n\n"
            f"Korrekte Zuordnung:\n"
        )
        
        for category, items in self.correct_mappings.items():
            category_clean = strip_html(category)
            items_clean_list = [strip_html(item) for item in items]
            result += f"  {category_clean}: {', '.join(items_clean_list)}\n"
        
        return result
    
@dataclass
class ImageHotspotQuestion(H5PLeaf):
    """
    H5P.ImageHotspot oder H5P.DragDrop mit visuellen Elementen auf einem Bild.
    Elemente (Texte) werden zu DropZones (Labels) zugeordnet.
    """
    type: str  # "H5P.ImageHotspot" oder ähnlich
    mappings: list[tuple[str, str]] = field(default_factory=list)  # (element_text, dropzone_label)
    
    @classmethod
    def from_h5p_package(cls, module, content: dict, h5p_zip_path: str, **kwargs) -> Optional[str]:
        """
        Handler für standalone Image Hotspot Question.
        Befüllt module.interactive_video mit Element-zu-Label Zuordnungen.
        """
        library = extract_library_from_h5p(h5p_zip_path) or "H5P.ImageHotspot"
        params = content
        
        question = cls.from_h5p_params(library, params)
        
        if question and question.mappings:
            # Speichere als dict (Dependency Inversion)
            module.interactive_video = {
                "video_url": "",
                "vimeo_id": None,
                "interactions": [question.to_text()]
            }
            return None
        
        return "Konnte Image Hotspot Question nicht extrahieren"
    
    @classmethod
    def from_h5p_params(cls, library: str, params: dict) -> Optional['ImageHotspotQuestion']:
        """
        Extrahiert Image Hotspot Question aus params.

        H5P.ImageHotspots-Struktur: params.hotspots[].header + params.hotspots[].content[].params.text
        """
        try:
            hotspots = params.get("hotspots", [])
            if not hotspots:
                return None

            mappings = []
            for hotspot in hotspots:
                header = strip_html(hotspot.get("header", "")).strip()
                content_texts = []
                for item in hotspot.get("content", []):
                    text_html = item.get("params", {}).get("text", "").strip()
                    if text_html:
                        content_texts.append(strip_html(text_html).strip())
                content = " ".join(content_texts).strip()
                if header or content:
                    mappings.append((header, content))

            if mappings:
                return cls(type=library, mappings=mappings)

        except Exception:
            pass

        return None

    def to_text(self) -> str:
        """Formatiert Hotspots als 'Header: Inhalt' Zeilen."""
        if not self.mappings:
            return "[Image Hotspot] Keine Hotspots gefunden"

        lines = []
        for header, content in self.mappings:
            if header and content:
                lines.append(f"{header}: {content}")
            elif header:
                lines.append(header)
            elif content:
                lines.append(content)
        return "\n".join(lines)
