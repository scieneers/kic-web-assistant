from dataclasses import dataclass
from typing import Optional
from src.loaders.models.hp5activities import strip_html
from src.loaders.models.h5pactivities.h5p_base import H5PLeaf


@dataclass
class Summary(H5PLeaf):
    """Summary am Ende eines Interactive Videos."""
    type: str  # "H5P.Summary"
    intro: str
    statement_groups: list[list[str]]  # Jede Gruppe: [korrekte Aussage, falsche Aussagen...]
    
    @classmethod
    def from_h5p_params(cls, library: str, params: dict) -> Optional['Summary']:
        """Extrahiert Summary aus H5P params (aus Interaction)."""
        intro = params.get("intro", "").strip()
        summaries = params.get("summaries", [])
        
        statement_groups = []
        for summary_group in summaries:
            statements = summary_group.get("summary", [])
            if statements:
                # Erstes Statement ist korrekt, Rest sind falsch
                clean_statements = [s.strip() for s in statements if s.strip()]
                if clean_statements:
                    statement_groups.append(clean_statements)
        
        # Nur Summary erstellen wenn tatsächlich Statements vorhanden sind
        if statement_groups:
            return cls(
                type=library,
                intro=intro if intro else "Abschließende Summary-Frage",
                statement_groups=statement_groups
            )
        
        return None
    
    @classmethod
    def from_h5p_summary_data(cls, summary_data: dict) -> Optional['Summary']:
        """Extrahiert Summary aus H5P summary-Struktur (aus interactiveVideo.summary)."""
        if "task" not in summary_data:
            return None
        
        task_params = summary_data["task"].get("params", {})
        intro = task_params.get("intro", "").strip()
        summaries = task_params.get("summaries", [])
        
        statement_groups = []
        for summary_group in summaries:
            statements = summary_group.get("summary", [])
            if statements:
                # Erstes Statement ist korrekt, Rest sind falsch
                clean_statements = [s.strip() for s in statements if s.strip()]
                if clean_statements:
                    statement_groups.append(clean_statements)
        
        # Nur Summary erstellen wenn tatsächlich Statements vorhanden sind
        if statement_groups:
            return cls(
                type="H5P.Summary",
                intro=intro if intro else "Abschließende Summary-Frage",
                statement_groups=statement_groups
            )
        
        return None
    
    @classmethod
    def from_h5p_package(cls, module, content: dict, h5p_zip_path: str, **kwargs) -> Optional[str]:
        """
        Handler for standalone H5P Summary content.
        Summary is typically embedded in InteractiveVideo, not standalone.
        Returns error message or None on success.
        """
        params = content.get("params", {})
        summary = cls.from_h5p_params("H5P.Summary", params)
        
        if summary:
            module.content = summary.to_text()
            return None
        
        return "Summary konnte nicht extrahiert werden"

    def to_text(self) -> str:
        # Answer-free rendering: statements listed without Korrekt/Falsch
        # marking (and sorted, so position carries no signal) — the
        # correct-first grouping lives only in the structured QuizItem payload.
        intro_clean = strip_html(self.intro)
        result = f"[Abschluss] {intro_clean}\n"

        for i, statements in enumerate(self.statement_groups, 1):
            result += f"Aussagengruppe {i}:\n"
            for statement in sorted(strip_html(s) for s in statements):
                result += f" - {statement}\n"
            result += "\n"

        return result.strip()