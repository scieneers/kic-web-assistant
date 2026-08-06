from dataclasses import dataclass, field
from typing import Optional
from src.loaders.models.hp5activities import strip_html, extract_library_from_h5p
from src.loaders.models.h5pactivities.h5p_base import H5PLeaf


@dataclass
class QuizQuestion(H5PLeaf):
    """Quiz-Frage (Multiple/Single Choice) im Interactive Video."""
    type: str  # "H5P.MultiChoice" oder "H5P.SingleChoiceSet"
    question: str
    correct_answers: list[str]
    incorrect_answers: list[str] = field(default_factory=list)
    
    @classmethod
    def from_h5p_package(cls, module, content: dict, h5p_zip_path: str, **kwargs) -> Optional[str]:
        """
        Handler für standalone H5P.MultiChoice / H5P.SingleChoiceSet.
        Befüllt module.interactive_video mit einer Quiz-Frage.
        
        Args:
            module: Module-Objekt zum Befüllen
            content: Geladenes content.json dict
            h5p_zip_path: Pfad zum H5P ZIP-File (nicht verwendet)
            **kwargs: Zusätzliche Services (nicht verwendet)
            
        Returns:
            Optional[str]: Fehlermeldung oder None
        """
        library = extract_library_from_h5p(h5p_zip_path)
        params = content
        
        quiz = cls.from_h5p_params(library, params)
        
        if quiz:
            # Speichere als dict (Dependency Inversion)
            module.interactive_video = {
                "video_url": "",
                "vimeo_id": None,
                "interactions": [quiz.to_text()]
            }
            return None
        
        return "Konnte Quiz-Frage nicht extrahieren"
    
    @classmethod
    def from_h5p_params(cls, library: str, params: dict) -> Optional['QuizQuestion']:
        """Extrahiert QuizQuestion aus H5P params."""
        # MultiChoice
        if "H5P.MultiChoice" in library:
            question_text = params.get("question", "").strip()
            
            correct = []
            incorrect = []
            for answer in params.get("answers", []):
                text = answer.get("text", "").strip()
                if text:
                    if answer.get("correct"):
                        correct.append(text)
                    else:
                        incorrect.append(text)
            
            if question_text and correct:
                return cls(
                    type=library,
                    question=question_text,
                    correct_answers=correct,
                    incorrect_answers=incorrect
                )
        
        # SingleChoiceSet
        elif "H5P.SingleChoiceSet" in library:
            choices = params.get("choices", [])
            results = []
            
            for choice in choices:
                question_text = choice.get("question", "").strip()
                answers = choice.get("answers", [])
                
                if question_text and answers:
                    # First answer is always correct in SingleChoiceSet
                    correct = [answers[0].strip()] if answers else []
                    incorrect = [a.strip() for a in answers[1:] if a.strip()]
                    
                    if correct:
                        results.append(cls(
                            type=library,
                            question=question_text,
                            correct_answers=correct,
                            incorrect_answers=incorrect
                        ))
            
            # Return first question or None
            return results[0] if results else None
        
        return None
    
    def to_text(self) -> str:
        # Answer-free rendering: this text ends up in the embedded module
        # document that normal chat retrieves — the correct/incorrect split
        # lives only in the structured QuizItem payload (h5p_payloads.py).
        # Options sorted alphabetically so the order carries no signal.
        question_clean = strip_html(self.question)
        options_clean = sorted(strip_html(a) for a in self.correct_answers + self.incorrect_answers)
        return f"[Quiz] {question_clean}\nAntwortoptionen: {', '.join(options_clean)}"


@dataclass
class TrueFalseQuestion(H5PLeaf):
    """Wahr/Falsch-Frage im Interactive Video."""
    type: str  # "H5P.TrueFalse"
    question: str
    correct_answer: bool
    
    @classmethod
    def from_h5p_package(cls, module, content: dict, h5p_zip_path: str, **kwargs) -> Optional[str]:
        """
        Handler für standalone H5P.TrueFalse.
        Befüllt module.interactive_video mit einer True/False-Frage.
        """
        library = extract_library_from_h5p(h5p_zip_path) or "H5P.TrueFalse"
        params = content
        
        question = cls.from_h5p_params(library, params)
        
        if question:
            # Speichere als dict (Dependency Inversion)
            module.interactive_video = {
                "video_url": "",
                "vimeo_id": None,
                "interactions": [question.to_text()]
            }
            return None
        
        return "Konnte True/False-Frage nicht extrahieren"
    
    @classmethod
    def from_h5p_params(cls, library: str, params: dict) -> Optional['TrueFalseQuestion']:
        """Extrahiert TrueFalseQuestion aus H5P params."""
        question_text = params.get("question", "").strip()
        correct_str = params.get("correct", "").lower()
        
        if question_text and correct_str in ["true", "false"]:
            correct_answer = (correct_str == "true")
            return cls(
                type=library,
                question=question_text,
                correct_answer=correct_answer
            )
        
        return None
    
    def to_text(self) -> str:
        # Answer-free rendering — see QuizQuestion.to_text; the solution lives
        # only in the structured QuizItem payload.
        question_clean = strip_html(self.question)
        return f"[Wahr/Falsch] {question_clean}"
