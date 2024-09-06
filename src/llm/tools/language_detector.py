from langfuse.decorators import observe
from lingua import Language, LanguageDetectorBuilder


class LanguageDetector:
    def __init__(self):
        languages = [
            Language.ENGLISH,
            Language.GERMAN,
        ]
        self.detector = LanguageDetectorBuilder.from_languages(*languages).build()

    @observe()
    def detect(self, text: str) -> str:
        language = self.detector.detect_language_of(text)
        if language is None:
            return "German"
        camelcase_name = language.name.capitalize()
        return camelcase_name
