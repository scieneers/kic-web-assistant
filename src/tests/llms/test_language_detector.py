import pytest

from src.llm.tools.language_detector import LanguageDetector


@pytest.fixture
def language_detector():
    return LanguageDetector()


examples = [
    ("What can you tell me about sport injuries?", "English"),
    ("Is 20 mio. the maximum contribution?", "English"),
    ("¿Qué lesiones deportivas se representan en nuestros productos?", "Spanish"),
    ("Welche Sportverletzungen sind in unseren Produkten abgebildet?", "German"),
    ("Welche Fragen kann ich dir stellen?", "German"),
    ("Welche Leistungen sind in der Unfallversicherung versichert?", "German"),
]


@pytest.mark.parametrize("text, expected_language", examples)
def test_language_detection(language_detector: LanguageDetector, text: str, expected_language: str):
    assert language_detector.detect(text) == expected_language
