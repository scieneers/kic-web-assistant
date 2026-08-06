"""Integration tests for LanguageDetector.

These tests make real LLM calls (Azure Mini) and are marked with @pytest.mark.integration.
Run with: pytest src/tests/llms/test_language_detector_integration.py -v -m integration
"""

import pytest

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.llm.objects.language_detector import DEFAULT_LANGUAGE, LanguageDetector


@pytest.fixture(scope="module")
def detector() -> LanguageDetector:
    return LanguageDetector()


def _history(*pairs: tuple[str, str]) -> list[SerializableChatMessage]:
    """Build a chat history from (user, assistant) pairs."""
    messages = []
    for user_msg, assistant_msg in pairs:
        messages.append(SerializableChatMessage(role="user", content=user_msg))
        messages.append(SerializableChatMessage(role="assistant", content=assistant_msg))
    return messages


# --- Clear language in query ---

@pytest.mark.integration
def test_clear_german_query(detector: LanguageDetector):
    assert detector.detect("Was ist maschinelles Lernen?") == "German"


@pytest.mark.integration
def test_clear_english_query(detector: LanguageDetector):
    assert detector.detect("What is machine learning?") == "English"


@pytest.mark.integration
def test_clear_spanish_query(detector: LanguageDetector):
    assert detector.detect("¿Qué es el aprendizaje automático?") == "Spanish"


# --- Explicit language switch overrides everything ---

@pytest.mark.integration
def test_explicit_switch_to_english(detector: LanguageDetector):
    history = _history(
        ("Was ist KI?", "KI steht für Künstliche Intelligenz."),
        ("Erkläre mir neuronale Netze.", "Neuronale Netze sind..."),
    )
    assert detector.detect("Bitte antworte ab jetzt auf Englisch.", chat_history=history) == "English"


@pytest.mark.integration
def test_explicit_switch_to_german(detector: LanguageDetector):
    history = _history(
        ("What is AI?", "AI stands for Artificial Intelligence."),
    )
    assert detector.detect("Bitte antworte auf Deutsch.", chat_history=history) == "German"


@pytest.mark.integration
def test_explicit_switch_english_phrased_in_english(detector: LanguageDetector):
    assert detector.detect("Please answer in English from now on.") == "English"


# --- Short / ambiguous messages with empty history → must fall back to German ---

@pytest.mark.integration
@pytest.mark.parametrize("query", ["hi", "ok", "hey", "danke", "thanks", "ja", "nein", "yes"])
def test_short_message_empty_history_returns_german(detector: LanguageDetector, query: str):
    result = detector.detect(query, chat_history=[])
    assert result == DEFAULT_LANGUAGE, (
        f"Expected '{DEFAULT_LANGUAGE}' for short query {query!r} with empty history, got {result!r}"
    )


# --- Short / ambiguous messages with history → infer from history ---

@pytest.mark.integration
def test_short_message_with_german_history(detector: LanguageDetector):
    history = _history(
        ("Was ist Deep Learning?", "Deep Learning ist ein Teilgebiet des maschinellen Lernens."),
    )
    assert detector.detect("ok", chat_history=history) == "German"


@pytest.mark.integration
def test_short_message_with_english_history(detector: LanguageDetector):
    history = _history(
        ("What is deep learning?", "Deep learning is a subset of machine learning."),
    )
    assert detector.detect("ok", chat_history=history) == "English"


@pytest.mark.integration
def test_hi_with_german_history(detector: LanguageDetector):
    history = _history(
        ("Hallo, kannst du mir helfen?", "Natürlich! Wie kann ich helfen?"),
    )
    assert detector.detect("hi", chat_history=history) == "German"


@pytest.mark.integration
def test_hi_with_english_history(detector: LanguageDetector):
    history = _history(
        ("Hello, can you help me?", "Of course! How can I help?"),
    )
    assert detector.detect("hi", chat_history=history) == "English"


# --- Other languages ---

@pytest.mark.integration
def test_clear_french_query(detector: LanguageDetector):
    assert detector.detect("Qu'est-ce que l'apprentissage automatique?") == "French"


@pytest.mark.integration
def test_clear_turkish_query(detector: LanguageDetector):
    assert detector.detect("Makine öğrenmesi nedir?") == "Turkish"


@pytest.mark.integration
def test_clear_italian_query(detector: LanguageDetector):
    assert detector.detect("Cos'è l'apprendimento automatico?") == "Italian"


# --- Explicit language switch to French / Spanish ---

@pytest.mark.integration
def test_explicit_switch_to_french(detector: LanguageDetector):
    history = _history(
        ("Was ist KI?", "KI steht für Künstliche Intelligenz."),
    )
    assert detector.detect("Réponds-moi en français s'il te plaît.", chat_history=history) == "French"


@pytest.mark.integration
def test_explicit_switch_to_spanish(detector: LanguageDetector):
    assert detector.detect("Por favor, responde en español.") == "Spanish"


# --- Follow-up short message with Spanish history → Spanish ---

@pytest.mark.integration
def test_short_message_with_spanish_history(detector: LanguageDetector):
    history = _history(
        ("¿Qué cursos hay sobre ética?", "Hay varios cursos sobre ética en IA."),
    )
    assert detector.detect("ok", chat_history=history) == "Spanish"


# --- German text with English technical terms ---

@pytest.mark.integration
def test_german_text_with_english_terms(detector: LanguageDetector):
    # German sentence that naturally contains English tech terms — should still be German
    assert detector.detect("Wie funktioniert ein Transformer-Modell beim Fine-Tuning?") == "German"


@pytest.mark.integration
def test_german_text_with_many_english_terms(detector: LanguageDetector):
    assert detector.detect("Was ist der Unterschied zwischen Supervised Learning und Unsupervised Learning?") == "German"


# --- Language switch back to German after English conversation ---

@pytest.mark.integration
def test_switch_back_to_german_after_english(detector: LanguageDetector):
    history = _history(
        ("What is AI?", "AI stands for Artificial Intelligence."),
        ("Can you explain neural networks?", "Neural networks are..."),
    )
    assert detector.detect("Jetzt lieber auf Deutsch bitte.", chat_history=history) == "German"


# --- Long history in one language, new query in different language ---

@pytest.mark.integration
def test_explicit_query_language_overrides_history(detector: LanguageDetector):
    # 3 German exchanges, but the new query is clearly English → should return English
    history = _history(
        ("Was ist KI?", "KI steht für Künstliche Intelligenz."),
        ("Erkläre neuronale Netze.", "Neuronale Netze ahmen das Gehirn nach."),
        ("Was ist Deep Learning?", "Deep Learning ist ein Teilgebiet des maschinellen Lernens."),
    )
    assert detector.detect("What is the difference between AI and machine learning?", chat_history=history) == "English"


# --- Edge cases ---

@pytest.mark.integration
def test_only_numbers_empty_history(detector: LanguageDetector):
    # Numbers are language-neutral → should fall back to German
    assert detector.detect("42", chat_history=[]) == DEFAULT_LANGUAGE


@pytest.mark.integration
def test_only_numbers_with_german_history(detector: LanguageDetector):
    history = _history(
        ("Was ist die Antwort auf alles?", "Die Antwort auf alles ist 42."),
    )
    assert detector.detect("42", chat_history=history) == "German"
