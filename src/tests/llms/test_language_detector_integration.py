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
