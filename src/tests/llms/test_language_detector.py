import pytest

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.llm.objects.language_detector import DEFAULT_LANGUAGE, LanguageDetector


@pytest.fixture
def language_detector() -> LanguageDetector:
    # __init__ only loads the prompt file and constructs an LLM client (no
    # network). The actual model call is stubbed per-test via detector.llm.chat.
    return LanguageDetector()


def _stub_chat(content: str):
    """Build a replacement for LLM.chat that returns a fixed model output."""

    def _chat(**kwargs) -> SerializableChatMessage:
        return SerializableChatMessage(role="assistant", content=content)

    return _chat


# The LLM is mocked, so these assert plumbing — the detector returns the model's
# answer verbatim (trimmed), not that the model classifies correctly.
@pytest.mark.parametrize(
    "model_output, expected",
    [
        ("English", "English"),
        ("German", "German"),
        ("Spanish", "Spanish"),
        ("  English \n", "English"),  # surrounding whitespace is stripped
    ],
)
def test_returns_detected_language(
    language_detector: LanguageDetector, model_output: str, expected: str
):
    language_detector.llm.chat = _stub_chat(model_output)
    assert language_detector.detect("any query") == expected


def test_falls_back_on_empty_output(language_detector: LanguageDetector):
    language_detector.llm.chat = _stub_chat("   ")
    assert language_detector.detect("hi") == DEFAULT_LANGUAGE


def test_falls_back_when_model_leaks_a_sentence(language_detector: LanguageDetector):
    # A full sentence must not be injected into the answer's system prompt.
    language_detector.llm.chat = _stub_chat("The language of the query is clearly German.")
    assert language_detector.detect("Was ist das?") == DEFAULT_LANGUAGE


def test_falls_back_on_llm_error(language_detector: LanguageDetector):
    def _raise(**kwargs):
        raise RuntimeError("mini deployment unavailable")

    language_detector.llm.chat = _raise
    assert language_detector.detect("any query") == DEFAULT_LANGUAGE
