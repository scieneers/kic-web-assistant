"""Streaming path of QuestionAnswerer: inline [docN] markers stream as short
numbered links, the descriptive source list follows below the answer
(testing feedback p. 17) — and the final message equals what was streamed."""

from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import TextNode

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.llm.objects.LLMs import Models
from src.llm.objects.question_answerer import QuestionAnswerer
from src.llm.streaming import citation_resolver_var, token_callback_var

A = "https://moodle.ki-campus.org/mod/page/view.php?id=1"
B = "https://moodle.ki-campus.org/mod/page/view.php?id=2"


@pytest.fixture
def answerer():
    with patch("src.llm.objects.question_answerer.load_prompt", return_value="system prompt {language}"):
        return QuestionAnswerer()


def _sources() -> list[TextNode]:
    return [
        TextNode(text="a", metadata={"title": "Wie entstehen Standards?", "url": A}),
        TextNode(text="b", metadata={"title": "Interoperabilität", "url": B}),
    ]


def _streaming_llm(text: str):
    """Mimics LLMs.chat in streaming mode: feeds deltas into the resolver the
    answerer registered, flushes, and returns the raw model text."""

    def chat(**_kwargs):
        resolver = citation_resolver_var.get()
        for i in range(0, len(text), 4):
            resolver.feed(text[i : i + 4])
        resolver.flush()
        return SerializableChatMessage(role="assistant", content=text)

    return chat


def _answer(answerer, llm_text: str, language: str = "German"):
    streamed: list[str] = []
    token = token_callback_var.set(streamed.append)
    try:
        with patch.object(answerer.llm, "chat", side_effect=_streaming_llm(llm_text)):
            response = answerer.answer_question(
                query="Wie entstehen Standards?",
                chat_history=[],
                sources=_sources(),
                model=Models.AZURE_FALLBACK,
                language=language,
                is_moodle=True,
                course_id=1,
            )
    finally:
        token_callback_var.reset(token)
    return response, "".join(streamed)


def test_streams_numbers_inline_and_list_at_end(answerer):
    response, streamed = _answer(
        answerer, "Zuerst werden Anforderungen formuliert.[doc1] Wichtig ist Interoperabilität.[doc2]"
    )
    assert streamed == response.content
    body, footer = streamed.split("\n\n**Zum Weiterlesen:**\n")
    assert body == (
        f'Zuerst werden Anforderungen formuliert.[<a href="{A}">1</a>] '
        f'Wichtig ist Interoperabilität.[<a href="{B}">2</a>]'
    )
    assert footer == f'1. <a href="{A}">Wie entstehen Standards?</a>\n2. <a href="{B}">Interoperabilität</a>'


def test_english_heading(answerer):
    response, _ = _answer(answerer, "Requirements come first.[doc1] More text follows here.", language="English")
    assert "\n\n**Further reading:**\n1. " in response.content


def test_answer_without_citations_gets_no_list(answerer):
    response, streamed = _answer(answerer, "Eine Antwort ganz ohne Quellenangabe, aber lang genug.")
    assert "**" not in streamed
    assert response.content == streamed


def test_no_answer_sentinel_gets_no_list(answerer):
    with patch("src.llm.objects.question_answerer.translate_fallback_text", side_effect=lambda t, _l: t):
        response, streamed = _answer(answerer, "NO ANSWER FOUND")
    assert streamed == ""
    assert "Weiterlesen" not in response.content


def test_non_streaming_answer_keeps_raw_markers_for_citation_node(answerer):
    """Without a token callback the answerer returns raw [docN]; parse_citations
    renders numbers + list later (covered in test_citation_parser)."""
    mock = MagicMock()
    mock.content = "Text [doc1] mit Quelle."
    with patch.object(answerer.llm, "chat", return_value=mock):
        response = answerer.answer_question(
            query="q", chat_history=[], sources=_sources(), model=Models.AZURE_FALLBACK,
            language="German", is_moodle=True, course_id=1,
        )
    assert response.content == "Text [doc1] mit Quelle."
