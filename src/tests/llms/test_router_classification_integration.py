"""Integration tests for the Router/Contextualizer scenario classification.

These tests call the real LLM API and verify that router_prompt.txt correctly
classifies queries as no_vectordb or simple_hop.

Run with: pytest src/tests/llms/test_router_classification_integration.py -v -m integration
Skip in fast CI: pytest -m "not integration"
"""

import pytest

from src.llm.objects.contextualizer import Contextualizer
from src.llm.objects.LLMs import Models

MODEL = Models.AZURE_FALLBACK


@pytest.fixture(scope="module")
def contextualizer() -> Contextualizer:
    return Contextualizer()


# ---------------------------------------------------------------------------
# Out-of-scope queries — expected: no_vectordb
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parametrize("query", [
    pytest.param("Wie kocht man Spaghetti Carbonara?", id="cooking"),
    pytest.param("Was ist das Wetter heute in Berlin?", id="weather"),
    pytest.param("Wer hat die Fußball-WM 2022 gewonnen?", id="sports"),
    pytest.param("Schreib mir ein Gedicht über den Herbst.", id="creative_writing"),
    pytest.param("Was ist Photosynthese?", id="biology"),
    pytest.param("Wer ist der aktuelle Bundeskanzler von Deutschland?", id="politics"),
    pytest.param("Kannst du mir eine Reiseroute für Japan planen?", id="travel"),
])
def test_out_of_scope_classified_as_no_vectordb(contextualizer, query):
    result = contextualizer.classify_scenario(query=query, model=MODEL)
    assert result == "no_vectordb", f"'{query}' → expected no_vectordb, got '{result}'"


# ---------------------------------------------------------------------------
# Gibberish and small talk — expected: no_vectordb
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parametrize("query", [
    pytest.param("asdfkjhgfds", id="gibberish_chars"),
    pytest.param("jkjkjkjkjk", id="gibberish_repeat"),
    pytest.param("!!!???###", id="gibberish_symbols"),
    pytest.param("Hallo!", id="greeting"),
    pytest.param("Tschüss, bis später!", id="farewell"),
    pytest.param("Wie geht es dir?", id="small_talk_how_are_you"),
    pytest.param("Danke, das war sehr hilfreich!", id="gratitude"),
])
def test_gibberish_and_small_talk_classified_as_no_vectordb(contextualizer, query):
    result = contextualizer.classify_scenario(query=query, model=MODEL)
    assert result == "no_vectordb", f"'{query}' → expected no_vectordb, got '{result}'"


# ---------------------------------------------------------------------------
# Relevant AI / KI-Campus queries — expected: simple_hop
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parametrize("query", [
    pytest.param("Was ist maschinelles Lernen?", id="ml_definition"),
    pytest.param("Erkläre mir neuronale Netze.", id="neural_networks"),
    pytest.param("Welche Kurse gibt es auf KI-Campus?", id="ki_campus_courses"),
    pytest.param("Was sind die ethischen Risiken von künstlicher Intelligenz?", id="ai_ethics"),
    pytest.param("Wie funktioniert ein Transformer-Modell?", id="transformer"),
    pytest.param("Was ist der Unterschied zwischen supervised und unsupervised learning?", id="supervised_vs_unsupervised"),
    pytest.param("Wie kann KI in der Bildung eingesetzt werden?", id="ai_in_education"),
    pytest.param("Was ist ein Large Language Model?", id="llm_definition"),
    pytest.param("Erkläre mir Overfitting und wie man es vermeidet.", id="overfitting"),
    pytest.param("Was ist Reinforcement Learning?", id="reinforcement_learning"),
])
def test_relevant_queries_routed_to_rag(contextualizer, query):
    result = contextualizer.classify_scenario(query=query, model=MODEL)
    assert result == "simple_hop", (
        f"'{query}' → expected simple_hop, got '{result}'"
    )


# ---------------------------------------------------------------------------
# Borderline cases — document expected behavior explicitly
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_conversational_followup_classified_as_no_vectordb(contextualizer):
    """Follow-up requests referencing prior conversation should not trigger RAG."""
    result = contextualizer.classify_scenario(
        query="Kannst du das nochmal einfacher erklären?", model=MODEL
    )
    assert result == "no_vectordb"


@pytest.mark.integration
def test_ki_campus_specific_question_routed_to_rag(contextualizer):
    """Questions about KI-Campus content should always reach the vector DB."""
    result = contextualizer.classify_scenario(
        query="Was lerne ich im Kurs 'KI für Einsteiger' auf KI-Campus?", model=MODEL
    )
    assert result == "simple_hop"


@pytest.mark.integration
def test_generic_capability_question_classified_as_no_vectordb(contextualizer):
    """'Was kannst du?' without course context is small talk, not a RAG query."""
    result = contextualizer.classify_scenario(
        query="Was kannst du alles?", model=MODEL
    )
    assert result == "no_vectordb"
