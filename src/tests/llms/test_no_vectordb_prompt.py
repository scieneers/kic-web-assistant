"""Smalltalk/no_vectordb prompt: assistant facts (Testing-Feedback S. 11).

The Socratic Tutor is only advertised when it is enabled in the deployment —
otherwise "Was kannst du?" would promise a feature that does not exist."""

from src.llm.graphs.no_vector_db import build_no_vectordb_prompt


def test_tutor_described_when_enabled():
    prompt = build_no_vectordb_prompt("German", {"enable_socratic_v2": True})
    assert "Sokratischer Tutor" in prompt
    assert "switch in the chat window" in prompt
    assert "{assistant_modes}" not in prompt and "{language}" not in prompt


def test_tutor_not_mentioned_when_disabled():
    prompt = build_no_vectordb_prompt("German", {"enable_socratic_v2": False})
    assert "Sokratischer Tutor" not in prompt
    assert "Socratic Tutor" not in prompt.split("# ABOUT THIS ASSISTANT #")[1].split("# RESPONSE INSTRUCTIONS #")[0]
    assert "{assistant_modes}" not in prompt


def test_missing_system_config_defaults_to_chat_only():
    prompt = build_no_vectordb_prompt("English", None)
    assert "Sokratischer Tutor" not in prompt
    assert "respond in English" in prompt or "in English" in prompt
