"""Shared fixtures for llms test suite."""

import pytest
from llama_index.core.llms import MessageRole
from llama_index.core.schema import TextNode

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.api.models.serializable_text_node import SerializableTextNode

# API key used across endpoint tests
TEST_API_KEY = "test-key-123"


@pytest.fixture
def make_serializable_node():
    """Factory for SerializableTextNode — used in reranker and graph tests."""
    def _factory(text: str = "content", score: float = 1.0, **metadata) -> SerializableTextNode:
        return SerializableTextNode(text=text, metadata=metadata, score=score)
    return _factory


@pytest.fixture
def make_text_node():
    """Factory for llama_index TextNode — used in citation parser tests."""
    def _factory(title=None, fullname=None, url=None, **extra) -> TextNode:
        meta = {k: v for k, v in {"title": title, "fullname": fullname, "url": url, **extra}.items() if v is not None}
        return TextNode(text="content", metadata=meta)
    return _factory


@pytest.fixture
def make_chat_message():
    """Factory for SerializableChatMessage."""
    def _factory(content: str, role: MessageRole = MessageRole.USER) -> SerializableChatMessage:
        return SerializableChatMessage(role=role, content=content)
    return _factory


@pytest.fixture
def make_chat_history(make_chat_message):
    """Factory for alternating user/assistant message lists of length 2*n."""
    def _factory(n: int) -> list[SerializableChatMessage]:
        messages = []
        for i in range(n):
            messages.append(make_chat_message(f"user {i}", MessageRole.USER))
            messages.append(make_chat_message(f"assistant {i}", MessageRole.ASSISTANT))
        return messages
    return _factory
