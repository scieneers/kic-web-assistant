"""Tests for FastAPI REST endpoints.

These tests use FastAPI's TestClient and mock the assistant + vector DB
so no real LLM or Azure AI Search calls are made.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from llama_index.core.llms import MessageRole

from src.api.models.serializable_chat_message import SerializableChatMessage

VALID_API_KEY = "test-key-123"


@pytest.fixture
def client():
    """TestClient with mocked singletons and a valid API key in env."""
    mock_assistant = MagicMock()
    mock_assistant.get_chat_history.return_value = []
    mock_assistant.chat.return_value = (
        SerializableChatMessage(role=MessageRole.ASSISTANT, content="Antwort"),
        "thread-abc",
    )
    mock_assistant.chat_with_course.return_value = (
        SerializableChatMessage(role=MessageRole.ASSISTANT, content="Kurs-Antwort"),
        "thread-abc",
    )

    mock_vector_db = MagicMock()
    mock_vector_db.check_if_course_exists.return_value = True
    mock_vector_db.check_if_module_exists.return_value = True

    with patch("src.api.rest.get_assistant", return_value=mock_assistant), \
         patch("src.api.rest.get_vector_db", return_value=mock_vector_db), \
         patch("src.api.rest.env") as mock_env:
        mock_env.REST_API_KEYS = [VALID_API_KEY]
        mock_env.DEBUG_MODE = False

        from src.api.rest import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c, mock_assistant


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_ok(self, client):
        c, _ = client
        response = c.get("/health")
        assert response.status_code == 200
        assert response.json() == "OK"


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class TestAuthentication:
    def test_missing_api_key_returns_403(self, client):
        c, _ = client
        response = c.post("/api/chat", json={
            "user_query": {"role": "user", "content": "Hallo"},
        })
        assert response.status_code in (401, 403)

    def test_invalid_api_key_returns_401(self, client):
        c, _ = client
        response = c.post(
            "/api/chat",
            json={"user_query": {"role": "user", "content": "Hallo"}},
            headers={"Api-Key": "wrong-key"},
        )
        assert response.status_code == 401

    def test_valid_api_key_passes_auth(self, client):
        c, _ = client
        response = c.post(
            "/api/chat",
            json={"user_query": {"role": "user", "content": "Hallo"}},
            headers={"Api-Key": VALID_API_KEY},
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# GET /api/chat/history/{thread_id}
# ---------------------------------------------------------------------------

class TestChatHistoryEndpoint:
    def test_empty_history_returns_empty_list(self, client):
        c, mock_assistant = client
        mock_assistant.get_chat_history.return_value = []
        response = c.get(
            "/api/chat/history/some-thread-id",
            headers={"Api-Key": VALID_API_KEY},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["thread_id"] == "some-thread-id"
        assert data["messages"] == []

    def test_existing_history_returned(self, client):
        c, mock_assistant = client
        mock_assistant.get_chat_history.return_value = [
            SerializableChatMessage(role=MessageRole.USER, content="Hallo"),
            SerializableChatMessage(role=MessageRole.ASSISTANT, content="Hi"),
        ]
        response = c.get(
            "/api/chat/history/thread-123",
            headers={"Api-Key": VALID_API_KEY},
        )
        assert response.status_code == 200
        messages = response.json()["messages"]
        assert len(messages) == 2
        assert messages[0]["content"] == "Hallo"
        assert messages[1]["content"] == "Hi"

    def test_unknown_thread_returns_empty(self, client):
        c, mock_assistant = client
        mock_assistant.get_chat_history.return_value = []
        response = c.get(
            "/api/chat/history/unknown-thread",
            headers={"Api-Key": VALID_API_KEY},
        )
        assert response.status_code == 200
        assert response.json()["messages"] == []


# ---------------------------------------------------------------------------
# POST /api/chat
# ---------------------------------------------------------------------------

class TestChatEndpoint:
    def test_returns_message_and_thread_id(self, client):
        c, _ = client
        response = c.post(
            "/api/chat",
            json={"user_query": {"role": "user", "content": "Was ist KI?"}},
            headers={"Api-Key": VALID_API_KEY},
        )
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "thread_id" in data
        assert "response_id" in data

    def test_module_id_without_course_id_returns_400(self, client):
        c, _ = client
        response = c.post(
            "/api/chat",
            json={
                "user_query": {"role": "user", "content": "Frage"},
                "module_id": 42,
            },
            headers={"Api-Key": VALID_API_KEY},
        )
        assert response.status_code == 400

    def test_thread_id_passed_through(self, client):
        c, mock_assistant = client
        c.post(
            "/api/chat",
            json={
                "user_query": {"role": "user", "content": "Frage"},
                "thread_id": "my-thread",
            },
            headers={"Api-Key": VALID_API_KEY},
        )
        call_kwargs = mock_assistant.chat.call_args
        assert call_kwargs.kwargs.get("thread_id") == "my-thread"
