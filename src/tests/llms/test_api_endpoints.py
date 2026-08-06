"""Tests for FastAPI REST endpoints.

These tests use FastAPI's TestClient and mock the assistant + vector DB
so no real LLM or Azure AI Search calls are made.
"""

import json
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


class TestChatEndpointModuleIdList:
    def test_module_id_list_is_passed_through(self, client):
        c, mock_assistant = client
        response = c.post(
            "/api/chat",
            json={
                "user_query": {"role": "user", "content": "Frage"},
                "course_id": 79,
                "module_id": [1, 33, 102],
            },
            headers={"Api-Key": VALID_API_KEY},
        )
        assert response.status_code == 200
        call_kwargs = mock_assistant.chat_with_course.call_args
        assert call_kwargs.kwargs.get("module_id") == [1, 33, 102]

    def test_empty_module_id_list_means_no_module_filter(self, client):
        c, mock_assistant = client
        response = c.post(
            "/api/chat",
            json={
                "user_query": {"role": "user", "content": "Frage"},
                "course_id": 79,
                "module_id": [],
            },
            headers={"Api-Key": VALID_API_KEY},
        )
        assert response.status_code == 200
        call_kwargs = mock_assistant.chat_with_course.call_args
        assert call_kwargs.kwargs.get("module_id") is None

    def test_module_id_list_without_course_id_returns_400(self, client):
        c, _ = client
        response = c.post(
            "/api/chat",
            json={
                "user_query": {"role": "user", "content": "Frage"},
                "module_id": [1, 33],
            },
            headers={"Api-Key": VALID_API_KEY},
        )
        assert response.status_code == 400

    def test_module_id_list_with_unknown_id_returns_400(self, client):
        c, _ = client
        with patch("src.api.rest.get_vector_db") as mock_get_vector_db:
            mock_db = MagicMock()
            mock_db.check_if_course_exists.return_value = True
            mock_db.check_if_module_exists.side_effect = lambda m: m != 999
            mock_get_vector_db.return_value = mock_db

            response = c.post(
                "/api/chat",
                json={
                    "user_query": {"role": "user", "content": "Frage"},
                    "course_id": 79,
                    "module_id": [1, 999],
                },
                headers={"Api-Key": VALID_API_KEY},
            )
        assert response.status_code == 400
        assert "999" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /api/chat/stream  (NDJSON)
# ---------------------------------------------------------------------------

def _parse_ndjson(text: str) -> list[dict]:
    return [json.loads(line) for line in text.strip().split("\n") if line.strip()]


def _make_streaming_chat(tokens: list[str], final_content: str, returned_thread_id: str = "generated-thread"):
    """Side effect for mock_assistant.chat/chat_with_course that pushes tokens
    through the real token_callback_var context — exercising rest.py's actual
    worker-thread/queue/NDJSON wiring instead of just its return value."""
    from src.llm.streaming import token_callback_var

    def _side_effect(*, query, model, thread_id, **kwargs):
        callback = token_callback_var.get()
        if callback is not None:
            for token in tokens:
                callback(token)
        return (
            SerializableChatMessage(role=MessageRole.ASSISTANT, content=final_content),
            thread_id or returned_thread_id,
        )

    return _side_effect


class TestChatStreamEndpoint:
    def test_requires_api_key(self, client):
        c, _ = client
        response = c.post(
            "/api/chat/stream",
            json={"user_query": {"role": "user", "content": "Hallo"}},
        )
        assert response.status_code in (401, 403)

    def test_ndjson_content_type(self, client):
        c, mock_assistant = client
        mock_assistant.chat.side_effect = _make_streaming_chat(["Hi"], "Hi")
        response = c.post(
            "/api/chat/stream",
            json={"user_query": {"role": "user", "content": "Hallo"}},
            headers={"Api-Key": VALID_API_KEY},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")

    def test_meta_then_tokens_then_final(self, client):
        c, mock_assistant = client
        mock_assistant.chat.side_effect = _make_streaming_chat(["Hallo ", "Welt"], "Hallo Welt")
        response = c.post(
            "/api/chat/stream",
            json={"user_query": {"role": "user", "content": "Frage"}},
            headers={"Api-Key": VALID_API_KEY},
        )
        events = _parse_ndjson(response.text)

        assert events[0]["type"] == "meta"
        assert events[-1]["type"] == "final"
        token_events = [e for e in events[1:-1] if e["type"] == "token"]
        assert [e["token"] for e in token_events] == ["Hallo ", "Welt"]
        assert events[-1]["message"] == "Hallo Welt"

    def test_meta_and_final_share_thread_and_response_id(self, client):
        c, mock_assistant = client
        mock_assistant.chat.side_effect = _make_streaming_chat(["x"], "x")
        response = c.post(
            "/api/chat/stream",
            json={"user_query": {"role": "user", "content": "Frage"}},
            headers={"Api-Key": VALID_API_KEY},
        )
        events = _parse_ndjson(response.text)
        meta, final = events[0], events[-1]
        assert meta["thread_id"] == final["thread_id"]
        assert meta["response_id"] == final["response_id"]

    def test_generates_thread_id_when_none_given(self, client):
        c, mock_assistant = client
        mock_assistant.chat.side_effect = _make_streaming_chat(["x"], "x")
        response = c.post(
            "/api/chat/stream",
            json={"user_query": {"role": "user", "content": "Frage"}},
            headers={"Api-Key": VALID_API_KEY},
        )
        meta = _parse_ndjson(response.text)[0]
        assert meta["thread_id"]

    def test_given_thread_id_is_used_for_chat_call(self, client):
        c, mock_assistant = client
        mock_assistant.chat.side_effect = _make_streaming_chat(["x"], "x")
        response = c.post(
            "/api/chat/stream",
            json={"user_query": {"role": "user", "content": "Frage"}, "thread_id": "fixed-thread"},
            headers={"Api-Key": VALID_API_KEY},
        )
        events = _parse_ndjson(response.text)
        assert events[0]["thread_id"] == "fixed-thread"
        assert events[-1]["thread_id"] == "fixed-thread"
        assert mock_assistant.chat.call_args.kwargs["thread_id"] == "fixed-thread"

    def test_course_scoped_request_uses_chat_with_course(self, client):
        c, mock_assistant = client
        mock_assistant.chat_with_course.side_effect = _make_streaming_chat(["x"], "x")
        c.post(
            "/api/chat/stream",
            json={"user_query": {"role": "user", "content": "Frage"}, "course_id": 79},
            headers={"Api-Key": VALID_API_KEY},
        )
        mock_assistant.chat_with_course.assert_called_once()
        mock_assistant.chat.assert_not_called()
        assert mock_assistant.chat_with_course.call_args.kwargs["course_id"] == 79

    def test_exception_during_streaming_emits_error_event(self, client):
        c, mock_assistant = client
        mock_assistant.chat.side_effect = RuntimeError("boom")
        response = c.post(
            "/api/chat/stream",
            json={"user_query": {"role": "user", "content": "Frage"}},
            headers={"Api-Key": VALID_API_KEY},
        )
        events = _parse_ndjson(response.text)
        assert events[0]["type"] == "meta"
        assert events[-1]["type"] == "error"
        assert "boom" in events[-1]["message"]
        # No final/success event follows a mid-stream failure.
        assert not any(e["type"] == "final" for e in events)

    def test_no_tokens_still_emits_meta_and_final(self, client):
        """A response short enough to never hit the streaming callback (e.g. a
        cached/instant answer) must still produce a well-formed NDJSON body."""
        c, mock_assistant = client
        mock_assistant.chat.side_effect = _make_streaming_chat([], "Sofortige Antwort")
        response = c.post(
            "/api/chat/stream",
            json={"user_query": {"role": "user", "content": "Frage"}},
            headers={"Api-Key": VALID_API_KEY},
        )
        events = _parse_ndjson(response.text)
        assert [e["type"] for e in events] == ["meta", "final"]
        assert events[-1]["message"] == "Sofortige Antwort"


# ---------------------------------------------------------------------------
# POST /api/feedback
# ---------------------------------------------------------------------------

class TestFeedbackEndpoint:
    def test_requires_api_key(self, client):
        c, _ = client
        response = c.post("/api/feedback", json={"response_id": "trace-1", "score": 1})
        assert response.status_code in (401, 403)

    def test_valid_feedback_forwards_to_langfuse_score(self, client):
        c, _ = client
        with patch("src.api.rest.Langfuse") as mock_langfuse_cls:
            response = c.post(
                "/api/feedback",
                json={"response_id": "trace-1", "score": 1, "feedback": "Sehr hilfreich"},
                headers={"Api-Key": VALID_API_KEY},
            )
        assert response.status_code == 200
        mock_langfuse_cls.return_value.score.assert_called_once_with(
            trace_id="trace-1",
            name="user-explicit-feedback",
            value=1,
            comment="Sehr hilfreich",
        )

    def test_feedback_text_is_optional(self, client):
        c, _ = client
        with patch("src.api.rest.Langfuse") as mock_langfuse_cls:
            response = c.post(
                "/api/feedback",
                json={"response_id": "trace-1", "score": 0},
                headers={"Api-Key": VALID_API_KEY},
            )
        assert response.status_code == 200
        assert mock_langfuse_cls.return_value.score.call_args.kwargs["comment"] is None

    @pytest.mark.parametrize("score", [-1, 2, 5])
    def test_score_outside_0_1_returns_400(self, client, score):
        c, _ = client
        with patch("src.api.rest.Langfuse"):
            response = c.post(
                "/api/feedback",
                json={"response_id": "trace-1", "score": score},
                headers={"Api-Key": VALID_API_KEY},
            )
        assert response.status_code == 400

    def test_missing_response_id_returns_422(self, client):
        c, _ = client
        response = c.post(
            "/api/feedback",
            json={"score": 1},
            headers={"Api-Key": VALID_API_KEY},
        )
        assert response.status_code == 422
