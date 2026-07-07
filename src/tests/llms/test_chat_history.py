"""Tests for chat history truncation in KICampusAssistant."""

from unittest.mock import MagicMock, patch

import pytest

from src.llm.assistant import KICampusAssistant


# ---------------------------------------------------------------------------
# limit_chat_history
# ---------------------------------------------------------------------------

class TestLimitChatHistory:
    def setup_method(self):
        with patch("src.llm.assistant.build_no_vectordb_graph"), \
             patch("src.llm.assistant.build_simple_hop_graph"), \
             patch("src.llm.assistant.build_socratic_graph"):
            self.assistant = KICampusAssistant.__new__(KICampusAssistant)
            self.assistant.system_config = {}
            self.assistant.checkpointer = MagicMock()
            self.assistant.graph = MagicMock()

    def test_no_truncation_when_within_limit(self, make_chat_history):
        history = make_chat_history(3)  # 6 messages
        result = self.assistant.limit_chat_history(history, limit=6)
        assert len(result) == 6

    def test_truncates_to_last_n(self, make_chat_history):
        history = make_chat_history(5)  # 10 messages
        result = self.assistant.limit_chat_history(history, limit=6)
        assert len(result) == 6
        assert result == history[-6:]

    def test_empty_history_unchanged(self):
        assert self.assistant.limit_chat_history([], limit=6) == []

    def test_exactly_at_limit_unchanged(self, make_chat_history):
        history = make_chat_history(3)  # exactly 6
        result = self.assistant.limit_chat_history(history, limit=6)
        assert result == history

    def test_single_message_under_limit(self, make_chat_message):
        history = [make_chat_message("hello")]
        result = self.assistant.limit_chat_history(history, limit=6)
        assert result == history

    def test_large_history_truncated_correctly(self, make_chat_message):
        history = [make_chat_message(f"msg {i}") for i in range(20)]
        result = self.assistant.limit_chat_history(history, limit=6)
        assert len(result) == 6
        assert result[0].content == "msg 14"
        assert result[-1].content == "msg 19"
