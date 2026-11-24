"""Unit tests for LLM factory and GWDG fallback logic.

Covers:
- get_model() returns correct LLM type per Models enum value
- GWDG timeout triggers fallback to Azure
- GWDG unavailability window is applied correctly
- GWDG availability resets after timeout window expires
"""

import datetime
from unittest.mock import MagicMock, patch

import pytest
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.openai_like import OpenAILike

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.llm.objects.LLMs import LLM, Models, TIME_TO_RESET_UNAVAILABLE_STATUS


@pytest.fixture(autouse=True)
def reset_gwdg_state():
    """Reset GWDG class-level state before each test to avoid cross-test contamination."""
    LLM.gwdg_unavailable = False
    LLM.gwdg_unavailable_since = None
    yield
    LLM.gwdg_unavailable = False
    LLM.gwdg_unavailable_since = None


class TestGetModel:
    def test_azure_fallback_returns_azure_openai(self):
        llm = LLM()
        model = llm.get_model(Models.AZURE_FALLBACK)
        assert isinstance(model, AzureOpenAI)

    def test_mini_returns_azure_openai(self):
        llm = LLM()
        model = llm.get_model(Models.MINI)
        assert isinstance(model, AzureOpenAI)

    def test_gemma4_returns_openai_like(self):
        llm = LLM()
        model = llm.get_model(Models.GEMMA4_31B)
        assert isinstance(model, OpenAILike)

    def test_llama3_returns_openai_like(self):
        llm = LLM()
        model = llm.get_model(Models.LLAMA3)
        assert isinstance(model, OpenAILike)

    def test_unknown_model_raises_value_error(self):
        llm = LLM()
        with pytest.raises((ValueError, Exception)):
            llm.get_model("unknown-model-xyz")


class TestGwdgFallback:
    def _mock_chat_response(self, text: str = "Antwort") -> SerializableChatMessage:
        return SerializableChatMessage(role="assistant", content=text)

    def _make_chat_engine(self, response_text: str = "Antwort"):
        engine = MagicMock()
        engine.chat.return_value = MagicMock(response=response_text)
        return engine

    def test_gwdg_exception_triggers_azure_fallback(self):
        """When the GWDG thread raises an exception, the Azure fallback model is used.

        We trigger the fallback via a thread exception rather than a real timeout,
        because a timeout-based test would require sleeping for TIME_TO_WAIT_FOR_GWDG (7s).
        The LLM.chat() code treats both timeout AND exception as GWDG failure.
        """
        llm = LLM()
        azure_engine = self._make_chat_engine("Azure Antwort")
        failing_engine = MagicMock()
        failing_engine.chat.side_effect = ConnectionError("GWDG nicht erreichbar")

        with patch("src.llm.objects.LLMs.SimpleChatEngine") as MockEngine:
            MockEngine.from_defaults.side_effect = [failing_engine, azure_engine]
            llm.chat(
                query="Test",
                chat_history=[],
                model=Models.GEMMA4_31B,
                system_prompt="Du bist ein Assistent.",
            )

        assert LLM.gwdg_unavailable is True
        assert LLM.gwdg_unavailable_since is not None

    def test_gwdg_marked_unavailable_routes_directly_to_azure(self):
        """If GWDG is already marked unavailable, the Azure model is used without trying GWDG."""
        LLM.gwdg_unavailable = True
        LLM.gwdg_unavailable_since = datetime.datetime.now()

        llm = LLM()
        azure_response = MagicMock(response="Azure Direkt")

        with patch("src.llm.objects.LLMs.SimpleChatEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.chat.return_value = azure_response
            MockEngine.from_defaults.return_value = mock_engine

            result = llm.chat(
                query="Testfrage",
                chat_history=[],
                model=Models.GEMMA4_31B,
                system_prompt="System prompt",
            )

        assert result.content == "Azure Direkt"
        # Only one engine was created (Azure, not GWDG)
        assert MockEngine.from_defaults.call_count == 1

    def test_gwdg_availability_resets_after_window(self):
        """After TIME_TO_RESET_UNAVAILABLE_STATUS passes, GWDG is considered available again."""
        LLM.gwdg_unavailable = True
        LLM.gwdg_unavailable_since = datetime.datetime.now() - datetime.timedelta(
            seconds=TIME_TO_RESET_UNAVAILABLE_STATUS + 1
        )

        llm = LLM()
        gwdg_response = MagicMock(response="GWDG Antwort")

        with patch("src.llm.objects.LLMs.SimpleChatEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.chat.return_value = gwdg_response
            MockEngine.from_defaults.return_value = mock_engine

            llm.chat(
                query="Test nach Reset",
                chat_history=[],
                model=Models.GEMMA4_31B,
                system_prompt="Prompt",
            )

        assert LLM.gwdg_unavailable is False
        assert LLM.gwdg_unavailable_since is None

    def test_gwdg_unavailability_not_reset_within_window(self):
        """GWDG stays marked unavailable while within the reset window."""
        unavailable_since = datetime.datetime.now() - datetime.timedelta(seconds=60)
        LLM.gwdg_unavailable = True
        LLM.gwdg_unavailable_since = unavailable_since

        llm = LLM()
        azure_response = MagicMock(response="Azure Antwort")

        with patch("src.llm.objects.LLMs.SimpleChatEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.chat.return_value = azure_response
            MockEngine.from_defaults.return_value = mock_engine

            llm.chat(
                query="Test",
                chat_history=[],
                model=Models.GEMMA4_31B,
                system_prompt="Prompt",
            )

        assert LLM.gwdg_unavailable is True


