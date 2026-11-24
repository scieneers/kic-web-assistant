import pytest
from llama_index.core.llms import MessageRole

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.llm.objects.LLMs import LLM, Models


@pytest.mark.integration
@pytest.mark.parametrize("model", [Models.AZURE_FALLBACK, Models.GEMMA4_31B])
def test_llm_responses(model: Models):
    llms = LLM()
    query = "What LLM model are you?"
    system_prompt = "You demonstrate a hello world example."
    chat_history = [
        SerializableChatMessage(content="Hello, this is a test.", role=MessageRole.USER),
        SerializableChatMessage(content="Hi", role=MessageRole.ASSISTANT),
    ]

    response = llms.chat(query=query, chat_history=chat_history, system_prompt=system_prompt, model=model)
    assert response is not None
    assert response.content is not None
    assert len(response.content) > 10
    assert response.role.value == "assistant"
    assert "error" not in response.content.lower() or len(response.content) > 50
