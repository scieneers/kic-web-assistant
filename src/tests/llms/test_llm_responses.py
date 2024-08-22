import pytest
from llama_index.core.llms import ChatMessage, MessageRole

from src.llm.LLMs import LLM, Models


@pytest.mark.parametrize("model", [Models.GPT4, Models.MISTRAL8, Models.LLAMA3, Models.QWEN2])  # type: ignore
def test_llm_responses(model: Models):
    llms = LLM()
    query = "What LLM model are you?"
    system_prompt = "You demonstrate a hello world example."
    chat_history = [
        ChatMessage(content="Hello, this is a test.", role=MessageRole.USER),
        ChatMessage(content="Hi", role=MessageRole.ASSISTANT),
    ]

    response = llms.chat(query=query, chat_history=chat_history, system_prompt=system_prompt, model=model)
    assert response is not None
