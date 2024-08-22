import pytest

from src.llm.LLMs import LLM, Models


@pytest.mark.parametrize("model", [Models.GPT4, Models.MISTRAL8, Models.LLAMA3, Models.QWEN2])  # type: ignore
def test_llm_responses(model: Models):
    llms = LLM()
    query = "Response with 'test'"
    system_prompt = "You demonstrate a hello world example."

    response = llms.chat(query=query, chat_history=[], system_prompt=system_prompt, model=model)
    assert response is not None
