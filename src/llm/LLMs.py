from enum import Enum

from langfuse.decorators import langfuse_context, observe
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import LLM as llama_llm
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.openai_like import OpenAILike

from src.env import env


class Models(str, Enum):
    GPT4 = "GPT-4"
    MISTRAL8 = "Mistral8"
    LLAMA3 = "Llama3"
    QWEN2 = "Qwen2"


class LLM:
    def get_embedder(self) -> AzureOpenAIEmbedding:
        embedder = AzureOpenAIEmbedding(
            model=env.AZURE_OPENAI_EMBEDDER_MODEL,
            deployment_name=env.AZURE_OPENAI_EMBEDDER_DEPLOYMENT,
            api_key=env.AZURE_OPENAI_API_KEY,
            azure_endpoint=env.AZURE_OPENAI_URL,
            api_version="2023-07-01-preview",
        )
        return embedder

    def get_model(self, model: Models) -> FunctionCallingLLM | llama_llm:
        match model:
            case Models.GPT4:
                llm = AzureOpenAI(
                    model=env.AZURE_OPENAI_GPT4_MODEL,
                    deployment=env.AZURE_OPENAI_GPT4_DEPLOYMENT,
                    api_key=env.AZURE_OPENAI_API_KEY,
                    azure_endpoint=env.AZURE_OPENAI_URL,
                    api_version="2023-05-15",
                    callback_manager=Settings.callback_manager,
                )
            case Models.MISTRAL8:
                llm = OpenAILike(
                    model="mixtral-8x7b-instruct",
                    is_chat_model=True,
                    temperature=0,
                    max_tokens=400,
                    api_key=env.GWDG_API_KEY,
                    api_base=env.GWDG_URL,
                    api_version="v1",
                    logprobs=None,
                    callback_manager=Settings.callback_manager,
                )
            case Models.LLAMA3:
                llm = OpenAILike(
                    model="meta-llama-3-70b-instruct",
                    is_chat_model=True,
                    temperature=0,
                    max_tokens=400,
                    api_key=env.GWDG_API_KEY,
                    api_base=env.GWDG_URL,
                    api_version="v1",
                    logprobs=None,
                    callback_manager=Settings.callback_manager,
                )
            case Models.QWEN2:
                llm = OpenAILike(
                    model="qwen2-72b-instruct",
                    is_chat_model=True,
                    temperature=0,
                    max_tokens=400,
                    api_key=env.GWDG_API_KEY,
                    api_base=env.GWDG_URL,
                    api_version="v1",
                    logprobs=None,
                    callback_manager=Settings.callback_manager,
                )
            case _:
                raise ValueError(f"Model '{model}' not yet supported")
        return llm

    @observe()
    def chat(self, query: str, chat_history: list[ChatMessage], model: Models, system_prompt: str) -> ChatMessage:
        langfuse_handler = langfuse_context.get_current_llama_index_handler()
        Settings.callback_manager = CallbackManager([langfuse_handler])

        llm = self.get_model(model)
        chat_engine = SimpleChatEngine.from_defaults(llm=llm, system_prompt=system_prompt)
        response = chat_engine.chat(message=query, chat_history=chat_history)
        if type(response.response) is not str:
            raise ValueError(f"Response is not a string. Please check the LLM implementation. Response: {response}")
        return ChatMessage(content=response.response, role=MessageRole.ASSISTANT)


if __name__ == "__main__":
    llm = LLM()

    response = llm.chat(
        query="Hello, this is a test. What model are you using?",
        chat_history=[],
        model=Models.GPT4,
        system_prompt="You are an assistant. Do what you do best.",
    )

    print(response)
