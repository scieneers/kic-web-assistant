from enum import Enum

from langfuse.decorators import observe
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.mistralai import MistralAI

from src.env import env


class Models(str, Enum):
    GPT4 = "GPT-4"
    MISTRAL = "Mistral"
    # llama3 = "Llama3"
    # luminous = "Luminous"


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

    def get_model(self, model: Models) -> FunctionCallingLLM:
        match model:
            case Models.GPT4:
                llm = AzureOpenAI(
                    model=env.AZURE_OPENAI_GPT4_MODEL,
                    engine=env.AZURE_OPENAI_GPT4_DEPLOYMENT,
                    api_key=env.AZURE_OPENAI_API_KEY,
                    azure_endpoint=env.AZURE_OPENAI_URL,
                    api_version="2023-05-15",
                )
            case Models.MISTRAL:
                llm = MistralAI(
                    api_key=env.AZURE_MISTRAL_KEY,
                    endpoint=env.AZURE_MISTRAL_URL,
                )
            case _:
                raise ValueError(f"Model '{model}' not yet supported")

            # case LLM.llama3:
            #     # TODO: auf streaming umbauen?
            #     pass
            #     # llm = AzureAIStudioLlama2(
            #     #     api_key=secrets.AZURE_OPENAI_LLAMA2_KEY, endpoint=secrets.AZURE_OPENAI_LLAMA2_URL
            #     # )
            # case LLM.luminous:
            #     llm = AlephAlpha(model="luminous-base-control")
        return llm

    @observe()
    def chat(self, query: str, chat_history: list[ChatMessage], model: Models) -> ChatMessage:
        llm = self.get_model(model)

        response = llm.chat(messages=chat_history + [ChatMessage(role=MessageRole.USER, content=query)])

        return response.message
