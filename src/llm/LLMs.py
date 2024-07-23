from enum import Enum

from langfuse.decorators import observe
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.mistralai import MistralAI

from src.env import EnvHelper


class Models(str, Enum):
    GPT4 = "GPT-4"
    MISTRAL = "Mistral"
    # llama3 = "Llama3"
    # luminous = "Luminous"


class LLM:
    def __init__(self):
        self.secrets = EnvHelper()

    def get_model(self, model: Models) -> FunctionCallingLLM:
        match model:
            case Models.GPT4:
                llm = AzureOpenAI(
                    model=self.secrets.AZURE_OPENAI_GPT4_MODEL,
                    engine=self.secrets.AZURE_OPENAI_GPT4_DEPLOYMENT,
                    api_key=self.secrets.AZURE_OPENAI_GPT4_KEY,
                    azure_endpoint=self.secrets.AZURE_OPENAI_GPT4_URL,
                    api_version="2023-05-15",
                )
            case Models.MISTRAL:
                llm = MistralAI(
                    api_key=self.secrets.AZURE_OPENAI_MISTRAL_KEY,
                    endpoint=self.secrets.AZURE_OPENAI_MISTRAL_URL,
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
