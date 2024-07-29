from enum import Enum

from langfuse.decorators import observe
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
    # INTEL = "Intel"
    # LUMINOUS = "Luminous"


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
                )
            case _:
                raise ValueError(f"Model '{model}' not yet supported")

            # case Models.INTEL:
            #     llm = OpenAILike(
            #         model="intel-neural-chat-7b",
            #         is_chat_model=True,
            #         temperature=0,
            #         max_tokens=400,
            #         api_key=env.GWDG_API_KEY,
            #         api_base=env.GWDG_URL,
            #         api_version="v1",
            #         logprobs=None
            #     )
            # case Models.LUMINOUS: # This is not a chat model, would require different prompting
            #     llm = AlephAlpha(model="luminous-base-control",
            #                      token=env.ALEPH_ALPHA_KEY
            #     )
            # case LLM.llama3:
            #     # TODO: auf streaming umbauen?
            #     pass
            #     # llm = AzureAIStudioLlama2(
            #     #     api_key=secrets.AZURE_OPENAI_LLAMA2_KEY, endpoint=secrets.AZURE_OPENAI_LLAMA2_URL
            #     # )

        return llm

    @observe()
    def chat(self, query: str, chat_history: list[ChatMessage], model: Models) -> ChatMessage:
        llm = self.get_model(model)

        response = llm.chat(messages=chat_history + [ChatMessage(role=MessageRole.USER, content=query)])

        return response.message


if __name__ == "__main__":
    print("Hello, World!")
    llm = LLM()

    response = llm.chat(
        query="Hello, this is a test. What model are you using?",
        chat_history=[],
        model=Models.GPT4,
    )

    print(response)
