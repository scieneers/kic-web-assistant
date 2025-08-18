import datetime
import threading
from enum import Enum

from langfuse.decorators import langfuse_context, observe
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import LLM as llama_llm
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_inference import AzureAICompletionsModel
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.openai_like import OpenAILike

from src.env import env

TIME_TO_WAIT_FOR_GWDG = 7  # in seconds
TIME_TO_RESET_UNAVAILABLE_STATUS = 60 * 5  # in seconds


class Models(str, Enum):
    GPT4 = "GPT-4"
    MISTRAL8 = "Mistral8"
    LLAMA3 = "Llama3"
    QWEN2 = "Qwen2"


class LLM:
    gwdg_unavailable = False
    gwdg_unavailable_since = None

    def get_embedder(self) -> AzureOpenAIEmbedding:
        embedder = AzureOpenAIEmbedding(
            model=env.AZURE_OPENAI_EMBEDDER_MODEL,
            deployment_name=env.AZURE_OPENAI_EMBEDDER_DEPLOYMENT,
            api_key=env.AZURE_OPENAI_API_KEY,
            azure_endpoint=env.AZURE_OPENAI_URL,
            api_version="2023-05-15",
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
                llm = AzureAICompletionsModel(
                    credential=env.AZURE_MISTRAL_KEY, endpoint=env.AZURE_MISTRAL_URL, model_name="mistral-large"
                )
                # GWDG instruct model for chat currently not working
                # llm = OpenAILike(
                #     model="mixtral-8x7b-instruct",
                #     is_chat_model=True,
                #     temperature=0,
                #     max_tokens=400,
                #     api_key=env.GWDG_API_KEY,
                #     api_base=env.GWDG_URL,
                #     api_version="v1",
                #     logprobs=None,
                #     callback_manager=Settings.callback_manager,
                # )
            case Models.LLAMA3:
                llm = OpenAILike(
                    model="llama-3.3-70b-instruct",
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

        if LLM.gwdg_unavailable and LLM.gwdg_unavailable_since:
            if datetime.datetime.now() - LLM.gwdg_unavailable_since > datetime.timedelta(
                seconds=TIME_TO_RESET_UNAVAILABLE_STATUS
            ):
                LLM.gwdg_unavailable = False
                LLM.gwdg_unavailable_since = None

        # If GWDG is unavailable, use GPT-4 instead
        if LLM.gwdg_unavailable:
            model = Models.GPT4

        llm = self.get_model(model)
        copy_chat_history = (
            chat_history.copy()
        )  # creating a copy of the history because the SimpleChatEngine modifies it
        # Only way of automatic tracing Langfuse is to use such an Engine. Direct calling llama_index models is not traced.
        chat_engine = SimpleChatEngine.from_defaults(
            llm=llm, system_prompt=system_prompt, chat_history=copy_chat_history
        )

        result = [None]  # Use a list to hold the result (mutable object to modify inside threads)

        def target():
            try:
                result.append(chat_engine.chat(message=query))  # Execute the chat function
            except Exception as e:
                result.append(e)  # If error, store the exception in the result

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=TIME_TO_WAIT_FOR_GWDG)

        if thread.is_alive() or isinstance(result[-1], Exception):
            LLM.gwdg_unavailable = True
            LLM.gwdg_unavailable_since = datetime.datetime.now()
            llm = self.get_model(Models.GPT4)
            chat_engine = SimpleChatEngine.from_defaults(
                llm=llm, system_prompt=system_prompt, chat_history=copy_chat_history
            )
            response = chat_engine.chat(message=query)
        else:
            response = result[-1]

        if isinstance(result[-1], Exception) or result[-1] is None:
            self.gwdg_unavailable = True
            llm = self.get_model(Models.GPT4)
            chat_engine = SimpleChatEngine.from_defaults(
                llm=llm, system_prompt=system_prompt, chat_history=copy_chat_history
            )
            response = chat_engine.chat(message=query)
        else:
            response = result[-1]

        if type(response.response) is not str:
            raise ValueError(f"Response is not a string. Please check the LLM implementation. Response: {response}")
        return ChatMessage(content=response.response, role=MessageRole.ASSISTANT)


if __name__ == "__main__":
    llm = LLM()

    response = llm.chat(
        query="Hello, this is a test. What model are you using?",
        chat_history=[],
        model=Models.LLAMA3,
        system_prompt="You are an assistant. Do what you do best.",
    )

    print(response)
