import datetime
import threading
from enum import Enum

from langfuse.decorators import langfuse_context, observe
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import LLM as llama_llm
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_inference import AzureAICompletionsModel
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.openai_like import OpenAILike

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.env import env
from src.llm.streaming import stream_phase_var, token_callback_var

TIME_TO_WAIT_FOR_GWDG = 7  # in seconds
TIME_TO_RESET_UNAVAILABLE_STATUS = 60 * 5  # in seconds

class Models(str, Enum):
    AZURE_FALLBACK = "Azure-Fallback"
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
            api_version="2024-12-01-preview",
        )
        return embedder

    def get_model(self, model: Models) -> FunctionCallingLLM | llama_llm:
        match model:
            case Models.AZURE_FALLBACK:
                llm = AzureOpenAI(
                    model=env.AZURE_FALLBACK_MODEL,
                    deployment=env.AZURE_FALLBACK_DEPLOYMENT,
                    api_key=env.AZURE_OPENAI_API_KEY,
                    azure_endpoint=env.AZURE_OPENAI_URL,
                    api_version="2024-12-01-preview",
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
    def chat(self, query: str, chat_history: list[SerializableChatMessage], model: Models, system_prompt: str) -> SerializableChatMessage:
        """Chat with the selected LLM.

        If a request-scoped token callback is set (via ``token_callback_var``),
        this method will *stream* token deltas to that callback while also
        returning the final assembled message.
        """
        langfuse_handler = langfuse_context.get_current_llama_index_handler()
        Settings.callback_manager = CallbackManager([langfuse_handler] if langfuse_handler else [])

        token_callback = token_callback_var.get()
        stream_phase = stream_phase_var.get()

        if LLM.gwdg_unavailable and LLM.gwdg_unavailable_since:
            if datetime.datetime.now() - LLM.gwdg_unavailable_since > datetime.timedelta(
                seconds=TIME_TO_RESET_UNAVAILABLE_STATUS
            ):
                LLM.gwdg_unavailable = False
                LLM.gwdg_unavailable_since = None

        # If GWDG is unavailable, use Azure fallback model instead
        if LLM.gwdg_unavailable:
            model = Models.AZURE_FALLBACK

        llm = self.get_model(model)
        # Convert SerializableChatMessage to ChatMessage for SimpleChatEngine
        chat_history_messages = [msg.to_chat_message() for msg in chat_history]
        copy_chat_history = (
            chat_history_messages.copy()
        )  # creating a copy of the history because the SimpleChatEngine modifies it
        # Only way of automatic tracing Langfuse is to use such an Engine. Direct calling llama_index models is not traced.
        chat_engine = SimpleChatEngine.from_defaults(
            llm=llm, system_prompt=system_prompt, chat_history=copy_chat_history
        )

        # --- Streaming path (if enabled for this request) --------------------
        # We do not apply the GWDG timeout fallback when streaming, because the
        # timeout logic is implemented with threads that would swallow token
        # deltas. Instead we try streaming; if it fails, we fall back to a
        # single non-streaming completion and emit it as one chunk.
        if token_callback is not None and stream_phase == "final":
            try:
                streaming_resp = chat_engine.stream_chat(message=query)
                full_text = ""
                last_text = ""
                for chunk in streaming_resp.response_gen:
                    # LlamaIndex StreamingAgentChatResponse.response_gen can yield either:
                    #   1) `str` deltas (common for some wrappers), OR
                    #   2) `ChatResponse` objects with `.delta` / `.message.content`.
                    #
                    # We normalize both into a string `delta`.
                    if isinstance(chunk, str):
                        delta = chunk
                    else:
                        # Depending on LLM wrapper, incremental text might be in:
                        #   - chunk.delta (preferred)
                        #   - chunk.message.content (common)
                        delta = getattr(chunk, "delta", None)

                        if not delta:
                            msg = getattr(chunk, "message", None)
                            msg_text = getattr(msg, "content", None) if msg is not None else None
                            if isinstance(msg_text, str) and msg_text:
                                # Derive delta from the growing message content.
                                if msg_text.startswith(last_text):
                                    delta = msg_text[len(last_text) :]
                                else:
                                    # If the provider rewrites the whole string (rare),
                                    # fall back to emitting the full text.
                                    delta = msg_text
                                last_text = msg_text

                    if not delta:
                        continue

                    full_text += delta
                    token_callback(delta)

                # In some edge cases, streaming yields chunks but we still couldn't
                # derive deltas. Fall back to final response string.
                if not full_text:
                    full_text = getattr(streaming_resp, "response", "") or getattr(
                        streaming_resp, "unformatted_response", ""
                    )
                    if full_text:
                        token_callback(full_text)

                return SerializableChatMessage(role="assistant", content=full_text)
            except Exception:
                # Fall back to non-streaming and emit as a single chunk.
                response = chat_engine.chat(message=query)
                text = response.response if isinstance(response.response, str) else str(response.response)
                token_callback(text)
                return SerializableChatMessage(role="assistant", content=text)

        result = [None]  # Use a list to hold the result (mutable object to modify inside threads)

        def target():
            try:
                result.append(chat_engine.chat(message=query))  # Execute the chat function
            except Exception as e:
                result.append(e)  # If error, store the exception in the result

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=TIME_TO_WAIT_FOR_GWDG)

        if thread.is_alive() or isinstance(result[-1], Exception) or result[-1] is None:
            # GWDG timeout or error - fallback to Azure model
            LLM.gwdg_unavailable = True
            LLM.gwdg_unavailable_since = datetime.datetime.now()
            llm = self.get_model(Models.AZURE_FALLBACK)
            chat_engine = SimpleChatEngine.from_defaults(
                llm=llm, system_prompt=system_prompt, chat_history=copy_chat_history
            )
            response = chat_engine.chat(message=query)
        else:
            response = result[-1]

        if type(response.response) is not str:
            raise ValueError(f"Response is not a string. Please check the LLM implementation. Response: {response}")
        return SerializableChatMessage(role="assistant", content=response.response)


if __name__ == "__main__":
    llm = LLM()

    response = llm.chat(
        query="Hello, this is a test. What model are you using?",
        chat_history=[],
        model=Models.LLAMA3,
        system_prompt="You are an assistant. Do what you do best.",
    )

    print(response)
