from langfuse import Langfuse
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core.llms import ChatMessage

from src.env import EnvHelper
from src.llm.LLMs import Models
from src.llm.retriever import KiCampusRetriever
from src.llm.tools.contextualizer import Contextualizer
from src.llm.tools.question_answerer import QuestionAnswerer


class KICampusAssistant:
    def __init__(self, verbose: bool = False):
        self.retriever = KiCampusRetriever()
        secrets = EnvHelper()

        self.langfuse = Langfuse(
            secret_key=secrets.LANGFUSE_SECRET_KEY, public_key=secrets.LANGFUSE_PUBLIC_KEY, host=secrets.LANGFUSE_HOST
        )
        self.langfuse_callback_handler = LlamaIndexCallbackHandler(
            secret_key=secrets.LANGFUSE_SECRET_KEY, public_key=secrets.LANGFUSE_PUBLIC_KEY
        )
        self.contextualizer = Contextualizer()
        self.question_answerer = QuestionAnswerer()

    def limit_chat_history(self, chat_history: list[ChatMessage], limit: int) -> list[ChatMessage]:
        if len(chat_history) > limit:
            chat_history = chat_history[-limit:]
        return chat_history

    def chat(self, query: str, model: Models, chat_history: list[ChatMessage] = []) -> str:
        """Chat with general bot about drupal and functions of ki-campus."""
        # root_trace = self.langfuse.trace(name="llamaindex-rag-chat")
        # trace_id = root_trace.trace_id
        # self.langfuse_callback_handler.set_root(root_trace)

        # Limiting context window to save resources
        limited_chat_history = self.limit_chat_history(chat_history, 10)

        rag_query = self.contextualizer.contextualize(query=query, chat_history=limited_chat_history, model=model)

        retrieved_chunks = self.retriever.retrieve(rag_query)

        response = self.question_answerer.answer_question(
            query=query, chat_history=chat_history, sources=retrieved_chunks, model=model
        )

        return response

    def chat_with_course(
        self, query: str, course: int, chat_history: list[ChatMessage] = [], module: int | None = None
    ) -> str:
        """Chat with the contents of a specific course and optionally submodule"""
        pass

    def submit_feedback(self, feedback_response: dict, trace_id: str):
        value = 1 if feedback_response["score"] == "👍" else -1
        comment = feedback_response["text"]

        self.langfuse.score(trace_id=trace_id, name="user-explicit-feedback", value=value, comment=comment)


if __name__ == "__main__":
    assistant = KICampusAssistant(verbose=True)
    assistant.chat(query="Eklär über den Kurs Deep Learning mit Tensorflow, Keras und Tensorflow.js", model=Models.gpt4)
