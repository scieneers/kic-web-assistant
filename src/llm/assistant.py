from langfuse.decorators import observe
from llama_index.core.llms import ChatMessage

from src.llm.LLMs import Models
from src.llm.retriever import KiCampusRetriever
from src.llm.tools.contextualizer import Contextualizer
from src.llm.tools.question_answerer import QuestionAnswerer


class KICampusAssistant:
    def __init__(self, verbose: bool = False):
        self.retriever = KiCampusRetriever()

        self.contextualizer = Contextualizer()
        self.question_answerer = QuestionAnswerer()

    @observe()
    def limit_chat_history(self, chat_history: list[ChatMessage], limit: int) -> list[ChatMessage]:
        if len(chat_history) > limit:
            chat_history = chat_history[-limit:]
        return chat_history

    @observe()
    def chat(self, query: str, model: Models, chat_history: list[ChatMessage] = []) -> ChatMessage:
        """Chat with general bot about drupal and functions of ki-campus. For frontend integrated drupal."""

        # Limiting context window to save resources
        limited_chat_history = self.limit_chat_history(chat_history, 10)

        rag_query = self.contextualizer.contextualize(query=query, chat_history=limited_chat_history, model=model)

        retrieved_chunks = self.retriever.retrieve(rag_query)

        response = self.question_answerer.answer_website_question(
            query=query, chat_history=limited_chat_history, sources=retrieved_chunks, model=model
        )

        return response

    @observe()
    def chat_with_course(
        self, query: str, model: Models, course: int, chat_history: list[ChatMessage] = [], module: int | None = None
    ) -> ChatMessage:
        """Chat with the contents of a specific course and optionally submodule. For frontend hosted on moodle."""

        pass
        # limited_chat_history = self.limit_chat_history(chat_history, 10)

        # rag_query = self.contextualizer.contextualize(query=query, chat_history=limited_chat_history, model=model)

        # retrieved_chunks = self.retriever.retrieve(rag_query, course=course, module=module)
        # course_name = self.retriever.get_course_name(course)
        # module_name = self.retriever.get_module_name(course, module) if module is not None else None

        # response = self.question_answerer.answer_course_question(
        #     query=query,
        #     chat_history=limited_chat_history,
        #     sources=retrieved_chunks,
        #     model=model,
        #     course_name=course_name,
        #     module_name=module_name
        # )

        # return response


if __name__ == "__main__":
    assistant = KICampusAssistant(verbose=True)
    assistant.chat(query="Eklär über den Kurs Deep Learning mit Tensorflow, Keras und Tensorflow.js", model=Models.GPT4)
