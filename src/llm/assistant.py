from langfuse.decorators import observe
from llama_index.core.llms import ChatMessage

from src.llm.LLMs import Models
from src.llm.parser.citation_parser import CitationParser
from src.llm.retriever import KiCampusRetriever
from src.llm.tools.contextualizer import Contextualizer
from src.llm.tools.language_detector import LanguageDetector
from src.llm.tools.question_answerer import QuestionAnswerer


class KICampusAssistant:
    def __init__(self):
        self.retriever = KiCampusRetriever()

        self.contextualizer = Contextualizer()
        self.question_answerer = QuestionAnswerer()
        self.output_formatter = CitationParser()
        self.language_detector = LanguageDetector()

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

        user_language = self.language_detector.detect(query)

        response = self.question_answerer.answer_question(
            query=query,
            chat_history=limited_chat_history,
            language=user_language,
            sources=retrieved_chunks,
            model=model,
            is_moodle=False,
            course_id=None,
        )

        response.content = self.output_formatter.parse(answer=response.content, source_documents=retrieved_chunks)
        return response

    @observe()
    def chat_with_course(
        self,
        query: str,
        model: Models,
        course_id: int | None = None,
        chat_history: list[ChatMessage] = [],
        module_id: int | None = None,
    ) -> ChatMessage:
        """Chat with the contents of a specific course and optionally submodule. For frontend hosted on moodle."""

        limited_chat_history = self.limit_chat_history(chat_history, 10)

        rag_query = self.contextualizer.contextualize(query=query, chat_history=limited_chat_history, model=model)

        retrieved_chunks = self.retriever.retrieve(rag_query, course_id=course_id, module_id=module_id)

        user_language = self.language_detector.detect(query)

        # TODO course name and module name are unused. Save this metadata in the vectorDB
        response = self.question_answerer.answer_question(
            query=query,
            chat_history=limited_chat_history,
            language=user_language,
            sources=retrieved_chunks,
            model=model,
            is_moodle=True,
            course_id=course_id,
        )

        response.content = self.output_formatter.parse(answer=response.content, source_documents=retrieved_chunks)

        return response


if __name__ == "__main__":
    assistant = KICampusAssistant()
    assistant.chat(query="Eklär über den Kurs Deep Learning mit Tensorflow, Keras und Tensorflow.js", model=Models.GPT4)
