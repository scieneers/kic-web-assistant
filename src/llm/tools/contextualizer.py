from langfuse.decorators import observe
from llama_index.core.llms import ChatMessage, MessageRole

from src.llm.LLMs import LLM, Models

CONDENSE_QUESTION_PROMPT = """You are a system that condenses a chat history for the retrieval step of a RAG System.
Given the following conversation, rephrase the latest message to be a standalone question. Do so by including all relevant context from the chat history needed for answering it.
If the latest message is already self-contained or does not relate to the chat history, repeat it without making any changes.
NEVER respond to the contents of the chat or the last message. Only follow the instructions above!
"""


class Contextualizer:
    """Contextualizes a message based on the chat history, so that it can effectively used as input for RAG retrieval."""

    def __init__(self):
        self.llm = LLM()

    @observe()
    def contextualize(self, query: str, chat_history: list[ChatMessage], model: Models) -> str:
        """Contextualize a message based on the chat history, so that it can effectively used as input for RAG retrieval."""

        chat = [ChatMessage(content=CONDENSE_QUESTION_PROMPT, role=MessageRole.SYSTEM)] + chat_history

        contextualized_question = self.llm.chat(query=query, chat_history=chat, model=model)
        if contextualized_question.content is None:
            raise ValueError(
                f"Contextualized question is None. Please check the LLM implementation. Response: {contextualized_question}"
            )

        return contextualized_question.content
