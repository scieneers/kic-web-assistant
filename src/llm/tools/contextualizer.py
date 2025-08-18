from langfuse.decorators import observe
from llama_index.core.llms import ChatMessage

from src.llm.LLMs import LLM, Models

CONDENSE_QUESTION_PROMPT = """
# CONTEXT #
You are a helpful AI subsystem that is part of a larger knowledge retrieval system tasked with refining user questions by adding
ONLY relevant context from the chat history, ensuring clarity, specificity, and self-containment.
Below you'll find the chat history followed by the current question.

# OBJECTIVE #
Reformulate the last question of a chat into a self-containing query.
The generated query will be used to query a knowledge database and retrieve documents.
To find the most relevant documents, the generated query must contain all relevant information including keywords.
Only the last question of the chat and related information should be reformulated into a query.
If the current question is already a self-containing query, return it unchanged.

# TONE #
Maintain a positive and motivational tone throughout, fostering a sense of empowerment and encouragement.
It should feel like a friendly guide offering valuable insights.

# AUDIENCE #
The documents that are in the knowledge base contain lectures and study materials for advanced students, studying property law.
To find the most relevant documents, style, tone and formality should be similar to the documents in the knowledge base.

# RESPONSE #
Query that can be used to retrieve documents from a knowledge base.

REMEMBER: Be precise and concise. Avoid unnecessary detail or unrelated additions. Do NOT add new sub-questions.
Ensure the question is understandable and retrieval-ready.
"""


class Contextualizer:
    """Contextualizes a message based on the chat history, so that it can effectively used as input for RAG retrieval."""

    def __init__(self):
        self.llm = LLM()

    @observe()
    def contextualize(self, query: str, chat_history: list[ChatMessage], model: Models) -> str:
        """Contextualize a message based on the chat history, so that it can effectively used as input for RAG retrieval."""

        if len(chat_history) == 0:
            return query

        contextualized_question = self.llm.chat(
            query=query, chat_history=chat_history, model=model, system_prompt=CONDENSE_QUESTION_PROMPT
        )
        if contextualized_question.content is None:
            raise ValueError(
                f"Contextualized question is None. Please check the LLM implementation. Response: {contextualized_question}"
            )

        return contextualized_question.content
