from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore

from src.llm.LLMs import LLM, Models

SYSTEM_PROMPT = """You are an expert Q&A system that is trusted around the world. \
You serve and help users of our website ki-campus.org. We are a learning platform funded by \
the German Federal Ministry of Education and Research for artificial intelligence with free online courses, \
videos and podcasts in various topics of AI and data literacy. \

Some rules to follow:
1. Always answer based of the chat dialog and provided context information in the recent query, and not your general knowledge.
2. If you can't answer a question using the context, reply politely that you could not find information on the website. DO NOT make up your own answers.
3. Never directly reference the given context in your answer.
4. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.
5. You audience mostly speaks german or english, always answer in the language the user asks the question in.
6. Keep your answers short and simple, your answer will be shown in a chat bubble."""

USER_QUERY_WITH_SOURCES_PROMPT = """Context for answering the question:
---------------------
{context}
---------------------
Query: {query}

"""


class QuestionAnswerer:
    def __init__(self) -> None:
        self.name = "QuestionAnswer"
        self.llm = LLM()

    def answer_question(
        self, query: str, chat_history: list[ChatMessage], sources: list[NodeWithScore], model: Models
    ) -> str:
        sources_text = "\n".join([source.get_text() for source in sources])
        prompted_user_query = USER_QUERY_WITH_SOURCES_PROMPT.format(context=sources_text, query=query)

        chat = [ChatMessage(content=SYSTEM_PROMPT, role=MessageRole.SYSTEM)] + chat_history

        response = self.llm.chat(query=prompted_user_query, chat_history=chat, model=model)
        if response.content is None:
            raise ValueError(f"LLM produced no response. Please check the LLM implementation. Response: {response}")

        return response.content
