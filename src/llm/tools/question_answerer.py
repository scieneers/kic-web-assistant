from langfuse.decorators import observe
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore

from src.llm.LLMs import LLM, Models

SYSTEM_RULES = """
Some rules to follow:
1. Always answer based of the chat dialog and provided context information in the recent query, and not your general knowledge.
2. If you can't answer a question using the context, reply politely that you could not find information on the website. DO NOT make up your own answers.
3. Never directly reference the given context in your answer.
4. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.
5. You audience mostly speaks german or english, always answer in the language the user asks the question in.
6. Keep your answers short and simple, your answer will be shown in a chat bubble."""

SYSTEM_INTRO = """You are an expert Q&A system that is trusted around the world. \
You serve and help users of our website ki-campus.org. We are a learning platform funded by \
the German Federal Ministry of Education and Research for artificial intelligence with free online courses, \
videos and podcasts in various topics of AI and data literacy."""

SYSTEM_PROMPT_WEBSITE = """{system_intro}
{system_rules}"""

SYSTEM_PROMPT_COURSE = """{system_intro}
You specify on and are only knowledgeable about the specific contents of the course '{course_name}'. \
Be like a tutor and assist the user with their question, while trying to challenge them a bit to come up with the answer themselves. \
If they persist, you can also answer straightforwardly.
{system_rules}"""

SYSTEM_PROMPT_MODULE = """{system_intro}
You specify on and are only knowledgeable about a submodule of the course '{course_name}' called '{module_name}'. \
Be like a tutor and assist the user with their question, while trying to challenge them a bit to come up with the answer themselves. \
If they persist, you can also answer straightforwardly.
{system_rules}"""


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

    def get_sources_text(self, sources: list[NodeWithScore]) -> str:
        return "\n".join([source.get_text() for source in sources])

    @observe()
    def answer_question(
        self,
        query: str,
        chat_history: list[ChatMessage],
        sources: list[NodeWithScore],
        model: Models,
        system_prompt: str,
    ) -> ChatMessage:
        sources_text = self.get_sources_text(sources)
        prompted_user_query = USER_QUERY_WITH_SOURCES_PROMPT.format(context=sources_text, query=query)

        response = self.llm.chat(
            query=prompted_user_query, chat_history=chat_history, model=model, system_prompt=system_prompt
        )
        if response.content is None:
            raise ValueError(f"LLM produced no response. Please check the LLM implementation. Response: {response}")
        return response

    @observe()
    def answer_website_question(
        self, query: str, chat_history: list[ChatMessage], sources: list[NodeWithScore], model: Models
    ) -> ChatMessage:
        system_prompt = SYSTEM_PROMPT_WEBSITE.format(system_intro=SYSTEM_INTRO, system_rules=SYSTEM_RULES)
        response = self.answer_question(
            query=query, chat_history=chat_history, sources=sources, model=model, system_prompt=system_prompt
        )
        return response

    @observe()
    def answer_course_question(
        self,
        query: str,
        chat_history: list[ChatMessage],
        sources: list[NodeWithScore],
        model: Models,
        course_name: str,
        module_name: str | None = None,
    ) -> ChatMessage:
        if module_name is None:
            system_prompt = SYSTEM_PROMPT_COURSE.format(
                system_intro=SYSTEM_INTRO, system_rules=SYSTEM_RULES, course_name=course_name
            )
        else:
            system_prompt = SYSTEM_PROMPT_MODULE.format(
                system_intro=SYSTEM_INTRO, system_rules=SYSTEM_RULES, course_name=course_name, module_name=module_name
            )
        response = self.answer_question(
            query=query, chat_history=chat_history, sources=sources, model=model, system_prompt=system_prompt
        )
        return response
