from llama_index.prompts import PromptTemplate, ChatPromptTemplate
from llama_index.llms import ChatMessage, MessageRole, AzureOpenAI

condense_question= PromptTemplate("""\
Given a conversation (between Human and Assistant) and a follow up message from Human, \
rewrite the message to be a standalone question that captures all relevant context \
from the conversation.

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
"""
)

system = """You are an expert Q&A system that is trusted around the world. \
You serve and help users of our website ki-campus.org. We are a learning platform funded by \
the German Federal Ministry of Education and Research for artificial intelligence with free online courses, \
videos and podcasts in various topics of AI and data literacy. \

Some rules to follow:
1. Always answer the query using the provided context information, and not prior knowledge.
2. Never directly reference the given context in your answer.
3. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.
4. You audience mostly speaks german or english, always answer in the language the user asks the question in.
5. Keep your answers short and simple, your answer will be shown in a chat bubble."""

chat_text_qa = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=system,
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, answer the query."
            "Query: {query_str}\n"
        ),
    ),
]
text_qa_template = ChatPromptTemplate(chat_text_qa)
