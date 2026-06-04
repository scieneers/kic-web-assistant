import sys

from langfuse.decorators import observe
from llama_index.core.llms import MessageRole
from llama_index.core.schema import TextNode

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.llm.objects.LLMs import LLM, Models
from src.llm.prompts.prompt_loader import load_prompt
from src.llm.streaming import StreamPhaseContext

ANSWER_NOT_FOUND_FIRST_TIME = """Entschuldige, ich habe deine Frage nicht ganz verstanden. Könntest du dein Problem bitte noch einmal etwas genauer erklären oder anders formulieren?
"""

ANSWER_NOT_FOUND_SECOND_TIME_DRUPAL = """Entschuldigung, ich habe deine Frage nicht immer noch verstanden, bitte wende dich an unseren Support unter support@ki-campus.org.
"""

ANSWER_NOT_FOUND_SECOND_TIME_MOODLE = """Es tut mir leid, aber ich konnte die benötigten Informationen im Kurs nicht finden, um deine Frage zu beantworten. Schau bitte im Kurs selbst nach, um weitere Hilfe zu erhalten. Hier ist der Kurslink: https://moodle.ki-campus.org/course/view.php?id={course_id}
"""

SHORT_SYSTEM_PROMPT = load_prompt("short_system_prompt")

SYSTEM_PROMPT = load_prompt("long_system_prompt")

USER_QUERY_WITH_SOURCES_PROMPT = """
[doc{index}]
Content: {content}
Metadata: {metadata}
"""


def format_sources(sources: list[TextNode], max_length: int = 8000) -> str:
    sources_text = ""
    for i, source in enumerate(sources):
        # Handle both TextNode (with get_text()) and SerializableTextNode (with .text attribute)
        content = source.get_text() if hasattr(source, 'get_text') else source.text
        source_entry = USER_QUERY_WITH_SOURCES_PROMPT.format(
            index=i + 1, content=content, metadata=source.metadata
        )
        # max_length must not exceed 8k for non-GPT models, otherwise the output will be garbled
        if len(sources_text) + len(source_entry) > max_length:
            break
        sources_text += source_entry + "\n"

    sources_text = sources_text.strip()

    return "<SOURCES>:\n" + sources_text


class QuestionAnswerer:
    def __init__(self) -> None:
        self.name = "QuestionAnswer"
        self.llm = LLM()

    @observe()
    def answer_question(
        self,
        query: str,
        chat_history: list[SerializableChatMessage],
        sources: list[TextNode],
        model: Models,
        language: str,
        is_moodle: bool,
        course_id: int,
    ) -> SerializableChatMessage:
        
        if model != Models.AZURE_FALLBACK:
            system_prompt = SHORT_SYSTEM_PROMPT.format(language=language)
            formatted_sources = format_sources(sources, max_length=8000)
        else:
            system_prompt = SYSTEM_PROMPT.format(language=language)
            formatted_sources = format_sources(sources, max_length=sys.maxsize)

        prompted_user_query = f"<QUERY>:\n {query}\n\n{formatted_sources}"

        with StreamPhaseContext("final"):
            response = self.llm.chat(
                query=prompted_user_query,
                chat_history=chat_history,
                model=model,
                system_prompt=system_prompt,
            )

        # Check if this is the second "NO ANSWER FOUND" in a row
        # Look for ASSISTANT messages (bot responses) in history to check if we already said we can't help
        previous_bot_response_was_no_answer = False
        if chat_history:
            # Find the last assistant message in the history
            for msg in reversed(chat_history):
                if msg.role == MessageRole.ASSISTANT:
                    previous_bot_response_was_no_answer = (msg.content == ANSWER_NOT_FOUND_FIRST_TIME)
                    break

        if response.content == "NO ANSWER FOUND":
            if not previous_bot_response_was_no_answer:
                # First time we can't answer - ask for clarification
                response.content = ANSWER_NOT_FOUND_FIRST_TIME
            else:
                # Second time in a row - provide support contact
                if is_moodle and course_id is not None:
                    response.content = ANSWER_NOT_FOUND_SECOND_TIME_MOODLE.format(course_id=course_id)
                elif is_moodle:
                    # Fallback if course_id is None
                    response.content = ANSWER_NOT_FOUND_SECOND_TIME_MOODLE.format(course_id="UNKNOWN")
                else:
                    response.content = ANSWER_NOT_FOUND_SECOND_TIME_DRUPAL

        if response is None:
            raise ValueError(f"LLM produced no response. Please check the LLM implementation. Response: {response}")
        
        return response
