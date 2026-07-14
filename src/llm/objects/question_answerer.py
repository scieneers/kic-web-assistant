import re
import sys

from langfuse.decorators import observe
from llama_index.core.llms import MessageRole
from llama_index.core.schema import TextNode

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.llm.objects.LLMs import LLM, Models
from src.llm.objects.citation_parser import CITATION_TEXT, _get_display_title, citation_suffix
from src.llm.prompts.prompt_loader import load_prompt
from src.llm.streaming import CitationStreamResolver, SmartStreamCallback, StreamPhaseContext, citation_resolver_var, token_callback_var

ANSWER_NOT_FOUND_FIRST_TIME = """Entschuldige, ich habe deine Frage nicht ganz verstanden. Könntest du dein Problem bitte noch einmal etwas genauer erklären oder anders formulieren?
"""

ANSWER_NOT_FOUND_SECOND_TIME_DRUPAL = """Entschuldigung, ich habe deine Frage nicht immer noch verstanden, bitte wende dich an unseren Support unter support@ki-campus.org.
"""

NO_RELEVANT_CONTENT = """Zu deiner Frage habe ich leider keine passenden Inhalte gefunden. Kannst du deine Frage anders formulieren oder konkretisieren?
"""

NO_CONTENT_IN_MODULE = "Das Modul '{module_name}' ist im Kurs vorhanden, enthält aber keinen weiteren Inhalt zu dem ich eine Antwort geben kann."

# Distinct from NO_CONTENT_IN_MODULE: the module DOES have content — we
# deliberately don't extract this content type (see
# Module.UNSUPPORTED_MODNAME_LABELS). Naming the reason instead of implying
# emptiness avoids the misleading impression that the module has nothing in it.
NO_CONTENT_UNSUPPORTED_TYPE = (
    "Das Modul '{module_name}' ist im Kurs vorhanden, aber sein Inhaltstyp ({label}) wird von mir "
    "aktuell nicht durchsucht. Schau bitte direkt im Kurs nach: {url}"
)

ANSWER_NOT_FOUND_SECOND_TIME_MOODLE = """Es tut mir leid, aber ich konnte die benötigten Informationen im Kurs nicht finden, um deine Frage zu beantworten. Schau bitte im Kurs selbst nach, um weitere Hilfe zu erhalten. Hier ist der Kurslink: https://moodle.ki-campus.org/course/view.php?id={course_id}
"""

# Statische Präfixe der Fallback-Templates mit Platzhaltern — aus den Konstanten
# abgeleitet, damit get_fallback_type() bei Textänderungen nicht auseinanderläuft.
_MOODLE_NOT_FOUND_PREFIX = ANSWER_NOT_FOUND_SECOND_TIME_MOODLE.split("{course_id}")[0]
_EMPTY_MODULE_PREFIX, _EMPTY_MODULE_SUFFIX = NO_CONTENT_IN_MODULE.split("{module_name}")
# Distinguishing fragment between the module_name and label placeholders —
# unique to NO_CONTENT_UNSUPPORTED_TYPE despite sharing the same "Das Modul
# '...'" opening as NO_CONTENT_IN_MODULE.
_UNSUPPORTED_TYPE_MARKER = NO_CONTENT_UNSUPPORTED_TYPE.split("{label}")[0].split("{module_name}")[1]


def get_fallback_type(answer: str | None) -> str | None:
    """Klassifiziert eine finale Antwort als bekannten Fallback-Text.

    Liefert einen kurzen Identifier für Tracing/Analyse (Langfuse-Metadaten
    und -Score) oder None, wenn die Antwort eine echte LLM-Antwort ist.
    """
    if not answer or not answer.strip():
        return "empty"
    text = answer.strip()
    if text == ANSWER_NOT_FOUND_FIRST_TIME.strip():
        return "not_understood_first_time"
    if text == NO_RELEVANT_CONTENT.strip():
        return "no_relevant_content"
    if text == ANSWER_NOT_FOUND_SECOND_TIME_DRUPAL.strip():
        return "not_found_second_time_drupal"
    if text.startswith(_MOODLE_NOT_FOUND_PREFIX):
        return "not_found_second_time_moodle"
    if text.startswith(_EMPTY_MODULE_PREFIX) and _UNSUPPORTED_TYPE_MARKER in text:
        return "unsupported_content_type"
    if text.startswith(_EMPTY_MODULE_PREFIX) and _EMPTY_MODULE_SUFFIX.strip() in text:
        return "empty_module"
    return None


SYSTEM_PROMPT = load_prompt("system_prompt")

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
        
        # Determine whether the previous assistant turn was already a "no answer"
        # before starting the LLM call — needed to pick the right fallback message.
        previous_bot_response_was_no_answer = False
        if chat_history:
            for msg in reversed(chat_history):
                if msg.role == MessageRole.ASSISTANT:
                    previous_bot_response_was_no_answer = (
                        msg.content in (ANSWER_NOT_FOUND_FIRST_TIME, NO_RELEVANT_CONTENT)
                    )
                    break

        # Early exit: no sources retrieved → skip both rerank and LLM answer call.
        if not sources:
            if not previous_bot_response_was_no_answer:
                fallback = NO_RELEVANT_CONTENT
            elif is_moodle and course_id is not None:
                fallback = ANSWER_NOT_FOUND_SECOND_TIME_MOODLE.format(course_id=course_id)
            elif is_moodle:
                fallback = ANSWER_NOT_FOUND_SECOND_TIME_MOODLE.format(course_id="UNKNOWN")
            else:
                fallback = ANSWER_NOT_FOUND_SECOND_TIME_DRUPAL
            return SerializableChatMessage(role=MessageRole.ASSISTANT, content=fallback)

        system_prompt = SYSTEM_PROMPT.format(language=language)
        formatted_sources = format_sources(sources, max_length=sys.maxsize)
        prompted_user_query = f"<QUERY>:\n {query}\n\n{formatted_sources}"

        # When streaming, replace [docN] markers with clickable links in real-time so
        # the streamed content already matches what the final event will send.
        outer_callback = token_callback_var.get()
        resolver = None
        smart_cb = None
        resolver_ctx = None

        if outer_callback is not None:
            seen_urls: set[str] = set()

            def _resolve(marker: str) -> str:
                m = re.match(r"\[doc(\d+)\]", marker)
                if not m:
                    return ""
                idx = int(m.group(1)) - 1
                try:
                    doc = sources[idx]
                except IndexError:
                    return ""
                url = doc.metadata.get("url", "")
                if url in seen_urls:
                    return ""
                seen_urls.add(url)
                return CITATION_TEXT.format(url=url, title=_get_display_title(doc), suffix=citation_suffix(doc))

            resolver = CitationStreamResolver(resolve=_resolve, callback=outer_callback)
            smart_cb = SmartStreamCallback(resolver=resolver, outer_callback=outer_callback)
            resolver_ctx = citation_resolver_var.set(smart_cb)

        try:
            with StreamPhaseContext("final"):
                response = self.llm.chat(
                    query=prompted_user_query,
                    chat_history=chat_history,
                    model=model,
                    system_prompt=system_prompt,
                )
        finally:
            if resolver_ctx is not None:
                citation_resolver_var.reset(resolver_ctx)
                # Flush happens via finalize() below, after the NO ANSWER check.

        is_no_answer = (response.content == "NO ANSWER FOUND")

        # If real streaming content was already sent to the user, don't treat
        # this as NO ANSWER even if the fallback returned the sentinel.  This
        # covers the LLMs.py exception-fallback path where streaming succeeded
        # partially, then chat_engine.chat() returned "NO ANSWER FOUND".
        if is_no_answer and resolver is not None and resolver.emitted_text:
            is_no_answer = False

        if is_no_answer:
            if not previous_bot_response_was_no_answer:
                response.content = ANSWER_NOT_FOUND_FIRST_TIME
            else:
                if is_moodle and course_id is not None:
                    response.content = ANSWER_NOT_FOUND_SECOND_TIME_MOODLE.format(course_id=course_id)
                elif is_moodle:
                    response.content = ANSWER_NOT_FOUND_SECOND_TIME_MOODLE.format(course_id="UNKNOWN")
                else:
                    response.content = ANSWER_NOT_FOUND_SECOND_TIME_DRUPAL

        # Flush streaming buffers (and emit friendly message for NO ANSWER case).
        if smart_cb is not None:
            smart_cb.finalize(
                is_no_answer=is_no_answer,
                friendly_message=response.content if is_no_answer else "",
            )
        elif resolver is not None:
            resolver.flush()

        # Bug 2 fix: use exactly what was streamed as the final response content.
        # This ensures citations_markdown from parse_citations matches the streamed
        # text, preventing the visual jump on the final event.
        if not is_no_answer and resolver is not None and resolver.emitted_text:
            response.content = resolver.emitted_text

        if response is None:
            raise ValueError(f"LLM produced no response. Please check the LLM implementation. Response: {response}")

        return response
