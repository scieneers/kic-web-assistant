import logging
import re

from langfuse.decorators import observe
from llama_index.core.llms import MessageRole
from llama_index.core.schema import TextNode

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.llm.objects.LLMs import LLM, Models
from src.llm.objects.citation_parser import CITATION_TEXT, _get_display_title, citation_suffix
from src.llm.prompts.prompt_loader import load_prompt
from src.llm.streaming import CitationStreamResolver, SmartStreamCallback, StreamPhaseContext, citation_resolver_var, token_callback_var

logger = logging.getLogger(__name__)

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

# Modul-Scope: statt einer Sackgasse die Eskalation aufs Kurslevel anbieten.
# Ein bejahender Folge-Turn wird deterministisch erkannt (contextualize.py,
# ESCALATION_ACCEPT_RESPONSES) und wiederholt die Frage ohne Modul-Filter.
NO_ANSWER_IN_MODULE_OFFER_COURSE = """Dazu habe ich in diesem Modul leider nichts gefunden. Soll ich stattdessen im gesamten Kurs suchen? Antworte einfach mit „Ja“.
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
    if text == NO_ANSWER_IN_MODULE_OFFER_COURSE.strip():
        return "no_answer_module_offer_course"
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


# The set of fallback types that count as "no answer" for the cross-turn
# escalation check in answer_question() below (previous_bot_response_was_no_answer).
_NO_ANSWER_FALLBACK_TYPES = {
    get_fallback_type(ANSWER_NOT_FOUND_FIRST_TIME),
    get_fallback_type(NO_RELEVANT_CONTENT),
    get_fallback_type(NO_ANSWER_IN_MODULE_OFFER_COURSE),
}

DEFAULT_FALLBACK_LANGUAGE = "German"
FALLBACK_TRANSLATOR_PROMPT = load_prompt("fallback_translator_prompt")


def translate_fallback_text(text: str, language: str | None) -> str:
    """Translates a canned German fallback message into the detected answer language.

    The fallback constants in this module are only ever authored in German.
    Skips the LLM call when the target language already is German (the
    common case) to avoid unnecessary latency/cost. Falls back to the
    original German text if translation fails for any reason.
    """
    if not language or language.strip().lower() == DEFAULT_FALLBACK_LANGUAGE.lower():
        return text
    try:
        translated = LLM().chat(
            query=text,
            chat_history=[],
            model=Models.MINI,
            system_prompt=FALLBACK_TRANSLATOR_PROMPT.format(language=language),
        )
        return translated.content.strip() or text
    except Exception:
        logger.warning("Fallback translation to %r failed, keeping German text", language, exc_info=True)
        return text


SYSTEM_PROMPT = load_prompt("system_prompt")

USER_QUERY_WITH_SOURCES_PROMPT = """
[doc{index}]
Content: {content}
Metadata: {metadata}
"""


# Character budget for the <SOURCES> block sent to the LLM. Applies to every
# model, including the Azure fallback: it's a fallback path, not a separate
# tier meant to carry more content, and an unbounded (sys.maxsize) source dump
# for a large course caused a real 400 BadRequestError in production (261745
# input tokens on a 262144-token model, 2026-07-23).
SOURCES_MAX_LENGTH = 8_000


def format_sources(sources: list[TextNode], max_length: int = SOURCES_MAX_LENGTH) -> str:
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


# A chunk only earns its place in the prompt if it can carry at least this
# much real content — otherwise the [docN]/Content:/Metadata: wrapper
# overhead alone can exceed a chunk's whole budget share, and cramming in
# every chunk anyway just produces near-empty entries that still add up to
# far more total length than max_length allows for (see
# format_sources_full_scope's docstring).
MIN_CONTENT_PER_CHUNK = 100


def format_sources_full_scope(sources: list[TextNode], max_length: int = SOURCES_MAX_LENGTH) -> str:
    """Formats an unranked, full scope of sources (e.g. a whole course or several
    selected modules) for summarization.

    Unlike format_sources() — built for a short, relevance-ranked citation list,
    where stopping once the budget is used up means only dropping the least
    relevant sources — a full scope has no relevance order: sources are simply
    grouped by module. Applying the same stop-once-full logic there means
    whichever module(s) happen to come first in the list eat the entire budget
    and every module after them is silently dropped instead of trimmed.

    Splitting the budget evenly per *chunk* instead doesn't work either: a
    whole-course scope can have thousands of chunks (e.g. 2207 for one real
    course), which would give each chunk only a handful of characters — worse,
    the fixed per-entry overhead ([docN]/Content:/Metadata: {...}) alone can
    exceed that, so including every chunk regardless would blow way past
    max_length in total even though each individual chunk looks "capped".

    Instead: split max_length evenly per *module* first. Within a module,
    only include as many of its chunks as its own share can afford at
    MIN_CONTENT_PER_CHUNK real characters each — the rest of that module's
    chunks are dropped, not shrunk to nothing. A module's total budget no
    longer depends on how many chunks it happens to have, and how much of it
    gets seen shrinks gracefully (matching COURSE_LEVEL_SCOPE_GUIDANCE's
    "1-2 sentences per module" for large courses) instead of every chunk
    surviving with unusable, budget-blowing scraps of text.

    [docN] markers must keep referring to a source's original position in
    `sources` (SummaryAnswerer's citation resolver looks sources up by that
    index) — so chunks are grouped/selected only to decide inclusion and
    budget, then written out in original order with the original index.
    """
    if not sources:
        return "<SOURCES>:\n"

    indices_per_module: dict[object, list[int]] = {}
    for i, source in enumerate(sources):
        indices_per_module.setdefault(source.metadata.get("module_id"), []).append(i)

    per_module_budget = max(max_length // len(indices_per_module), 1)

    included_indices: set[int] = set()
    content_budget_by_index: dict[int, int] = {}
    for indices in indices_per_module.values():
        overhead = len(
            USER_QUERY_WITH_SOURCES_PROMPT.format(index=indices[0] + 1, content="", metadata=sources[indices[0]].metadata)
        )
        max_affordable_chunks = max(per_module_budget // (overhead + MIN_CONTENT_PER_CHUNK), 1)
        selected = indices[:max_affordable_chunks]
        per_chunk_budget = per_module_budget // len(selected)
        for idx in selected:
            included_indices.add(idx)
            content_budget_by_index[idx] = max(per_chunk_budget - overhead, 0)

    sources_text = ""
    for i, source in enumerate(sources):
        if i not in included_indices:
            continue
        content_budget = content_budget_by_index[i]
        content = source.get_text() if hasattr(source, 'get_text') else source.text
        if len(content) > content_budget:
            content = content[:content_budget] + "…"
        source_entry = USER_QUERY_WITH_SOURCES_PROMPT.format(
            index=i + 1, content=content, metadata=source.metadata
        )
        sources_text += source_entry + "\n"

    return "<SOURCES>:\n" + sources_text.strip()


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
        offer_course_escalation: bool = False,
    ) -> SerializableChatMessage:
        """offer_course_escalation: the query was module-scoped and a course-level
        retry is possible — a first-time no-answer then offers the escalation
        (NO_ANSWER_IN_MODULE_OFFER_COURSE) instead of the generic fallback."""

        # Determine whether the previous assistant turn was already a "no answer"
        # before starting the LLM call — needed to pick the right fallback message.
        previous_bot_response_was_no_answer = False
        if chat_history:
            for msg in reversed(chat_history):
                if msg.role == MessageRole.ASSISTANT:
                    if msg.fallback_type is not None:
                        # Translated fallback text no longer matches the German
                        # constants below — the type survives translation.
                        previous_bot_response_was_no_answer = msg.fallback_type in _NO_ANSWER_FALLBACK_TYPES
                    else:
                        previous_bot_response_was_no_answer = (
                            msg.content
                            in (ANSWER_NOT_FOUND_FIRST_TIME, NO_RELEVANT_CONTENT, NO_ANSWER_IN_MODULE_OFFER_COURSE)
                        )
                    break

        # Early exit: no sources retrieved → skip both rerank and LLM answer call.
        if not sources:
            # offer_course_escalation reflects this turn's actual scope state
            # (module+course present, not yet escalated) and must win over
            # previous_bot_response_was_no_answer: that flag only looks at the
            # previous message's text, so a declined offer followed by a new,
            # unrelated question would otherwise be misread as "same question,
            # second failure" just because the offer text is itself a no-answer
            # fallback. A fresh question always gets a fresh offer.
            if offer_course_escalation:
                fallback = NO_ANSWER_IN_MODULE_OFFER_COURSE
            elif not previous_bot_response_was_no_answer:
                fallback = NO_RELEVANT_CONTENT
            elif is_moodle and course_id is not None:
                fallback = ANSWER_NOT_FOUND_SECOND_TIME_MOODLE.format(course_id=course_id)
            elif is_moodle:
                fallback = ANSWER_NOT_FOUND_SECOND_TIME_MOODLE.format(course_id="UNKNOWN")
            else:
                fallback = ANSWER_NOT_FOUND_SECOND_TIME_DRUPAL
            return SerializableChatMessage(
                role=MessageRole.ASSISTANT,
                content=translate_fallback_text(fallback, language),
                fallback_type=get_fallback_type(fallback),
            )

        system_prompt = SYSTEM_PROMPT.format(language=language)
        formatted_sources = format_sources(sources, max_length=SOURCES_MAX_LENGTH)
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
            # See the analogous no-sources branch above for why
            # offer_course_escalation takes priority.
            if offer_course_escalation:
                fallback_text = NO_ANSWER_IN_MODULE_OFFER_COURSE
            elif not previous_bot_response_was_no_answer:
                fallback_text = ANSWER_NOT_FOUND_FIRST_TIME
            else:
                if is_moodle and course_id is not None:
                    fallback_text = ANSWER_NOT_FOUND_SECOND_TIME_MOODLE.format(course_id=course_id)
                elif is_moodle:
                    fallback_text = ANSWER_NOT_FOUND_SECOND_TIME_MOODLE.format(course_id="UNKNOWN")
                else:
                    fallback_text = ANSWER_NOT_FOUND_SECOND_TIME_DRUPAL
            response.fallback_type = get_fallback_type(fallback_text)
            response.content = translate_fallback_text(fallback_text, language)

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
