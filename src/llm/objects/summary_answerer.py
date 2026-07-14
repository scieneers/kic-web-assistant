import re
import sys

from langfuse.decorators import observe
from llama_index.core.llms import MessageRole
from llama_index.core.schema import TextNode

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.llm.objects.LLMs import LLM, Models
from src.llm.objects.citation_parser import CITATION_TEXT, _get_display_title, citation_suffix
from src.llm.objects.question_answerer import NO_CONTENT_IN_MODULE, NO_CONTENT_UNSUPPORTED_TYPE, format_sources
from src.llm.prompts.prompt_loader import load_prompt
from src.llm.streaming import CitationStreamResolver, SmartStreamCallback, StreamPhaseContext, citation_resolver_var, token_callback_var

NO_CONTENT_TO_SUMMARIZE = "Zu {scope_name} liegt mir kein Inhalt vor, den ich zusammenfassen könnte."

SUMMARY_SYSTEM_PROMPT = load_prompt("summary_prompt")

USER_TASK_WITH_SOURCES_PROMPT = """<TASK>:
Fasse den folgenden Lerninhalt vollständig zusammen.{focus_line}

{sources}"""


class SummaryAnswerer:
    """Generates a content summary from a full, unranked scope of sources.

    Deliberately separate from QuestionAnswerer: summaries have no "NO ANSWER
    FOUND" concept (a scope that yielded sources always gets a summary) and no
    two-stage no-answer escalation — those concepts only make sense for a
    query that may or may not be answerable from the sources.
    """

    def __init__(self) -> None:
        self.name = "SummaryAnswerer"
        self.llm = LLM()

    @observe()
    def summarize(
        self,
        chat_history: list[SerializableChatMessage],
        sources: list[TextNode],
        model: Models,
        language: str,
        scope_name: str,
        focus_hint: str | None = None,
    ) -> SerializableChatMessage:
        """
        Args:
            chat_history: Prior conversation (e.g. for "no, kürzer bitte"-style refinements)
            sources: Full, ordered scope of chunks to summarize (see retrieve_full_scope)
            model: LLM model to use
            language: Detected response language
            scope_name: Short German phrase describing the scope (e.g. "diesem Modul",
                "diesem Kurs") — only used when `sources` is empty and there is no
                document to pull an actual name from
            focus_hint: Optional contextualized query hint (e.g. "nur den zweiten Teil")
                to weight the summary towards, without dropping other major parts
        """
        if not sources:
            return SerializableChatMessage(
                role=MessageRole.ASSISTANT,
                content=NO_CONTENT_TO_SUMMARIZE.format(scope_name=scope_name),
            )

        # EmptyModule-Marker: gesamter Scope besteht nur aus Modulen ohne Inhalt.
        # Kein LLM-Aufruf — direkt den fest definierten Fallback-Text zurückgeben.
        # Gleiche Unterscheidung wie in answer.py: bekannter, bewusst nicht
        # unterstützter Inhaltstyp benennt den Grund konkret.
        if all(s.metadata.get("type") == "EmptyModule" for s in sources):
            first = sources[0].metadata
            module_name = first.get("fullname", scope_name)
            unsupported_label = first.get("unsupported_label")
            if unsupported_label:
                return SerializableChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=NO_CONTENT_UNSUPPORTED_TYPE.format(
                        module_name=module_name, label=unsupported_label, url=first.get("url", "")
                    ),
                )
            return SerializableChatMessage(
                role=MessageRole.ASSISTANT,
                content=NO_CONTENT_IN_MODULE.format(module_name=module_name),
            )

        system_prompt = SUMMARY_SYSTEM_PROMPT.format(language=language)
        formatted_sources = format_sources(sources, max_length=sys.maxsize)
        focus_line = f"\nFokus: {focus_hint}" if focus_hint else ""
        prompted_user_query = USER_TASK_WITH_SOURCES_PROMPT.format(
            focus_line=focus_line, sources=formatted_sources
        )

        # When streaming, replace [docN] markers with clickable links in real-time —
        # same wiring as QuestionAnswerer.answer_question (see there for details).
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

        # Summaries have no NO-ANSWER sentinel — always finalize as real content.
        if smart_cb is not None:
            smart_cb.finalize(is_no_answer=False, friendly_message="")
        elif resolver is not None:
            resolver.flush()

        # Use exactly what was streamed as the final response content (matches
        # QuestionAnswerer's "Bug 2 fix" — keeps citations_markdown in sync).
        if resolver is not None and resolver.emitted_text:
            response.content = resolver.emitted_text

        if response is None:
            raise ValueError(f"LLM produced no response. Please check the LLM implementation. Response: {response}")

        return response
