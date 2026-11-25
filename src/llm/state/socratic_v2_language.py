"""Session language + localized fixed texts for the Socratic v2 tutor.

Testing feedback (2026-11): the tutor started and continued in German even
when the conversation before was English, because (a) all fixed texts
(welcome, quiz frame, exit) were hard-coded German, (b) the generation prompt
only said "language of the learner (default: German)" while the whole context
— prompt, welcome, objectives — was German, and (c) nothing ever detected the
language inside a session.

Now the session language is an explicit state field (v2_language):
- resolved once at opening from the conversation so far, else from the
  frontend's ui_language hint, else German (resolve_opening_language)
- re-detected on every turn with exactly the normal chat path's logic
  (tools/language.py: detect_language_socratic_v2), so answering in English
  switches the tutor to English
- injected as "SPRACHE: <language>" into every v2 generation query
- used to localize the deterministic texts via localize()

A message that asks to translate / switch language ("auf Englisch bitte",
"translate that") is recognized by a cheap regex prefilter plus a MINI-model
confirmation (classify_language_request) and handled by the translate node,
which translates the previous tutor message 1:1 instead of paraphrasing it.
"""

import logging
import re
from typing import List, Optional

from llama_index.core.llms import MessageRole

from src.api.models.serializable_chat_message import SerializableChatMessage

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE = "German"

# Language requests are short; a long learner answer that merely mentions a
# language is never treated as one.
MAX_WORDS_LANGUAGE_REQUEST = 15

_LANGUAGE_ALIASES = {
    "German": ("german", "deutsch", "de"),
    "English": ("english", "englisch", "en"),
    "French": ("french", "französisch", "franzoesisch", "fr"),
    "Spanish": ("spanish", "spanisch", "es"),
    "Italian": ("italian", "italienisch", "it"),
    "Turkish": ("turkish", "türkisch", "tuerkisch", "tr"),
    "Ukrainian": ("ukrainian", "ukrainisch", "uk"),
    "Polish": ("polish", "polnisch", "pl"),
    "Russian": ("russian", "russisch", "ru"),
    "Arabic": ("arabic", "arabisch", "ar"),
    "Dutch": ("dutch", "niederländisch", "niederlaendisch", "nl"),
    "Portuguese": ("portuguese", "portugiesisch", "pt"),
}
_ALIAS_TO_LANGUAGE = {alias: name for name, aliases in _LANGUAGE_ALIASES.items() for alias in aliases}


def normalize_language(value: Optional[str]) -> Optional[str]:
    """Map a language hint to the English language name used in prompts.

    Accepts names in English/German ("English", "Englisch") and ISO / BCP-47
    codes ("en", "en-US", "de_DE"). Unknown values return None.
    """
    if not value or not isinstance(value, str):
        return None
    key = value.strip().lower().replace("_", "-")
    if key in _ALIAS_TO_LANGUAGE:
        return _ALIAS_TO_LANGUAGE[key]
    base = key.split("-")[0]
    return _ALIAS_TO_LANGUAGE.get(base)


def canonical_language(value: Optional[str]) -> str:
    """Language name for state/prompts: known aliases normalized, other
    detector output (e.g. "Swedish") kept, empty → German."""
    if not value or not value.strip():
        return DEFAULT_LANGUAGE
    return normalize_language(value) or value.strip()


def is_german(language: Optional[str]) -> bool:
    return canonical_language(language) == DEFAULT_LANGUAGE


def is_english(language: Optional[str]) -> bool:
    return canonical_language(language) == "English"


# ---------------------------------------------------------------------------
# Language detection (session start + every turn)
# ---------------------------------------------------------------------------


def _detector():
    # Lazy import: shares the singleton with the RAG path's detect_language node.
    from src.llm.tools.language import get_language_detector

    return get_language_detector()


def _user_messages(chat_history: List[SerializableChatMessage]) -> List[SerializableChatMessage]:
    return [m for m in chat_history or [] if m.role == MessageRole.USER and (m.content or "").strip()]


def resolve_opening_language(
    user_query: str,
    chat_history: List[SerializableChatMessage],
    ui_language: Optional[str] = None,
    started_via_flag: bool = False,
) -> str:
    """Language for a new tutor session.

    The start request's own text is often a frontend-generated signal (Moodle
    sends a German "Starte den Lernmodus" with the start flag) — it must not
    decide the language. Priority:
    1. the conversation so far (the learner's own messages)
    2. the frontend's ui_language hint (device/UI language)
    3. German
    """
    from src.env import env

    # Same history window as the normal chat path's detection.
    history = (chat_history or [])[-env.CHAT_HISTORY_LIMIT:]
    user_messages = _user_messages(history)
    hint = normalize_language(ui_language)
    if not user_messages:
        return hint or DEFAULT_LANGUAGE

    # Same detector as the normal chat path. Only difference: a flag-started
    # session's query text is the frontend's signal, not the learner's words,
    # so the learner's last own message stands in for it.
    query = user_messages[-1].content if started_via_flag else user_query
    detected = _detector().detect(query=query, chat_history=history)
    return canonical_language(detected)


# ---------------------------------------------------------------------------
# Translation / language-switch requests
# ---------------------------------------------------------------------------

_LANGUAGE_WORD = (
    r"(?:englisch|english|deutsch|german|französisch|franzoesisch|french|spanisch|spanish|"
    r"italienisch|italian|türkisch|tuerkisch|turkish|ukrainisch|ukrainian|polnisch|polish|"
    r"russisch|russian|arabisch|arabic|niederländisch|niederlaendisch|dutch|portugiesisch|portuguese)\w*"
)
_LANGUAGE_WORD_RE = re.compile(rf"\b{_LANGUAGE_WORD}", re.IGNORECASE)
# Imperative/infinitive/noun forms only — "übersetzt"/"translates" are usually
# subject matter in a learner answer, not a request.
_TRANSLATE_VERB_RE = re.compile(
    r"\b(?:übersetze|übersetz|übersetzen|übersetzung|uebersetze|uebersetzen|translate|translation|"
    r"traduce|traduis|traduire|traducir|traduzir)\b",
    re.IGNORECASE,
)
_LANGUAGE_PHRASE_RE = re.compile(rf"\b(?:auf|in|into|ins|to|en)\s+(?:the\s+)?{_LANGUAGE_WORD}", re.IGNORECASE)


def looks_like_language_request(text: str) -> bool:
    """Cheap prefilter: short message naming a language with a translate verb
    or a "auf/in/into <language>" phrase. Confirmed by classify_language_request."""
    words = (text or "").split()
    if not words or len(words) > MAX_WORDS_LANGUAGE_REQUEST:
        return False
    if not _LANGUAGE_WORD_RE.search(text):
        return False
    return bool(_TRANSLATE_VERB_RE.search(text) or _LANGUAGE_PHRASE_RE.search(text))


def last_assistant_message(chat_history: List[SerializableChatMessage]) -> Optional[str]:
    for message in reversed(chat_history or []):
        if message.role == MessageRole.ASSISTANT and (message.content or "").strip():
            return message.content
    return None


def classify_language_request(user_query: str, chat_history: List[SerializableChatMessage]) -> Optional[str]:
    """Target language if the message asks to translate the previous tutor
    message / switch language; None otherwise (or on any failure).

    Only called when looks_like_language_request matched, so the MINI call is
    rare. The previous tutor message is passed as context so an ANSWER that
    merely mentions translation ("Übersetzung ins Englische" as reply to a
    question about machine translation) is not mistaken for a request.
    """
    from src.llm.objects.LLMs import LLM, Models
    from src.llm.prompts.prompt_loader import load_prompt

    previous = last_assistant_message(chat_history) or ""
    # Both messages as explicit fields of ONE query: the decisive question is
    # whether LATEST answers PREVIOUS, which the model judges far more
    # reliably side by side than across chat turns.
    query = f"PREVIOUS: {previous!r}\nLATEST: {user_query!r}"
    try:
        response = LLM().chat(
            query=query,
            chat_history=[],
            model=Models.MINI,
            system_prompt=load_prompt("socratic_v2_language_request"),
        )
    except Exception:
        logger.warning("Language-request classification failed — treating as normal answer", exc_info=True)
        return None
    answer = (response.content or "").strip().strip(".\"'")
    if not answer or answer.upper() == "NONE" or len(answer) > 30:
        return None
    return canonical_language(answer)


# ---------------------------------------------------------------------------
# Localized fixed texts
# ---------------------------------------------------------------------------

TUTOR_NAME = "Sokratischer Tutor"

# German is the source of truth; English is authored by hand (the common
# second language on KI-Campus); any other language is translated from the
# German text at runtime (translate_fallback_text, MINI model).
_TEXTS = {
    "welcome_title": {
        "German": "🎓 Willkommen beim Sokratischen Tutor!{title_part}",
        "English": "🎓 Welcome to the Socratic Tutor!{title_part}",
    },
    "welcome_scope": {
        "German": " Wir arbeiten mit **{scope_title}**.",
        "English": " We're working with **{scope_title}**.",
    },
    "welcome_objectives": {
        "German": "Darum geht es hier:",
        "English": "Here's what this is about:",
    },
    "welcome_question": {
        "German": (
            "Womit möchtest du anfangen — eines dieser Themen oder eine eigene Frage aus dem Modul? "
            "Erzähl mir gern auch kurz, was du dazu schon weißt."
        ),
        "English": (
            "Where would you like to start — one of these topics or a question of your own about the module? "
            "Feel free to tell me briefly what you already know about it."
        ),
    },
    "welcome_exit_hint": {
        "German": "_(Beenden kannst du den Sokratischen Tutor jederzeit über den Schalter oder mit „stopp“.)_",
        "English": "_(You can end the Socratic Tutor at any time with the toggle or by typing “stop”.)_",
    },
    "no_scope": {
        "German": (
            "Um den Sokratischen Tutor zu starten, wähle bitte zuerst einen Kurs oder ein Modul aus — "
            "dann weiß ich, mit welchen Inhalten wir arbeiten."
        ),
        "English": (
            "To start the Socratic Tutor, please select a course or module first — "
            "then I know which content we're working with."
        ),
    },
    "no_content": {
        "German": (
            "Zu diesem Modul habe ich leider keine Inhalte gefunden, mit denen wir im Sokratischen Tutor "
            "arbeiten können. Wähle gern ein anderes Modul aus."
        ),
        "English": (
            "Unfortunately I couldn't find any content in this module that we could work with in the "
            "Socratic Tutor. Feel free to choose another module."
        ),
    },
    "objective_fallback_titled": {
        "German": "Die Inhalte von „{scope_title}“ verstehen",
        "English": "Understand the content of “{scope_title}”",
    },
    "objective_fallback": {
        "German": "Die Inhalte dieses Moduls verstehen",
        "English": "Understand the content of this module",
    },
    "exit": {
        "German": "Du hast den Sokratischen Tutor beendet. Wenn du weitere Fragen hast, stehe ich dir gerne zur Verfügung!",
        "English": "You have ended the Socratic Tutor. If you have any further questions, I'm happy to help!",
    },
    "consolidation_fallback": {
        "German": (
            "Danke für deine Zusammenfassung — damit ist die Lernsession abgeschlossen! "
            "Du kannst jetzt normal weiterfragen oder den Sokratischen Tutor jederzeit erneut starten."
        ),
        "English": (
            "Thanks for your summary — that concludes this learning session! "
            "You can keep asking questions as usual or start the Socratic Tutor again at any time."
        ),
    },
    "generation_fallback": {
        "German": "Kannst du deine Überlegung genauer erklären?",
        "English": "Can you explain your reasoning in a bit more detail?",
    },
    "quiz_header": {
        "German": "📝 Quizfrage aus dem Kurs:",
        "English": "📝 Quiz question from the course:",
    },
    "quiz_instruction_single": {
        "German": "Genau eine Antwort ist richtig — antworte mit dem Buchstaben oder deiner Antwort.",
        "English": "Exactly one answer is correct — reply with the letter or your answer.",
    },
    "quiz_instruction_multi": {
        "German": "Mehrere Antworten können richtig sein — nenne alle passenden Buchstaben (z. B. „A, C“).",
        "English": "More than one answer may be correct — give all matching letters (e.g. “A, C”).",
    },
    "quiz_instruction_truefalse": {
        "German": "Antworte mit dem Buchstaben oder mit „wahr“ bzw. „falsch“.",
        "English": "Reply with the letter or with “true” or “false”.",
    },
    "quiz_partial": {
        "German": "Teilweise richtig – ein Teil deiner Auswahl stimmt.",
        "English": "Partly right — some of your choices are correct.",
    },
    "quiz_partial_wrong": {
        "German": "Mindestens eine der gewählten Antworten passt aber nicht.",
        "English": "But at least one of the options you chose doesn't fit.",
    },
    "quiz_partial_missing": {
        "German": "Es ist noch mindestens eine weitere Antwort richtig.",
        "English": "At least one more option is also correct.",
    },
    "quiz_partial_retry": {
        "German": "Schau dir die Optionen noch einmal an und nenne alle passenden Buchstaben.",
        "English": "Take another look at the options and give all matching letters.",
    },
    "quiz_none_left": {
        "German": "Zu diesem Abschnitt gibt es keine weiteren Quizfragen aus dem Kurs. Ich kann dir aber gern eine eigene Übungsfrage zum Thema stellen – sag einfach Bescheid.",
        "English": "There are no more quiz questions from the course for this section. I'm happy to ask you a practice question of my own on the topic instead — just let me know.",
    },
    "quiz_none_in_scope": {
        "German": "Zu diesem Abschnitt gibt es im Kurs keine Quizfragen. Ich kann dir aber gern eine eigene Übungsfrage zum Thema stellen – sag einfach Bescheid.",
        "English": "The course has no quiz questions for this section. I'm happy to ask you a practice question of my own on the topic instead — just let me know.",
    },
    # Second partial attempt: more concrete, counts only (no option names).
    "quiz_partial_again": {
        "German": "Noch nicht ganz.",
        "English": "Not quite yet.",
    },
    "quiz_count_all_right": {
        "German": "Alles, was du gewählt hast, stimmt.",
        "English": "Everything you chose is correct.",
    },
    # Everything ticked: no counts — they would reveal how many are correct.
    "quiz_all_selected": {
        "German": "Du hast alle Optionen gewählt – aber nicht alle sind richtig. Wähle nur die passenden aus.",
        "English": "You chose every option — but not all of them are correct. Pick only the ones that fit.",
    },
    "quiz_count_right_one": {
        "German": "Eine deiner Antworten stimmt.",
        "English": "One of your choices is correct.",
    },
    "quiz_count_right_many": {
        "German": "{n} deiner Antworten stimmen.",
        "English": "{n} of your choices are correct.",
    },
    "quiz_count_missing_one": {
        "German": "Es fehlt noch genau eine richtige Antwort.",
        "English": "Exactly one correct option is still missing.",
    },
    "quiz_count_missing_many": {
        "German": "Es fehlen noch {n} richtige Antworten.",
        "English": "{n} correct options are still missing.",
    },
    "quiz_count_wrong_one": {
        "German": "Eine der gewählten Antworten passt nicht.",
        "English": "One of the options you chose doesn't fit.",
    },
    "quiz_count_wrong_many": {
        "German": "{n} der gewählten Antworten passen nicht.",
        "English": "{n} of the options you chose don't fit.",
    },
    "quiz_last_try": {
        "German": "Das ist dein letzter Versuch, danach lösen wir gemeinsam auf.",
        "English": "This is your last try — after that we'll go through the solution together.",
    },
    # After a wrong pick the next reply replaces the selection — say so explicitly.
    "quiz_partial_retry_full": {
        "German": "Schau dir die Optionen noch einmal an und nenne deine komplette Antwort neu, also alle passenden Buchstaben (z. B. „A, C“).",
        "English": "Take another look at the options and give your complete answer again, i.e. all matching letters (e.g. “A, C”).",
    },
    "quiz_true": {"German": "Wahr", "English": "True"},
    "quiz_false": {"German": "Falsch", "English": "False"},
}


def localize(key: str, language: Optional[str] = None, **kwargs) -> str:
    """Fixed tutor text in the session language (German if unknown)."""
    variants = _TEXTS[key]
    lang = canonical_language(language)
    if lang in variants:
        return variants[lang].format(**kwargs)
    german = variants[DEFAULT_LANGUAGE].format(**kwargs)
    # Lazy import: question_answerer pulls in the whole RAG answer stack.
    from src.llm.objects.question_answerer import translate_fallback_text

    return translate_fallback_text(german, lang)
