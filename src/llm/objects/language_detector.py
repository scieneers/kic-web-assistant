import logging

from langfuse.decorators import observe

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.llm.objects.LLMs import LLM, Models
from src.llm.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE = "German"
# Guard against the model leaking a sentence into the answer's system prompt.
# A language name is a single short word; anything longer is treated as unusable.
MAX_LANGUAGE_NAME_LEN = 30


class LanguageDetector:
    """Detects the answer language for a query using a small LLM.

    Replaces the previous statistical (lingua) detector. The LLM understands
    short/mixed/technical queries and, unlike a statistical detector, honors
    explicit requests such as "Bitte antworte auf Englisch".
    """

    def __init__(self):
        self.llm = LLM()
        self.prompt = load_prompt("language_detector_prompt")

    @observe()
    def detect(self, query: str, chat_history: list[SerializableChatMessage] | None = None) -> str:
        """Detect the language the assistant should answer in.

        Args:
            query: The current user query.
            chat_history: Previous messages, used to disambiguate short queries.

        Returns:
            Language name in English (e.g. 'German', 'English'). Falls back to
            'German' if the LLM call fails or returns something unusable.
        """
        # A single-word message (e.g. "hi", "ok", "thanks") carries too little
        # signal to detect reliably. Skip the LLM and fall back immediately.
        if len(query.strip().split()) <= 1 and not chat_history:
            return DEFAULT_LANGUAGE

        try:
            # The mini model is selected internally and is independent of the
            # conversation model, so detection latency never depends on GWDG.
            response = self.llm.chat(
                query=query,
                chat_history=chat_history or [],
                model=Models.MINI,
                system_prompt=self.prompt,
            )
            language = (response.content or "").strip()

            if not language or len(language) > MAX_LANGUAGE_NAME_LEN:
                logger.warning("Unusable language detection result %r, defaulting to %s", language, DEFAULT_LANGUAGE)
                return DEFAULT_LANGUAGE

            return language
        except Exception as exc:
            logger.warning("Language detection failed, defaulting to %s", DEFAULT_LANGUAGE, exc_info=exc)
            return DEFAULT_LANGUAGE
