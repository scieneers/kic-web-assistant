from langfuse.decorators import observe

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.llm.objects.LLMs import LLM, Models
from src.llm.state.models import Scenario
from src.llm.prompts.prompt_loader import load_prompt

class Contextualizer:
    """Contextualizes a message based on the chat history, so that it can effectively used as input for RAG retrieval."""

    def __init__(self):
        # Initialize LLM instance
        self.llm = LLM()
        # Load prompts once at initialization
        self.CONDENSE_QUESTION_PROMPT = load_prompt("contextualizer_prompt")
        self.CONDENSE_SOCRATIC_PROMPT = load_prompt("contextualizer_socratic_prompt")
        self.ROUTER_PROMPT = load_prompt("router_prompt")

    @observe()
    def contextualize(self, query: str, chat_history: list[SerializableChatMessage], model: Models) -> str:
        """Contextualize a message based on the chat history, so that it can effectively used as input for RAG retrieval."""

        contextualized_question = self.llm.chat(
            query=query, chat_history=chat_history, model=model, system_prompt=self.CONDENSE_QUESTION_PROMPT
        )
        if contextualized_question.content is None:
            raise ValueError(
                f"Contextualized question is None. Please check the LLM implementation. Response: {contextualized_question}"
            )

        return contextualized_question.content

    @observe()
    def contextualize_socratic(self, query: str, chat_history: list[SerializableChatMessage], model: Models, learning_objective: str) -> str:
        """
        Contextualize a message for socratic core mode retrieval.
        
        Uses contextualizer_socratic_prompt.txt to create a retrieval query
        that considers the learning objective and socratic dialogue context.
        
        Args:
            query: Current user query/response
            chat_history: Previous conversation messages
            model: LLM model to use
            learning_objective: The identified learning goal
            
        Returns:
            Contextualized query optimized for socratic retrieval
        """
        # Prepend learning objective to query if available
        if learning_objective:
            query_with_objective = f"Learning Goal: {learning_objective}\nCurrent Response: {query}"
        else:
            query_with_objective = query
        
        contextualized_question = self.llm.chat(
            query=query_with_objective,
            chat_history=chat_history,
            model=model,
            system_prompt=self.CONDENSE_SOCRATIC_PROMPT
        )
        
        if contextualized_question.content is None:
            raise ValueError(
                f"Contextualized socratic question is None. Please check the LLM implementation. Response: {contextualized_question}"
            )
        
        return contextualized_question.content

    @observe()
    def classify_scenario(self, query: str, model: Models, has_prior_history: bool = False) -> Scenario:
        """
        LLM-based classification of user query into scenario.

        Chat history itself is not passed to the classifier (classification is based
        on the current query) — but whether a prior conversation exists at all is
        needed to disambiguate bare "fasse zusammen" style requests between
        "summarize" (the material) and "no_vectordb" (the conversation), see
        router_prompt.txt's "DISAMBIGUATING BARE SUMMARIZE REQUESTS" section.

        Args:
            query: User's current query
            model: LLM model to use for classification
            has_prior_history: Whether this thread already has prior chat history

        Returns:
            Scenario classification
        """
        history_context = (
            "Session context: a prior conversation already exists in this session."
            if has_prior_history
            else "Session context: this is the first message in this session (no prior conversation)."
        )
        router_prompt = self.ROUTER_PROMPT.format(history_context=history_context)

        # Call LLM to classify scenario (no chat_history needed)
        mode = self.llm.chat(
            query=query, chat_history=[], model=model, system_prompt=router_prompt
        )

        if mode.content is None:
            raise ValueError(
                f"Contextualized socratic question is None. Please check the LLM implementation. Response: {mode}"
            )

        return mode.content