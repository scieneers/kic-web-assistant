from langgraph.graph import StateGraph, START, END

from src.llm.state.models import GraphState
from src.llm.objects.LLMs import LLM
from src.llm.streaming import StreamPhaseContext
from src.llm.tools.language import detect_language
from src.llm.prompts.prompt_loader import load_prompt

llm = LLM()
NO_VECTORDB_PROMPT = load_prompt("no_vector_db_prompt")
# Facts for "what can you do?" / "what is the Socratic Tutor?" — the tutor is
# only advertised when it is actually enabled in this deployment.
ASSISTANT_MODES_WITH_TUTOR = load_prompt("assistant_modes_with_tutor")
ASSISTANT_MODES_CHAT_ONLY = load_prompt("assistant_modes_chat_only")


def build_no_vectordb_prompt(language: str, system_config: dict | None) -> str:
    tutor_enabled = bool((system_config or {}).get("enable_socratic_v2"))
    modes = ASSISTANT_MODES_WITH_TUTOR if tutor_enabled else ASSISTANT_MODES_CHAT_ONLY
    return NO_VECTORDB_PROMPT.replace("{assistant_modes}", modes.strip()).replace("{language}", language)

def direct_answer_node(state: GraphState) -> dict:
    """
    Generate direct conversational response without retrieval.
    
    For no_vectordb scenario: Simple, friendly responses to conversational queries.
    """
    # Get necessary variables from state
    model = state["runtime_config"]["model"]
    query = state["user_query"]
    chat_history = state["chat_history"]
    language = state["detected_language"]

    # Insert language + assistant facts (mode list depends on enabled features)
    language_enriched_prompt = build_no_vectordb_prompt(language, state.get("system_config"))
    
    # Simple conversational response (no sources)
    with StreamPhaseContext("final"):
        response = llm.chat(
            query=query,
            chat_history=chat_history,
            system_prompt=language_enriched_prompt,
            model=model,
        )
    
    return {
        "answer": response.content,
        "citations_markdown": None  # No citations in no_vectordb mode
    }


def build_no_vectordb_graph() -> StateGraph:
    """
    Builds the no_vectordb subgraph for conversational queries.
    
    Flow:
    START → detect_language → direct_answer → END
    """
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("detect_language_node", detect_language)
    graph.add_node("direct_answer_node", direct_answer_node)
    
    # Define linear flow
    graph.add_edge(START, "detect_language_node")
    graph.add_edge("detect_language_node", "direct_answer_node")
    graph.add_edge("direct_answer_node", END)
    
    return graph.compile()
