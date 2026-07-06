from typing import List, Optional, Literal, Dict, Any, TypedDict

from llama_index.core.schema import TextNode
from src.api.models.serializable_chat_message import SerializableChatMessage
from src.api.models.serializable_text_node import SerializableTextNode

Scenario = Literal["no_vectordb", "simple_hop", "socratic", "exit_complete"]
SocraticMode = Literal["contract", "diagnose", "core"]
RerankerType = Literal["llm", "azure_semantic", "bge"]
# "core" routes to "hinting", "reflection", "explain" internally

class GraphState(TypedDict, total=False):
    # user input and context
    user_query: str
    chat_history: List[SerializableChatMessage]

    # classified scenario
    mode: Optional[Scenario]
    
    # runtime configuration (from API/Frontend)
    runtime_config: Dict[str, Any]  # model, course_id, module_id, thread_id
    
    # system configuration (hardcoded defaults, not exposed to frontend)
    system_config: Dict[str, Any]  # rerank_top_n and retrieval_k

    # shared intermediate artifacts
    contextualized_query: Optional[str]
    detected_language: Optional[str]

    # retrieval artifacts
    retrieved: List[SerializableTextNode]
    reranked: List[SerializableTextNode]

    # socratic specific artifacts
    socratic_mode: Optional[SocraticMode]  # Internal routing: "contract" | "diagnose" | "core" ("hinting", "reflection" and "explain" handled in core)
    #diagnostic
    learning_objective: Optional[str]  # Identified learning goal for the interaction
    #core
    attempt_count: int  # Total number of attempts student made at current question/concept
    number_given_hints: int  # Total number of hints given so far (for hint graduation)

    # output
    answer: Optional[str]
    citations_markdown: Optional[str]


def get_doc_as_textnodes(state: GraphState, node: str) -> List[TextNode]:
    """Helper function to convert SerializableTextNode to TextNode for component usage."""
    return [node.to_text_node() for node in state.get(node, [])]