from typing import List, Optional, Literal, Dict, Any, TypedDict

from llama_index.core.schema import TextNode
from src.api.models.serializable_chat_message import SerializableChatMessage
from src.api.models.serializable_text_node import SerializableTextNode

Scenario = Literal["no_vectordb", "simple_hop", "socratic", "socratic_v2", "summarize", "exit_complete"]
SocraticMode = Literal["contract", "diagnose", "core"]
# Socratic v2 ("Lernmodus v2") phases: opening (load content, extract
# objectives, welcome) → core (move-policy loop) → consolidation (learner
# summarizes, session closes). See src/llm/graphs/socratic_v2.py.
SocraticV2Phase = Literal["opening", "core", "consolidation"]
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
    # True when retrieval already ranked the results with Azure's semantic
    # ranker (integrated mode) — the rerank node then only cuts/thresholds.
    retrieval_semantic_ranked: bool
    reranked: List[SerializableTextNode]

    # socratic specific artifacts
    socratic_mode: Optional[SocraticMode]  # Internal routing: "contract" | "diagnose" | "core" ("hinting", "reflection" and "explain" handled in core)
    #diagnostic
    learning_objective: Optional[str]  # Identified learning goal for the interaction
    #core
    attempt_count: int  # Total number of attempts student made at current question/concept
    number_given_hints: int  # Total number of hints given so far (for hint graduation)

    # socratic v2 ("Lernmodus v2") artifacts — session-scoped, live only in the
    # in-memory checkpoint. Independent of the v1 fields above so both variants
    # can be compared side by side without touching each other.
    socratic_v2_phase: Optional[SocraticV2Phase]
    v2_learning_objectives: Optional[List[str]]  # extracted from module content at opening
    v2_key_concepts: Optional[List[str]]  # key concepts of the module, extracted at opening
    v2_session_goal: Optional[str]  # goal negotiated with the learner during the session
    v2_learner_model: Optional[Dict[str, Any]]  # {"konzepte": {name: status}, "missverstaendnisse": [...], "affekt": str}
    v2_target_concept: Optional[str]  # concept the current core exchange focuses on
    v2_hint_count: int  # hints given for the CURRENT target concept (2 → forced micro-explain)
    v2_question_streak: int  # consecutive question moves without giving anything back (3 → forced hint)
    v2_scope_title: Optional[str]  # module/course display name for tutor messages

    # output
    answer: Optional[str]
    citations_markdown: Optional[str]


def get_doc_as_textnodes(state: GraphState, node: str) -> List[TextNode]:
    """Helper function to convert SerializableTextNode to TextNode for component usage."""
    return [node.to_text_node() for node in state.get(node, [])]