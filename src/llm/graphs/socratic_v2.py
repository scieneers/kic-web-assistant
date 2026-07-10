"""
Socratic v2 Subgraph — "Lernmodus v2", the redesigned guided-learning tutor.

Concept: documentation/KONZEPT_SOKRATISCHER_LERNASSISTENT.md. Coexists with the
v1 subgraph (graphs/socratic.py) so both variants can be compared side by side
in the Streamlit frontend; all state fields are v2-prefixed and independent.

Session dramaturgy (3 phases, tracked in socratic_v2_phase):
- opening: load the module content (retrieve_all), extract learning
  objectives + key concepts once, welcome the learner and negotiate the goal
- core: move-policy loop — a policy LLM picks the next move (FRAGE, HINT,
  MICRO_EXPLAIN, ERKLAER_ZURUECK, FEHLER_FINDEN, ZWISCHENFAZIT, ENCOURAGE,
  CONSOLIDATE, EXIT) and updates the learner model; deterministic guards
  enforce the hint ladder; a generation LLM produces the grounded message
- consolidation: the learner summarizes, the tutor gives feedback, shows the
  learning status and closes the session (full v2 state reset)

Multi-Turn Design (same as v1):
- Each request executes EXACTLY ONE path, then returns to the user
- The next node is chosen from socratic_v2_phase persisted in the checkpoint
"""

from langgraph.graph import StateGraph, START, END

from src.llm.state.models import GraphState
from src.llm.tools.retrieve import retrieve_chunks
from src.llm.tools.rerank import rerank_chunks
from src.llm.tools.socratic_v2_opening import socratic_v2_opening
from src.llm.tools.socratic_v2_core import socratic_v2_core
from src.llm.tools.socratic_v2_consolidation import socratic_v2_consolidation


def build_socratic_v2_graph() -> StateGraph:
    """
    Builds the socratic v2 subgraph.

    Multi-Turn Flow (phase persisted via checkpointer):
    Request 1: phase=None/"opening" → opening (content + objectives + welcome) → END
    Request 2..n: phase="core"      → retrieve → rerank → core (policy loop) → END
    Final:     phase="consolidation" → consolidation (feedback + close, reset) → END
    """
    graph = StateGraph(GraphState)

    # Retrieval path grounds the core loop in course material (module scope,
    # incl. previous modules when module_id is a list).
    graph.add_node("retrieve", retrieve_chunks)
    graph.add_node("rerank", rerank_chunks)

    graph.add_node("socratic_v2_opening_node", socratic_v2_opening)
    graph.add_node("socratic_v2_core_node", socratic_v2_core)
    graph.add_node("socratic_v2_consolidation_node", socratic_v2_consolidation)

    def route_socratic_v2_phase(state: GraphState) -> str:
        """Route to exactly one node based on the persisted session phase."""
        phase = state.get("socratic_v2_phase")
        if phase == "core":
            return "retrieve"
        if phase == "consolidation":
            return "socratic_v2_consolidation_node"
        # None or "opening": first entry into the v2 workflow
        return "socratic_v2_opening_node"

    graph.add_conditional_edges(
        START,
        route_socratic_v2_phase,
        {
            "socratic_v2_opening_node": "socratic_v2_opening_node",
            "retrieve": "retrieve",
            "socratic_v2_consolidation_node": "socratic_v2_consolidation_node",
        },
    )

    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "socratic_v2_core_node")

    graph.add_edge("socratic_v2_opening_node", END)
    graph.add_edge("socratic_v2_core_node", END)
    graph.add_edge("socratic_v2_consolidation_node", END)

    return graph.compile()
