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
from src.llm.tools.language import detect_language_socratic_v2
from src.llm.tools.retrieve import retrieve_chunks
from src.llm.tools.rerank import rerank_chunks
from src.llm.tools.socratic_v2_opening import socratic_v2_opening
from src.llm.tools.socratic_v2_core import socratic_v2_core
from src.llm.tools.socratic_v2_consolidation import socratic_v2_consolidation
from src.llm.tools.socratic_v2_translate import socratic_v2_translate


def build_socratic_v2_graph() -> StateGraph:
    """
    Builds the socratic v2 subgraph.

    Multi-Turn Flow (phase persisted via checkpointer):
    Request 1: phase=None/"opening" → opening (content + objectives + welcome) → END
    Request 2..n: phase="core"      ─┬→ retrieve → rerank ─┐
                                     └→ detect_language ───┴→ core (policy loop) → END
    Final:     phase="consolidation" → detect_language → consolidation (feedback + close, reset) → END
    Any phase, translation request  → translate (previous tutor message 1:1) → END

    Language detection is the same as in the normal chat path (every turn,
    same detector, same history window — see detect_language_socratic_v2) and
    runs in parallel with retrieval, so it adds no latency to core turns.
    """
    graph = StateGraph(GraphState)

    # Retrieval path grounds the core loop in course material (module scope,
    # incl. previous modules when module_id is a list).
    graph.add_node("retrieve", retrieve_chunks)
    graph.add_node("rerank", rerank_chunks)
    # Two instances of the same detection node — one per phase, because their
    # successors differ (join into core vs. straight into consolidation).
    graph.add_node("detect_language_core", detect_language_socratic_v2)
    graph.add_node("detect_language_consolidation", detect_language_socratic_v2)

    graph.add_node("socratic_v2_opening_node", socratic_v2_opening)
    graph.add_node("socratic_v2_core_node", socratic_v2_core)
    graph.add_node("socratic_v2_consolidation_node", socratic_v2_consolidation)
    graph.add_node("socratic_v2_translate_node", socratic_v2_translate)

    def route_socratic_v2_phase(state: GraphState) -> str | list[str]:
        """Route based on the persisted session phase.

        A translation request (flagged per turn by contextualize_and_route)
        takes precedence in any phase — it re-renders the previous tutor
        message and leaves the phase untouched.
        """
        if state.get("v2_translate_last"):
            return "socratic_v2_translate_node"
        phase = state.get("socratic_v2_phase")
        if phase == "core":
            # Fan out: retrieval and language detection run concurrently.
            return ["retrieve", "detect_language_core"]
        if phase == "consolidation":
            return "detect_language_consolidation"
        # None or "opening": first entry into the v2 workflow (the opening
        # node resolves the session language itself)
        return "socratic_v2_opening_node"

    graph.add_conditional_edges(
        START,
        route_socratic_v2_phase,
        [
            "socratic_v2_opening_node",
            "retrieve",
            "detect_language_core",
            "detect_language_consolidation",
            "socratic_v2_translate_node",
        ],
    )

    graph.add_edge("retrieve", "rerank")
    # Fan in as a real join: core runs exactly once, after BOTH branches. (Two
    # separate edges would fire core as soon as the faster branch finishes.)
    graph.add_edge(["rerank", "detect_language_core"], "socratic_v2_core_node")
    graph.add_edge("detect_language_consolidation", "socratic_v2_consolidation_node")

    graph.add_edge("socratic_v2_opening_node", END)
    graph.add_edge("socratic_v2_core_node", END)
    graph.add_edge("socratic_v2_consolidation_node", END)
    graph.add_edge("socratic_v2_translate_node", END)

    return graph.compile()
