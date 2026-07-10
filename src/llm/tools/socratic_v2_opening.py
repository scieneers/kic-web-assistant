"""Socratic v2 Opening Node — loads module content and starts the session.

First step of the v2 workflow ("Lernmodus v2"):
1. Fetch ALL chunks of the current module from the search index (same
   mechanism as the summarize scenario, see retriever.retrieve_all) — this is
   how the tutor "gets the content" without any ingestion changes.
2. One LLM call extracts learning objectives + key concepts from that content.
3. Emit a deterministic welcome message that names the objectives and asks the
   learner what to work on and what they already know.

Multi-module scope: module_id may be a list (course progress). The LAST entry
is treated as the current/target module whose objectives drive the session;
the earlier modules stay in the retrieval scope during the core loop, so
questions may build on their content as expected prior knowledge.
"""

import logging

from langfuse.decorators import langfuse_context, observe

from src.llm.objects.LLMs import LLM, Models
from src.llm.prompts.prompt_loader import load_prompt
from src.llm.state.models import GraphState
from src.llm.state.socratic_v2_routing import initial_learner_model, reset_socratic_v2_state
from src.llm.tools.retrieve import get_retriever

logger = logging.getLogger(__name__)

# Load prompt once at module level
SOCRATIC_V2_OPENING_PROMPT = load_prompt("socratic_v2_opening")

# Initialize LLM instance at module level
_llm = LLM()

# Cap the extraction input — whole modules can be large, and the extraction
# only needs enough material to name objectives/concepts, not every chunk.
MAX_EXTRACTION_CHARS = 60_000

NO_SCOPE_MESSAGE = (
    "Um den Lernmodus zu starten, wähle bitte zuerst einen Kurs oder ein Modul aus — "
    "dann weiß ich, mit welchen Inhalten wir arbeiten."
)
NO_CONTENT_MESSAGE = (
    "Zu diesem Modul habe ich leider keine Inhalte gefunden, mit denen wir im Lernmodus "
    "arbeiten können. Wähle gern ein anderes Modul aus."
)


def _compose_welcome(scope_title: str | None, objectives: list[str]) -> str:
    """Deterministic German welcome — only the objectives inside come from the LLM."""
    title_part = f" Wir arbeiten mit **{scope_title}**." if scope_title else ""
    lines = [f"🎓 Willkommen im Lernmodus!{title_part}", "", "Darum geht es hier:"]
    lines += [f"- {objective}" for objective in objectives]
    lines += [
        "",
        "Womit möchtest du anfangen — eines dieser Themen oder eine eigene Frage aus dem Modul? "
        "Erzähl mir gern auch kurz, was du dazu schon weißt.",
        "",
        "_(Du kannst den Lernmodus jederzeit beenden, z. B. mit „stopp“.)_",
    ]
    return "\n".join(lines)


def _parse_extraction(content: str | None) -> tuple[list[str], list[str]]:
    """Parse LERNZIEL:/KONZEPT: lines from the extraction response."""
    objectives: list[str] = []
    concepts: list[str] = []
    if content:
        for line in content.strip().splitlines():
            line = line.strip()
            if line.upper().startswith("LERNZIEL:"):
                value = line.split(":", 1)[1].strip()
                if value:
                    objectives.append(value)
            elif line.upper().startswith("KONZEPT:"):
                value = line.split(":", 1)[1].strip()
                if value:
                    concepts.append(value)
    return objectives, concepts


@observe()
def socratic_v2_opening(state: GraphState) -> dict:
    """
    Opens a v2 learning session: content → objectives → welcome message.

    Changes:
    - Sets socratic_v2_phase = "core" for the next turn
    - Sets v2_learning_objectives, v2_key_concepts, v2_scope_title
    - Initializes v2_learner_model and the counters
    - Sets answer with the welcome message

    Without a course/module scope (or with an empty module) the session does
    NOT start — a deterministic message asks for a selection and all v2 state
    stays reset, so the next message is handled as normal chat again.
    """
    runtime_config = state["runtime_config"]
    course_id = runtime_config.get("course_id")
    module_id = runtime_config.get("module_id")

    if not course_id and not module_id:
        logger.debug("socratic_v2_opening: no course/module scope — not starting a session")
        return {**reset_socratic_v2_state(), "answer": NO_SCOPE_MESSAGE, "citations_markdown": None}

    # Multi-module scope: the last ID is the current/target module.
    target_module = module_id[-1] if isinstance(module_id, list) and module_id else module_id

    retriever = get_retriever()
    nodes = retriever.retrieve_all(course_id=course_id, module_id=target_module)
    content_nodes = [n for n in nodes if n.metadata.get("type") != "EmptyModule"]

    if not content_nodes:
        logger.debug(
            "socratic_v2_opening: no content in scope (course_id=%s, module_id=%s)", course_id, target_module
        )
        return {**reset_socratic_v2_state(), "answer": NO_CONTENT_MESSAGE, "citations_markdown": None}

    scope_title = next(
        (n.metadata.get("fullname") or n.metadata.get("title") for n in content_nodes if n.metadata.get("fullname") or n.metadata.get("title")),
        None,
    )

    content = "\n\n".join(n.text for n in content_nodes)[:MAX_EXTRACTION_CHARS]

    # MINI: long-context auxiliary extraction — deliberately NOT the runtime
    # model, because a large prompt through the GWDG path would routinely trip
    # the 7s timeout and mark GWDG unavailable for all other requests.
    response = _llm.chat(
        query=content,
        chat_history=[],
        model=Models.MINI,
        system_prompt=SOCRATIC_V2_OPENING_PROMPT,
    )
    objectives, concepts = _parse_extraction(response.content)

    # Fallback if extraction fails: session still starts, just without a
    # pre-structured objective list.
    if not objectives:
        objectives = [f"Die Inhalte von „{scope_title}“ verstehen" if scope_title else "Die Inhalte dieses Moduls verstehen"]

    langfuse_context.update_current_observation(
        metadata={
            "course_id": course_id,
            "target_module": target_module,
            "n_content_nodes": len(content_nodes),
            "n_objectives": len(objectives),
            "n_concepts": len(concepts),
            "scope_title": scope_title,
        }
    )

    return {
        "socratic_v2_phase": "core",
        "v2_learning_objectives": objectives,
        "v2_key_concepts": concepts,
        "v2_session_goal": None,
        "v2_learner_model": initial_learner_model(),
        "v2_target_concept": None,
        "v2_hint_count": 0,
        "v2_question_streak": 0,
        "v2_scope_title": scope_title,
        "answer": _compose_welcome(scope_title, objectives),
        "citations_markdown": None,
    }
