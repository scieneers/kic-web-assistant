"""Socratic v2 Opening Node — loads module content and starts the session.

First step of the v2 workflow ("Lernmodus v2"):
1. Fetch ALL chunks of the current module from the search index (same
   mechanism as the summarize scenario, see retriever.retrieve_all) — this is
   how the tutor "gets the content" without any ingestion changes.
2. One LLM call extracts learning objectives + key concepts from that content.
3. Emit a deterministic welcome message that names the objectives and asks the
   learner what to work on and what they already know.

Multi-module scope: module_id may be a list when the learner selects several
modules of one course. ALL selected modules are treated equally — objectives,
key concepts, the welcome scope title and the quiz pool are built from the
combined content of every selected module (not just one). The extraction
input's character budget is split evenly per module (_cap_content_per_module)
so a content-heavy module can't crowd smaller ones out of the LLM's input.
"""

import logging

from langfuse.decorators import langfuse_context, observe

from src.llm.objects.LLMs import LLM, Models
from src.llm.prompts.prompt_loader import load_prompt
from src.llm.state.models import GraphState
from src.llm.state.socratic_v2_routing import (
    GRADEABLE_QUIZ_KINDS,
    initial_learner_model,
    reset_socratic_v2_state,
)
from src.llm.tools.retrieve import get_retriever
from src.vectordb.doc_types import QUIZ_ITEM

logger = logging.getLogger(__name__)

# Load prompt once at module level
SOCRATIC_V2_OPENING_PROMPT = load_prompt("socratic_v2_opening")

# Initialize LLM instance at module level
_llm = LLM()

# Cap the extraction input — whole modules can be large, and the extraction
# only needs enough material to name objectives/concepts, not every chunk.
# Split evenly across the selected modules (see _cap_content_per_module) so a
# single content-heavy module can't consume the whole budget and starve the
# others out of the extraction entirely.
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


def _cap_content_per_module(content_nodes: list, max_chars: int) -> str:
    """Concatenate node text, capping each module's share of the budget evenly.

    content_nodes arrive grouped module-by-module (retrieve_all sorts by
    source_doc_key, see retriever._scope_sort_key), not interleaved. A single
    global [:max_chars] cut would then let one content-heavy module consume
    the whole budget and cut later modules out of the extraction input
    entirely. Splitting the budget evenly per module_id keeps every selected
    module represented regardless of size or sort order.
    """
    modules: dict[object, list[str]] = {}
    order: list[object] = []
    for node in content_nodes:
        key = (node.metadata or {}).get("module_id")
        if key not in modules:
            modules[key] = []
            order.append(key)
        modules[key].append(node.text)

    per_module_budget = max_chars // len(order) if order else max_chars
    return "\n\n".join("\n\n".join(modules[key])[:per_module_budget] for key in order)


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

    # Multi-module scope: all selected modules are worked on equally. The
    # retriever accepts the whole list, so content/objectives/quiz span them.
    retriever = get_retriever()
    nodes = retriever.retrieve_all(course_id=course_id, module_id=module_id)
    content_nodes = [n for n in nodes if n.metadata.get("type") != "EmptyModule"]

    if not content_nodes:
        logger.debug(
            "socratic_v2_opening: no content in scope (course_id=%s, module_id=%s)", course_id, module_id
        )
        return {**reset_socratic_v2_state(), "answer": NO_CONTENT_MESSAGE, "citations_markdown": None}

    # Scope title spans every selected module (dedup, keep order); cap the
    # length so the welcome line stays readable when many modules are picked.
    module_titles: list[str] = []
    for n in content_nodes:
        name = n.metadata.get("fullname") or n.metadata.get("title")
        if name and name not in module_titles:
            module_titles.append(name)
    if not module_titles:
        scope_title = None
    elif len(module_titles) <= 3:
        scope_title = ", ".join(module_titles)
    else:
        scope_title = ", ".join(module_titles[:3]) + " u. a."

    content = _cap_content_per_module(content_nodes, MAX_EXTRACTION_CHARS)

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

    # Real course quiz items across all selected modules (structured QuizItem
    # docs with known solutions) → enable the deterministic QUIZ move. Best-
    # effort: a failing typed fetch must not block the session opening.
    quiz_items: list[dict] = []
    try:
        quiz_nodes = retriever.retrieve_items(QUIZ_ITEM, course_id=course_id, module_id=module_id, top=0)
        for node in quiz_nodes:
            payload = (node.metadata or {}).get("payload") or {}
            if payload.get("kind") in GRADEABLE_QUIZ_KINDS and payload.get("question"):
                quiz_items.append({**payload, "asked": False})
    except Exception:
        logger.warning("socratic_v2_opening: quiz item fetch failed — QUIZ move disabled", exc_info=True)

    langfuse_context.update_current_observation(
        metadata={
            "course_id": course_id,
            "module_id": module_id,
            "n_content_nodes": len(content_nodes),
            "n_objectives": len(objectives),
            "n_concepts": len(concepts),
            "n_quiz_items": len(quiz_items),
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
        "v2_core_turns": 0,
        "v2_scope_title": scope_title,
        "v2_quiz_items": quiz_items,
        "v2_pending_quiz": None,
        "v2_last_policy_move": None,
        "v2_last_move": None,
        "answer": _compose_welcome(scope_title, objectives),
        "citations_markdown": None,
    }
