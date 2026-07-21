import json
import re
import time
import httpx
import streamlit as st
import random
import streamlit_antd_components as sac
from llama_index.core.llms import MessageRole
from streamlit_antd_components import AntIcon
from streamlit_feedback import streamlit_feedback

from src.env import env
from src.llm.objects.LLMs import Models
from src.vectordb.azure_search import VectorDBAzureSearch


# Texts shown while waiting for the first streamed token.
# You can adjust/extend this list later.
THINKING_PHRASES = [
    "Aktiviere Neuronen",
    "Philosophiere über KI",
    "Verstärke Konzentration",
    "Tue intelligente Dinge",
    "Denke angestrengt nach",
    "Verarbeite seriös die Anfrage",
    "Optimiere Gedankenfluss"
]


def render_thinking_indicator(phrases: list[str], switch_seconds: float = 1.0) -> str:
    """Return HTML/CSS for a non-iframe thinking indicator.

    We render this via `st.markdown(..., unsafe_allow_html=True)` so it inherits
    the Streamlit chat bubble styling (no iframe / no background changes).

    Implementation:
    - rotate through phrases using CSS opacity animation
    - show dot animation using ::after
    """

    random.shuffle(phrases)

    # We animate one item at a time. Each phrase gets an animation delay.
    cycle = max(1, len(phrases)) * switch_seconds

    items = []
    for i, phrase in enumerate(phrases):
        delay = i * switch_seconds
        # Use HTML escape via replacement (avoid importing html module just for this)
        safe_phrase = (
            phrase.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        items.append(
            f'<span class="thinking-item" style="animation-delay:{delay:.3f}s">'
            f'{safe_phrase}<span class="thinking-dots"></span>'
            f"</span>"
        )

    items_html = "".join(items)

    # Note: no leading indentation -> avoid markdown code block rendering.
    return (
        f"<style>"
        f".thinking-wrap{{position:absolute;display:inline-block;font-style:italic;opacity:.85;}}"
        f".thinking-item{{position:absolute;left:0;top:0;opacity:0;white-space:nowrap;"
        f"animation:thinking-show {cycle:.3f}s linear infinite;}}"
        f"@keyframes thinking-show{{0%{{opacity:1}}{(switch_seconds/cycle)*100 - 10:.2f}%{{opacity:1}}"
        f"{(switch_seconds/cycle)*100:.2f}%{{opacity:0}}100%{{opacity:0}}}}"
        f".thinking-dots::after{{content:'';animation:thinking-dots 1.2s infinite;white-space:pre;}}"
        f"@keyframes thinking-dots{{0%{{content:''}}25%{{content:'.'}}50%{{content:'..'}}75%{{content:'...'}}100%{{content:''}}}}"
        f"</style>"
        f"<span class=\"thinking-wrap\">{items_html}</span>"
    )


# Streamlit's markdown renderer treats ":word" (colon directly followed by a
# letter) as its colored-text directive (`:color[text]`). Only a handful of
# color names are recognized (red, blue, green, violet, orange, gray, grey,
# rainbow) — any other word is parsed as an unknown directive and silently
# dropped, colon included. This bites German colon-gendering (e.g. "Ärzt:innen"
# renders as "Ärzt"). Escaping the colon disables directive parsing for it.
GENDER_COLON_PATTERN = re.compile(r"(?<=\w):(?=[a-zA-ZäöüÄÖÜß])")


def escape_markdown_directives(text: str) -> str:
    return GENDER_COLON_PATTERN.sub(r"\\:", text)


@st.cache_resource
def get_api_client() -> httpx.Client:
    # Long-ish timeout because LLM responses can take a while.
    return httpx.Client(base_url=env.REST_API_URL, timeout=httpx.Timeout(120.0))


# The st_ant_tree component doesn't accept parent and child nodes with the same value.
# So we prepend every course_id with "cid_" and every module_id with "mid_". After the
# selection is done, we remove the prefix.
@st.cache_resource
def create_courses_modules_tree() -> list:
    course_records, module_records = VectorDBAzureSearch().get_course_module_records()
    tree_dict = {}

    # Sets to track unique course_ids and module_ids
    seen_courses = set()
    seen_modules = set()

    tree_dict[0] = sac.TreeItem(
        "Alle Inhalte aus Drupal",
        icon=AntIcon(name="GlobalOutlined"),
        description=None,
    )

    # Add Courses
    for record in course_records:
        payload = record.payload
        course_id = payload["course_id"]
        fullname = payload["fullname"]

        if course_id not in seen_courses:
            tree_dict[course_id] = sac.TreeItem(fullname, description=course_id, children=[])
            seen_courses.add(course_id)

    # Add Modules
    for record in module_records:
        payload = record.payload
        course_id = payload["course_id"]
        fullname = payload["fullname"]
        module_id = payload.get("module_id")

        if module_id not in seen_modules:
            if course_id not in tree_dict:
                tree_dict[course_id] = sac.TreeItem(f"Kurs {course_id}", description=course_id, children=[])
                seen_courses.add(course_id)
            child_node = sac.TreeItem(fullname, description=module_id)
            tree_dict[course_id].children.append(child_node)
            seen_modules.add(module_id)

    # Convert the dictionary to a list of TreeItem
    tree_items = list(tree_dict.values())
    return tree_items


def _flatten_tree_nodes() -> list[dict]:
    """DFS-ordered flat list of tree nodes, matching sac.tree's return_index numbering.

    sac.tree(return_index=True) numbers nodes depth-first: root=0, then each
    course, immediately followed by that course's modules, then the next
    course. Every helper that maps an `index` (or list of indices) back to
    IDs relies on this exact ordering.

    Each entry carries: index, course_id, module_id, is_module, is_root.
    """
    tree = create_courses_modules_tree()
    nodes = []
    i = 0
    for course in tree:
        nodes.append(
            {
                "index": i,
                "course_id": course.description,
                "module_id": None,
                "is_module": False,
                "is_root": course.description is None,  # "Alle Inhalte aus Drupal"
            }
        )
        i += 1
        if course.children:
            for module in course.children:
                nodes.append(
                    {
                        "index": i,
                        "course_id": course.description,
                        "module_id": module["description"],
                        "is_module": True,
                        "is_root": False,
                    }
                )
                i += 1
    return nodes


def _restrict_to_single_course(new_indices: list[int], prev_indices: list[int]) -> list[int]:
    """Prune a checkbox selection down to at most one course.

    The tree's checkboxes physically allow ticking nodes across several
    courses, but the API only accepts module IDs belonging to a single
    course_id. Rather than rejecting an invalid selection after the fact
    (which leaves the stray checkboxes visually ticked), we auto-switch:
    checking something in a new course drops the previously selected course.

    on_change fires per single click, so the item(s) in `new` but not in
    `prev` identify the course the user just moved into.

    - checking the root ("Alle Inhalte") wins -> whole corpus
    - otherwise keep only the nodes of the course the user just touched
    """
    by_index = {n["index"]: n for n in _flatten_tree_nodes()}
    new = [i for i in new_indices if i in by_index]
    added = [i for i in new if i not in set(prev_indices)]

    # A freshly ticked root means "talk to everything" — clear the rest.
    root_idx = next((n["index"] for n in by_index.values() if n["is_root"]), None)
    if any(by_index[i]["is_root"] for i in added):
        return [root_idx] if root_idx is not None else []

    non_root = [i for i in new if not by_index[i]["is_root"]]
    courses = {by_index[i]["course_id"] for i in non_root}
    if len(courses) <= 1:
        # Single course (or nothing but root): a course pick beats the corpus,
        # so drop a co-selected root and keep the course nodes.
        return non_root if non_root else new

    # Spans >1 course: keep only the course the user just clicked into
    # (fall back to any present course if the diff is ambiguous).
    added_courses = {by_index[i]["course_id"] for i in added if not by_index[i]["is_root"]}
    target = next(iter(added_courses), next(iter(courses)))
    return [i for i in non_root if by_index[i]["course_id"] == target]


def resolve_selection(indices: list[int]) -> dict:
    """Map a list of checked tree indices to a (course_id, module_id) filter.

    Assumes the indices have already been restricted to a single course
    (see _restrict_to_single_course). The API requires all module IDs to
    belong to one course_id.

    - root / nothing checked         -> course_id=None, module_id=None   (whole corpus)
    - only a course node checked      -> course_id=X,    module_id=None   (whole course)
    - one or more module nodes checked-> course_id=X,    module_id=[...]  (those modules)

    Explicit module checkboxes always win over a co-checked course node: the
    more specific selection is what the user means. (checkbox_strict means the
    course node is never auto-ticked, so a co-check is an intentional extra.)
    """
    by_index = {n["index"]: n for n in _flatten_tree_nodes()}
    selected = [by_index[i] for i in indices if i in by_index]

    course_nodes = [n for n in selected if not n["is_root"] and not n["is_module"]]
    module_nodes = [n for n in selected if n["is_module"]]
    root_selected = any(n["is_root"] for n in selected)

    # Root or empty selection means "talk to everything".
    if root_selected or not selected:
        return {"course_id": None, "module_id": None}

    course_id = next(iter({n["course_id"] for n in course_nodes} | {n["course_id"] for n in module_nodes}))
    # Specific modules win; only a bare course selection means the whole course.
    module_id = [n["module_id"] for n in module_nodes] if module_nodes else None
    return {"course_id": course_id, "module_id": module_id}


def _tree_default_indices() -> list[int]:
    """Reflects the last known tree selection back as its `index`.

    sac.tree() treats `index` as authoritative on every rerun, not just on
    first mount — passing a hardcoded default would snap the selection back on
    any unrelated rerun (e.g. a different sidebar button), even though the user
    never touched the tree. It also lets us reject an invalid (multi-course)
    selection by snapping the widget back to the last valid indices.

    Reads the non-widget mirror `course_selection_indices` (maintained in
    select_course_or_module) instead of the widget key itself: Streamlit drops
    widget state whenever the tree doesn't get instantiated during a run, while
    the mirror survives — and it lets reset_history() clear the tree.
    """
    return st.session_state.get("course_selection_indices", [])


def _tree_widget_key() -> str:
    """Versioned key for the course tree.

    sac.tree only honours the `index` prop on a FRESH mount, not on later
    reruns — so setting the mirror alone can't visually un-tick a stray
    cross-course checkbox. Bumping `tree_version` (via _bump_tree) changes the
    key, which forces Streamlit to remount the widget; the fresh mount then
    reads `index=_tree_default_indices()` and renders exactly the pruned set.
    """
    return f"course_tree_{st.session_state.get('tree_version', 0)}"


def _bump_tree() -> None:
    """Remount the tree on the next run so it re-reads `index` (the mirror)."""
    st.session_state.tree_version = st.session_state.get("tree_version", 0) + 1


def select_course_or_module():
    # Read the value under the *current* versioned key (tree_version hasn't been
    # bumped yet at callback time, so this is the key the widget rendered with).
    raw = st.session_state.get(_tree_widget_key())
    # checkbox mode returns a list of checked indices; guard against a bare int/None.
    if isinstance(raw, list):
        indices = raw
    elif raw is None:
        indices = []
    else:
        indices = [raw]

    # Enforce the single-course rule up front: ticking a node in another course
    # auto-switches instead of piling up a cross-course (API-invalid) selection.
    prev = st.session_state.get("course_selection_indices", [])
    pruned = _restrict_to_single_course(indices, prev)

    resolved = resolve_selection(pruned)

    if (
        st.session_state["course_id"] != resolved["course_id"]
        or st.session_state["module_id"] != resolved["module_id"]
    ):
        # keep_socratic: erst "Lernmodus starten" klicken und dann den Kurs
        # wählen ist ein legitimer Ablauf — die Aktivierung darf der
        # Kurswechsel-Reset nicht stillschweigend wieder löschen.
        # remount_tree=False: we manage the tree's own state below.
        reset_history(keep_socratic=True, remount_tree=False)

    st.session_state["course_id"] = resolved["course_id"]
    st.session_state["module_id"] = resolved["module_id"]
    # Mirror the pruned indices so a remount can restore exactly them.
    st.session_state.course_selection_indices = pruned
    # If pruning dropped a cross-course tick, remount to clear it visually
    # (sac.tree won't drop the tick from a plain index update).
    if set(pruned) != set(indices):
        _bump_tree()


def reset_history(keep_socratic: bool = False, remount_tree: bool = True):
    st.session_state.messages = []
    st.session_state.course_id = None
    st.session_state.module_id = None
    st.session_state.thread_id = None
    st.session_state.last_activity = None
    st.session_state._auto_restored = False
    st.session_state._load_error = None
    # remount_tree=False when called from the tree's own callback: the widget
    # already shows the new selection, and remounting would collapse the tree.
    # Hard resets (New Session / LLM change / TTL) keep the default so the tree
    # clears its ticks (sac.tree ignores a plain index update).
    if remount_tree:
        st.session_state.course_selection_indices = []
        _bump_tree()
    if not keep_socratic:
        st.session_state.start_socratic = False
        st.session_state.start_socratic_v2 = False
    st.experimental_set_query_params()


def _arm_socratic(version: str) -> None:
    """on_click-Callback für die Lernmodus-Buttons.

    Callbacks laufen VOR dem Rerun, den der Klick ohnehin auslöst — es
    braucht also kein st.rerun(). Das frühere `if st.button(...): ...
    st.rerun()`-Muster brach das Skript mitten in der Sidebar ab; die noch
    nicht instanziierten Widgets darunter (LLM-Auswahl, Kursbaum) verloren
    dadurch ihren Widget-State und sprangen auf den Default zurück.
    """
    st.session_state.start_socratic = version == "v1"
    st.session_state.start_socratic_v2 = version == "v2"


SESSION_TTL = 30 * 60  # 30 Minuten – muss mit Backend-Wert übereinstimmen


def _load_session_from_backend(thread_id: str, auto_restored: bool = False) -> None:
    """Lädt Gesprächsverlauf vom Backend – simuliert Moodle-Page-Reload.

    auto_restored=True: vom URL-Query-Param ausgelöst (Reload-Simulation).
    auto_restored=False: manuell über Sidebar-Eingabe ausgelöst.
    """
    response = get_api_client().get(
        f"/api/chat/history/{thread_id}",
        headers={"Api-Key": env.REST_API_KEYS[0]},
    )
    if response.status_code != 200:
        # Fehler über Session-State statt st.sidebar.error(): diese Funktion
        # läuft auch in on_click-Callbacks, wo direkt gerenderte Elemente
        # nicht zuverlässig in der Sidebar landen. Anzeige erfolgt im
        # Sidebar-Block unter dem Laden-Button.
        st.session_state._load_error = f"Session nicht gefunden: {response.status_code}"
        return
    data = response.json()
    messages = data.get("messages", [])
    st.session_state._load_error = None
    st.session_state.thread_id = thread_id
    st.session_state.last_activity = time.time()
    st.session_state._auto_restored = auto_restored
    st.session_state.messages = [
        {"role": msg["role"], "content": msg["content"]} for msg in messages
    ]


def _load_session_clicked() -> None:
    load_id = st.session_state.get("load_session_input", "").strip()
    if load_id:
        _load_session_from_backend(load_id)


def submit_feedback(feedback: dict, trace_id: str):
    score = 1 if feedback["score"] == "👍" else 0

    response = st.session_state.api_client.post(
        "/api/feedback",
        headers={"Api-Key": env.REST_API_KEYS[0]},
        json={"response_id": trace_id, "feedback": feedback["text"], "score": score},
    )

    if response.status_code != 200:
        raise ValueError(f"Error: {response}")


def _frontend_password_gate() -> None:
    """Blocks the page behind a single shared password if FRONTEND_PASSWORD is set.

    Skipped entirely when unconfigured (local dev). No Azure permissions needed —
    unlike App Service Easy Auth, this doesn't require an Azure AD app registration.
    """
    if not hasattr(env, "FRONTEND_PASSWORD"):
        return
    if st.session_state.get("_authenticated"):
        return

    st.title("🔒 KI-Campus Assistant")
    password = st.text_input("Passwort", type="password")
    if st.button("Anmelden"):
        if password == env.FRONTEND_PASSWORD:
            st.session_state._authenticated = True
            st.rerun()
        else:
            st.error("Falsches Passwort.")
    st.stop()


_frontend_password_gate()

# Starting Bot ---------
st.title("KI-Campus Assistant")

if "llm_select" not in st.session_state:
    st.session_state.llm_select = Models.GEMMA4_31B

if "course_id" not in st.session_state:
    st.session_state.course_id = None

if "module_id" not in st.session_state:
    st.session_state.module_id = None

if "tree_version" not in st.session_state:
    st.session_state.tree_version = 0

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "last_activity" not in st.session_state:
    st.session_state.last_activity = None

if "_auto_restored" not in st.session_state:
    st.session_state._auto_restored = False


with st.sidebar:
    st.caption("🎓 Sokratischer Lernmodus")
    if st.session_state.get("start_socratic"):
        st.info("v1 aktiviert für die nächste Nachricht.")
    if st.session_state.get("start_socratic_v2"):
        st.info("v2 aktiviert für die nächste Nachricht.")
    st.button("Lernmodus starten (v1)", key="start_socratic_btn", on_click=_arm_socratic, args=("v1",))
    st.button("✨ Lernmodus starten (v2)", key="start_socratic_v2_btn", on_click=_arm_socratic, args=("v2",))

    st.divider()
    st.caption("🧪 Moodle-Simulation")
    if st.session_state.thread_id:
        if st.session_state._auto_restored:
            st.success("↩ Session wiederhergestellt (Reload)")
        st.text("Aktive Session-ID:")
        st.code(st.session_state.thread_id, language=None)
    else:
        st.text("Keine aktive Session")

    st.text_input("Session-ID laden", placeholder="thread_id hier einfügen…", key="load_session_input")
    st.button("Laden", key="load_session_btn", on_click=_load_session_clicked)
    if st.session_state.get("_load_error"):
        st.error(st.session_state._load_error)
    st.button("Neue Session", key="new_session_btn", on_click=reset_history)

    st.divider()
    _model_options = list(Models)
    st.session_state["llm_select"] = st.selectbox(
        "LLM Modelauswahl",
        options=_model_options,
        index=_model_options.index(Models.GEMMA4_31B),
        on_change=reset_history,
        format_func=lambda model: model.value,
        placeholder=Models.GEMMA4_31B.name,
    )
    st.divider()

    sac.tree(
        items=create_courses_modules_tree(),
        index=_tree_default_indices(),
        key=_tree_widget_key(),
        size="sm",
        show_line=False,
        checkbox=True,
        # Eltern-/Kind-Knoten entkoppeln: ein Kurs-Häkchen = ganzer Kurs,
        # Modul-Häkchen = einzelnes Modul. Ohne strict würde das Ankreuzen
        # eines Kurses automatisch alle Module mit-selektieren.
        checkbox_strict=True,
        return_index=True,
        on_change=select_course_or_module,
        label="Mehrere Module eines Kurses auswählbar – oder ganzer Kurs",
    )

# Initialize assistant
if "api_client" not in st.session_state or not st.session_state.api_client:
    with st.empty():  # Use st.empty to hold the place for conditional messages
        st.write("Bitte warten...")
        st.session_state.api_client = get_api_client()


with st.chat_message("assistant"):
    st.write("Herzlich willkommen auf dem KI-Campus! Wie kann ich dir weiterhelfen?")

# Initialize chat history — auf frischem Page-Load URL-Param prüfen (Moodle: localStorage)
if "messages" not in st.session_state:
    url_thread_id = st.experimental_get_query_params().get("thread_id", [None])[0]
    if url_thread_id:
        _load_session_from_backend(url_thread_id, auto_restored=True)
        if "messages" in st.session_state:
            st.rerun()  # Sidebar neu rendern damit Badge + Session-ID sofort sichtbar sind
        # Laden fehlgeschlagen (z.B. abgelaufene Session in der URL): Param
        # entfernen, sonst rerunnt die Seite endlos gegen dieselbe tote thread_id.
        st.session_state.messages = []
        st.experimental_set_query_params()
    else:
        st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(escape_markdown_directives(message["content"]), unsafe_allow_html=True)


# React to user input
if query := st.chat_input("Wie lautet Ihre Frage?"):
    # TTL-Check: Session nach 30 min Inaktivität zurücksetzen
    if (
        st.session_state.last_activity is not None
        and time.time() - st.session_state.last_activity > SESSION_TTL
    ):
        reset_history()
        st.info("Session abgelaufen – neue Konversation gestartet.")

    st.session_state.last_activity = time.time()

    with st.chat_message("user"):
        st.markdown(escape_markdown_directives(query))

    # Store user message in UI history (for display only)
    st.session_state.messages.append({"role": MessageRole.USER, "content": query})

    payload = {
        "user_query": {"role": MessageRole.USER, "content": query},  # Single message object (not array)
        "model": st.session_state.llm_select.value
        if isinstance(st.session_state.llm_select, Models)
        else st.session_state.llm_select,
        "course_id": st.session_state.course_id if hasattr(st.session_state, "course_id") else None,
        "module_id": st.session_state.module_id if hasattr(st.session_state, "module_id") else None,
        "thread_id": st.session_state.thread_id,
        # pop: the trigger should only fire for this one message, the server
        # persists socratic_mode / socratic_v2_phase via the checkpointer for
        # subsequent turns.
        "start_socratic": st.session_state.pop("start_socratic", False),
        "start_socratic_v2": st.session_state.pop("start_socratic_v2", False),
    }

    # Stream tokens from backend and render progressively.
    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed_text = ""
        received_first_token = False

        # Render immediate "thinking" indicator until first token/final arrives.
        placeholder.markdown(
            render_thinking_indicator(THINKING_PHRASES, switch_seconds=3.0),
            unsafe_allow_html=True,
        )

        with st.session_state.api_client.stream(
            "POST",
            "/api/chat/stream",
            headers={"Api-Key": env.REST_API_KEYS[0]},
            json=payload,
        ) as response:
            if response.status_code != 200:
                raise ValueError(f"Error: {response.text}")

            for line in response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)

                if event.get("type") == "meta":
                    st.session_state.thread_id = event.get("thread_id")
                    st.session_state["trace_id"] = event.get("response_id")
                elif event.get("type") == "token":
                    if not received_first_token:
                        received_first_token = True
                        placeholder.empty()
                    streamed_text += event.get("token", "")
                    placeholder.markdown(escape_markdown_directives(streamed_text), unsafe_allow_html=True)
                elif event.get("type") == "final":
                    final_message = event.get("message", streamed_text)
                    if not received_first_token:
                        placeholder.empty()
                    placeholder.markdown(escape_markdown_directives(final_message), unsafe_allow_html=True)
                    st.session_state.thread_id = event.get("thread_id")
                    st.session_state["trace_id"] = event.get("response_id")
                    streamed_text = final_message
                elif event.get("type") == "error":
                    placeholder.empty()
                    placeholder.error(event.get("message", "Unknown streaming error"))
                    raise ValueError(event.get("message", "Unknown streaming error"))

    # Store assistant response in UI history (for display only)
    st.session_state.messages.append({"role": MessageRole.ASSISTANT, "content": streamed_text})
    # URL-Param setzen (Moodle-Analog: localStorage.setItem) → überlebt Browser-Reload
    if st.session_state.thread_id:
        st.experimental_set_query_params(thread_id=st.session_state.thread_id)
    # Rerun so the sidebar shows the updated thread_id immediately.
    st.rerun()

if trace_id := st.session_state.get("trace_id"):
    streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label="[Optional] Please provide an explanation",
        on_submit=submit_feedback,
        kwargs={"trace_id": trace_id},
        key=f"run-{trace_id}",
    )
