import json
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


def convert_selected_index_to_id(course_or_module_index: int) -> dict:
    tree = create_courses_modules_tree()

    course_id = None
    course_name = None
    module_level = False
    i = -1
    flag_done = False
    for course in tree:
        if flag_done:
            break
        module_id = None
        module_name = None
        course_id = course.description
        course_name = course.label
        module_level = False
        i = i + 1
        if course_or_module_index == i:
            break
        if course.children:
            for module in course.children:
                module_id = module["description"]
                module_name = module["label"]
                module_level = True
                i = i + 1
                if course_or_module_index == i:
                    flag_done = True
                    break

    response = {
        "module_level": module_level,
        "course_id": course_id,
        "course_name": course_name,
        "module_id": module_id,
        "module_name": module_name,
    }

    return response


def select_course_or_module():
    # st.session_state.course_selection is a list with a single item
    # everytime the user collapses the tree :(
    if type(st.session_state.course_selection) is list:
        index_selected = st.session_state.course_selection[0]
    else:
        index_selected = st.session_state.course_selection

    talk_to_course_ids = convert_selected_index_to_id(index_selected)

    if (
        st.session_state["course_id"] != talk_to_course_ids["course_id"]
        or st.session_state["module_id"] != talk_to_course_ids["module_id"]
    ):
        reset_history()

    st.session_state["course_id"] = talk_to_course_ids["course_id"]
    st.session_state["module_id"] = talk_to_course_ids["module_id"]


def reset_history():
    st.session_state.messages = []
    st.session_state.course_id = None
    st.session_state.module_id = None
    st.session_state.thread_id = None  # Reset thread_id for new conversation


def submit_feedback(feedback: dict, trace_id: str):
    score = 1 if feedback["score"] == "👍" else 0

    response = st.session_state.api_client.post(
        "/api/feedback",
        headers={"Api-Key": env.REST_API_KEYS[0]},
        json={"response_id": trace_id, "feedback": feedback["text"], "score": score},
    )

    if response.status_code != 200:
        raise ValueError(f"Error: {response}")


# Starting Bot ---------
st.title("KI-Campus Assistant")

if "llm_select" not in st.session_state:
    st.session_state.llm_select = Models.AZURE_FALLBACK

if "course_id" not in st.session_state:
    st.session_state.course_id = None

if "module_id" not in st.session_state:
    st.session_state.module_id = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None  # Managed by backend, stored for UI display


with st.sidebar:
    st.session_state["llm_select"] = st.selectbox(
        "LLM Modelauswahl",
        options=(model for model in Models),
        index=0,
        on_change=reset_history,
        format_func=lambda model: model.value,
        placeholder=Models.AZURE_FALLBACK.name,
    )
    st.divider()

    sac.tree(
        items=create_courses_modules_tree(),
        index=0,
        key="course_selection",
        size="sm",
        show_line=False,
        checkbox=False,
        return_index=True,
        on_change=select_course_or_module,
        label="Make a selection to talk to a course - or module",
    )

# Initialize assistant
if "api_client" not in st.session_state or not st.session_state.api_client:
    with st.empty():  # Use st.empty to hold the place for conditional messages
        st.write("Bitte warten...")
        st.session_state.api_client = get_api_client()


with st.chat_message("assistant"):
    st.write("Herzlich willkommen auf dem KI-Campus! Wie kann ich dir weiterhelfen?")

# Initialize chat history & display chat messages from history on app rerun
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)


# React to user input
if query := st.chat_input("Wie lautet Ihre Frage?"):
    with st.chat_message("user"):
        st.markdown(query)

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
                    placeholder.markdown(streamed_text, unsafe_allow_html=True)
                elif event.get("type") == "final":
                    final_message = event.get("message", streamed_text)
                    if not received_first_token:
                        placeholder.empty()
                    placeholder.markdown(final_message, unsafe_allow_html=True)
                    st.session_state.thread_id = event.get("thread_id")
                    st.session_state["trace_id"] = event.get("response_id")
                    streamed_text = final_message
                elif event.get("type") == "error":
                    placeholder.empty()
                    placeholder.error(event.get("message", "Unknown streaming error"))
                    raise ValueError(event.get("message", "Unknown streaming error"))

    # Store assistant response in UI history (for display only)
    st.session_state.messages.append({"role": MessageRole.ASSISTANT, "content": streamed_text})

if trace_id := st.session_state.get("trace_id"):
    streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label="[Optional] Please provide an explanation",
        on_submit=submit_feedback,
        kwargs={"trace_id": trace_id},
        key=f"run-{trace_id}",
    )
