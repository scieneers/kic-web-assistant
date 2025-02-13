import json

import streamlit as st
import streamlit_antd_components as sac
from fastapi.testclient import TestClient
from llama_index.core.llms import MessageRole
from streamlit_antd_components import AntIcon
from streamlit_feedback import streamlit_feedback

from src.api.rest import app
from src.env import env
from src.llm.LLMs import Models
from vectordb.qdrant import VectorDBQdrant


@st.cache_resource
def instantiate_assistant() -> TestClient:
    return TestClient(app)


# The st_ant_tree component doesn't accept parent and child nodes with the same value.
# So we prepend every course_id with "cid_" and every module_id with "mid_". After the
# selection is done, we remove the prefix.
@st.cache_resource
def create_courses_modules_tree() -> list:
    course_records, module_records = VectorDBQdrant("prod_remote").get_course_module_records("web_assistant")
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


def submit_feedback(feedback: dict, trace_id: str):
    score = 1 if feedback["score"] == "👍" else 0

    response = st.session_state.assistant.post(
        "/api/feedback",
        headers={"Api-Key": env.REST_API_KEYS[0]},
        json={"response_id": trace_id, "feedback": feedback["text"], "score": score},
    )

    if response.status_code != 200:
        raise ValueError(f"Error: {response}")


# Starting Bot ---------
st.title("KI-Campus Assistant")

if "llm_select" not in st.session_state:
    st.session_state.llm_select = Models.GPT4

if "course_id" not in st.session_state:
    st.session_state.course_id = None

if "module_id" not in st.session_state:
    st.session_state.module_id = None


with st.sidebar:
    st.session_state["llm_select"] = st.selectbox(
        "LLM Modelauswahl",
        options=(model for model in Models),
        index=0,
        on_change=reset_history,
        format_func=lambda model: model.value,
        placeholder=Models.GPT4.name,
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
if "assistant" not in st.session_state or not st.session_state.assistant:
    with st.empty():  # Use st.empty to hold the place for conditional messages
        st.write("Bitte warten...")
        st.session_state.assistant = instantiate_assistant()


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

    st.session_state.messages.append({"role": MessageRole.USER, "content": query})
    response = st.session_state.assistant.post(
        "/api/chat",
        headers={"Api-Key": env.REST_API_KEYS[0]},
        json={
            "messages": st.session_state.messages,
            "model": st.session_state.llm_select,
            "course_id": st.session_state.course_id if hasattr(st.session_state, "course_id") else None,
            "module_id": st.session_state.module_id if hasattr(st.session_state, "module_id") else None,
        },
    )
    if response.status_code != 200:
        raise ValueError(f"Error: {response.content}")

    response_content = json.loads(response.content)
    with st.chat_message("assistant"):
        st.markdown(response_content["message"], unsafe_allow_html=True)

    st.session_state["trace_id"] = response_content["response_id"]
    st.session_state.messages.append({"role": MessageRole.ASSISTANT, "content": response_content["message"]})

if trace_id := st.session_state.get("trace_id"):
    streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label="[Optional] Please provide an explanation",
        on_submit=submit_feedback,
        kwargs={"trace_id": trace_id},
        key=f"run-{trace_id}",
    )
