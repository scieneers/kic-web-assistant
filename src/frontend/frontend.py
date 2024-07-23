import json
from typing import Any

import streamlit as st
import streamlit_antd_components as sac
from fastapi.testclient import TestClient
from llama_index.core.llms import ChatMessage, MessageRole
from streamlit_feedback import streamlit_feedback

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.api.rest import app
from src.llm.LLMs import Models
from vectordb.qdrant import VectorDBQdrant


def parse_selected_course(selected_course):
    if selected_course is None:
        return

    if "_" in selected_course:
        if selected_course.startswith("cid_"):
            st.session_state["course_id"] = int(selected_course[4:])
        elif selected_course.startswith("mid_"):
            st.session_state["module_id"] = int(selected_course[4:])
        else:
            raise Exception(f"Unknown id format: {selected_course}")
    else:
        raise Exception(f"Unknown id format: {selected_course}")


@st.cache_resource
def instantiate_assistant() -> TestClient:
    return TestClient(app)


# The st_ant_tree component doesn't accept parent and child nodes with the same value.
# So we prepend every course_id with "cid_" and every module_id with "mid_". After the
# selection is done, we remove the prefix.
@st.cache_resource
def create_courses_modules_tree() -> list:
    course_records, module_records = VectorDBQdrant().get_course_module_records("web_assistant")
    tree_dict = {}

    # Sets to track unique course_ids and module_ids
    seen_courses = set()
    seen_modules = set()

    # Add Courses
    for record in course_records:
        payload = record.payload
        course_id = payload["course_id"]
        fullname = payload["fullname"]

        # If course_id is unique, create a parent node
        if course_id not in seen_courses:
            tree_dict[course_id] = {"value": f"cid_{course_id}", "title": fullname, "children": []}
            seen_courses.add(course_id)

    # Add Modules
    for record in module_records:
        payload = record.payload
        course_id = payload["course_id"]
        fullname = payload["fullname"]
        module_id = payload.get("module_id")

        if module_id not in seen_modules:
            child_node = {"value": f"mid_{module_id}", "title": fullname}
            tree_dict[course_id]["children"].append(child_node)
            seen_modules.add(module_id)

    # Convert the dictionary to a list
    tree_data = list(tree_dict.values())

    # Remove 'children' key if it is empty
    for item in tree_data:
        if not item["children"]:
            del item["children"]

    return tree_data


def get_course_module(course_or_module_index: int) -> dict:
    tree = create_courses_modules_tree()

    course_id = None
    course_name = None
    module_id = None
    module_name = None
    module_level = False
    i = -1
    flag_done = False
    for course in tree:
        if flag_done:
            break
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


def reset_history():
    st.session_state.messages = []


def reset_course_selection():
    st.session_state.course_selection = None


def set_course_selection():
    if type(st.session_state.course_selection) == list:
        index_selected = st.session_state.course_selection[0]
    else:
        index_selected = st.session_state.course_selection

    talk_to_course = get_course_module(index_selected)
    if talk_to_course["module_level"]:
        st.write(
            f"Talk to a module is activated for course: {talk_to_course['course_name']} and module: {talk_to_course['module_name']}"
        )
    else:
        st.write(f"Talk to a course is activated for course: {talk_to_course['course_name']}")


def submit_feedback(feedback: dict, trace_id: str):
    score = 1 if feedback["score"] == "👍" else 0

    response = st.session_state.assistant.post(
        "/api/feedback",
        headers={"Api-Key": "example_todelete_123"},
        json={"response_id": trace_id, "feedback": feedback["text"], "score": score},
    )

    if response.status_code != 200:
        raise ValueError(f"Error: {response}")


# Starting Bot ---------
st.title("KI-Campus Assistant")

if "llm_select" not in st.session_state:
    st.session_state.llm_select = Models.GPT4

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
# Initialize assistant
with st.chat_message("assistant"):
    st.write("Bitte warten...")
    tree_of_courses = create_courses_modules_tree()
    assistant = instantiate_assistant()
    st.write("Hallo 👋 Wie kann ich Ihnen helfen?")

    sac.tree(
        items=tree_of_courses,
        on_change=set_course_selection,
        key="course_selection",
        size="lg",
        return_index=True,
        label="Make a selection to talk to a course - or module",
    )

    st.button("Deactivate talk to a course", on_click=reset_course_selection)


# Initialize assistant
if "assistant" not in st.session_state or not st.session_state.assistant:
    with st.empty():  # Use st.empty to hold the place for conditional messages
        st.write("Bitte warten...")
        st.session_state.assistant = instantiate_assistant()


with st.chat_message("assistant"):
    st.write("Hallo 👋 Wie kann ich Ihnen helfen?")

# Initialize chat history & display chat messages from history on app rerun
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# React to user input
if query := st.chat_input("Wie lautet Ihre Frage?"):
    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append({"role": MessageRole.USER, "content": query})
    response = st.session_state.assistant.post(
        "/api/chat",
        headers={"Api-Key": "example_todelete_123"},
        json={"messages": st.session_state.messages, "model": st.session_state.llm_select},
    )
    if response.status_code != 200:
        raise ValueError(f"Error: {response}")

    response_content = json.loads(response.content)
    with st.chat_message("assistant"):
        st.markdown(response_content["message"])

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
