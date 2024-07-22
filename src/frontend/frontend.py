import streamlit as st
from llama_index.core.llms import ChatMessage, MessageRole
from st_ant_tree import st_ant_tree
from streamlit_feedback import streamlit_feedback

from src.llm.assistant import KICampusAssistant
from src.llm.LLMs import Models
from src.vectordb.qdrant import VectorDBQdrant


@st.cache_resource
def instantiate_assistant() -> KICampusAssistant:
    return KICampusAssistant()


@st.cache_resource
def load_tree() -> list:
    return [
        {
            "value": "18",
            "title": "Testkurs 1",
            "children": [
                {"value": "422", "title": "Testmodul 1"},
                {"value": "423", "title": "Testmodul 2"},
            ],
        },
        {
            "value": "19",
            "title": "Testkurs 2",
        },
    ]


def empty_history():
    st.session_state.messages = []


st.title("KI-Campus Assistant")

if "llm_select" not in st.session_state:
    st.session_state["llm_select"] = Models.gpt4
with st.sidebar:
    st.session_state["llm_select"] = st.selectbox(
        "Bitte wählen Sie das LLM aus",
        options=(model for model in Models),
        index=0,
        on_change=empty_history,
        format_func=lambda model: model.value,
        placeholder=Models.gpt4.name,
    )

# Initialize assistant
with st.chat_message("assistant"):
    st.write("Bitte warten...")
    tree_of_courses = load_tree()
    assistant = instantiate_assistant()
    st.write("Hallo 👋 Wie kann ich Ihnen helfen?")

selected_course = st_ant_tree(
    treeData=tree_of_courses,
    placeholder="Hier können Sie Antworten auf einen Kurs oder Modul begrenzen",
    treeCheckable=False,
)

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

    chat_history = [
        ChatMessage(role=message["role"], content=message["content"]) for message in st.session_state.messages
    ]
    response = assistant.chat(query=query, chat_history=chat_history, model=st.session_state.llm_select)

    with st.chat_message("assistant"):
        st.markdown(response)

    # st.session_state["trace_id"] = trace_id

    st.session_state.messages.append({"role": MessageRole.USER, "content": query})
    st.session_state.messages.append({"role": MessageRole.ASSISTANT, "content": response})

# Implement feedback system
# if st.session_state.get("trace_id"):
#     trace_id = st.session_state.get("trace_id")
#     feedback = streamlit_feedback(
#         feedback_type="thumbs",
#         optional_text_label="[Optional] Please provide an explanation",
#         on_submit=assistant.submit_feedback,
#         kwargs={"trace_id": trace_id},
#         key=f"run-{trace_id}",
#     )
