import streamlit as st
from llama_index.core.llms import ChatMessage, MessageRole
from streamlit_feedback import streamlit_feedback

from src.frontend.LLMs import LLM
from src.llm.inference import KICampusAssistant, KiCampusRetriever


@st.cache_resource
def instantiate_embedder():
    return KiCampusRetriever()


def instantiate_assistant(model) -> KICampusAssistant:
    return KICampusAssistant(model, instantiate_embedder())


def empty_history():
    st.session_state.messages = []


st.title("KI-Campus Assistant")

if "llm_select" not in st.session_state:
    st.session_state["llm_select"] = LLM.gpt4
with st.sidebar:
    st.session_state["llm_select"] = st.selectbox(
        "Bitte wählen Sie das LLM aus",
        options=(model for model in LLM),
        index=0,
        on_change=empty_history,
        format_func=lambda model: model.value,
        placeholder=LLM.gpt4.name,
    )

# Initialize assistant
with st.chat_message("assistant"):
    st.write("Bitte warten...")
    assistant = instantiate_assistant(st.session_state.llm_select)
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

    chat_history = [
        ChatMessage(role=message["role"], content=message["content"]) for message in st.session_state.messages
    ]
    response, trace_id = assistant.chat(query=query, chat_history=chat_history)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state["trace_id"] = trace_id

    st.session_state.messages.append({"role": MessageRole.USER, "content": query})
    st.session_state.messages.append({"role": MessageRole.ASSISTANT, "content": response})

# Implement feedback system
if st.session_state.get("trace_id"):
    trace_id = st.session_state.get("trace_id")
    feedback = streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label="[Optional] Please provide an explanation",
        on_submit=assistant.submit_feedback,
        kwargs={"trace_id": trace_id},
        key=f"run-{trace_id}",
    )
