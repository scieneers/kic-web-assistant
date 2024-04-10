import streamlit as st
from langfuse import Langfuse
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole
from streamlit_feedback import streamlit_feedback

from src.llm.inference import KICampusAssistant


@st.cache_resource
def instantiate_assistant() -> KICampusAssistant:
    return KICampusAssistant()


st.title("KI-Campus Assistant")

# Initialize assistant
with st.chat_message("assistant"):
    st.write("Please wait...")
    assistant = instantiate_assistant()
    st.write("Hello 👋 How can I help you?")

# Initialize chat history & display chat messages from history on app rerun
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if query := st.chat_input("What is up?"):
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
