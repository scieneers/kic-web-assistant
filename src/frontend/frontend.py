import streamlit as st
from src.llm.inference import KICampusAssistant
from llama_index.llms import ChatMessage, MessageRole

@st.cache_resource
def instantiate_assistant() -> KICampusAssistant:
    return KICampusAssistant()

st.title('KI-Campus Assistant')
with st.chat_message('assistant'):
    st.write('Hello 👋')

# Initialize chat history & display chat messages from history on app rerun
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])    

# React to user input
if query := st.chat_input('What is up?'):
    with st.chat_message('user'):
        st.markdown(query)
else:
    st.stop()
    
assistant = instantiate_assistant()
chat_history = [ChatMessage(role=message['role'], content=message['content']) for message in st.session_state.messages]
response = assistant.chat(query=query, chat_history=chat_history)
with st.chat_message('assistant'):
    st.markdown(response)

st.session_state.messages.append({'role': MessageRole.USER, 'content': query})
st.session_state.messages.append({'role': MessageRole.ASSISTANT, 'content': response})

