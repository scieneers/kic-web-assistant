from llama_index.core.llms import ChatMessage, MessageRole
from pydantic import BaseModel


class SerializableChatMessage(BaseModel):
    # The ChatMessage from llama_index are not serializable
    role: MessageRole
    content: str

    def to_chat_message(self) -> ChatMessage:
        return ChatMessage(role=self.role, content=self.content)

    @staticmethod
    def from_chat_message(chat_message: ChatMessage) -> "SerializableChatMessage":
        if type(chat_message.content) != str:
            raise ValueError(f"Response content is not a string: {chat_message.content}")
        return SerializableChatMessage(role=chat_message.role, content=chat_message.content)
