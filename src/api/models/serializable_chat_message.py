from llama_index.core.llms import MessageRole
from pydantic import BaseModel


class SerializableChatMessage(BaseModel):
    "The ChatMessage from llama_index is not serializable"
    role: MessageRole
    content: str
