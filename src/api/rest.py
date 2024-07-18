from contextlib import asynccontextmanager
from enum import Enum

# Fixing MIME types for static files under Windows
from typing import Annotated

import requests
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import APIKeyHeader
from langfuse import Langfuse
from llama_index.core.llms import ChatMessage, MessageRole
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.responses import FileResponse

from llm.assistant import KICampusAssistant
from src.api.models.serializable_chat_message import SerializableChatMessage

app = FastAPI()

# authentication with OAuth2
api_key_hearder = APIKeyHeader(name="Api-Key")


async def api_key_auth(api_key: Annotated[str, Depends(api_key_hearder)]):
    ALLOWED_API_KEYS = ["example_todelete_123"]

    if api_key not in ALLOWED_API_KEYS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")


# Rest API Endpoints
class ModelChoices(str, Enum):
    GPT_4 = "gpt-4"
    MISTRAL = "mistral"


@app.get("/health")
def health() -> str:
    """Health check endpoint. Returns 'OK' if the service is running."""
    return "OK"


class RetrievalRequest(BaseModel):
    message: str = Field(
        description="The query to find the most fitting sources to.",
        examples=["How much is the maximum dental police?"],
    )
    course_id: int | None = Field(
        default=None,
        description="The course identifier to restrict the search on.",
        examples=[79, 102, 91],
    )
    module_id: int | None = Field(
        default=None,
        description="The course module / topic / unit to restrict the search on. course_id is required when module_id is set.",
        examples=[1, 102, 33],
    )
    get_content_embeddings: bool = Field(
        default=False, description="If True, include the embeddings for the retrieved documents."
    )

    @model_validator(mode="after")
    def validate_module_id(self):
        if self.course_id and not self.module_id:
            raise HTTPException(
                status_code=400,
                detail="module_id is required when course_id is set.",
            )
        return self


@app.post("/api/retrieval", dependencies=[Depends(api_key_auth)])
def retrieval(retrieval_request: RetrievalRequest):
    """Returns the most similar documents from the Search Index."""
    pass


class ChatRequest(BaseModel):
    messages: list[SerializableChatMessage] = Field(
        description="All chat messages of the current conversation. The most recent should be a user message.",
        examples=[
            [
                SerializableChatMessage(content="Hello", role=MessageRole.USER),
                SerializableChatMessage(content="Hi, how can I help?", role=MessageRole.ASSISTANT),
                SerializableChatMessage(content="I need help with my assignment", role=MessageRole.USER),
            ]
        ],
    )
    course_id: int | None = Field(
        default=None,
        description="The course identifier to restrict the search on.",
        examples=[79, 102, 91],
    )
    module_id: int | None = Field(
        default=None,
        description="The course module / topic / unit to restrict the search on. course_id is required when module_id is set.",
        examples=[1, 102, 33],
    )
    model: ModelChoices = Field(
        default=ModelChoices.GPT_4,
        description="The LLM to use for the conversation.",
        examples=[ModelChoices.GPT_4, ModelChoices.MISTRAL],
    )


class ChatResponse(BaseModel):
    message: str = Field(
        description="The assistant response to the user message.",
        examples=["I can help you with that. What is the assignment about?"],
    )
    response_id: str = Field(description="An ID for the response, that is needed for using the feedback endpoint.")


@app.post("/api/chat", dependencies=[Depends(api_key_auth)])
def chat(chat_request: ChatRequest) -> ChatResponse:
    """Returns the response to the user message in one response (no streaming)."""
    assistant = KICampusAssistant()

    assistant.chat(query=chat_request)
    pass


class FeedbackRequest(BaseModel):
    response_id: str = Field(description="The ID of the response that the feedback belongs to.")
    feedback: str = Field(description="Feedback on the conversation.")
    score: int = Field(default="Score between 0 and 1, where 1 is good and 0 is bad.")


@app.post("/api/feedback", dependencies=[Depends(api_key_auth)])
def track_feedback(feedback_request: FeedbackRequest):
    """Update feedback in langfuse logs."""
    pass
