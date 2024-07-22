from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import APIKeyHeader
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import ChatMessage, MessageRole
from pydantic import BaseModel, Field, field_validator, model_validator

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.llm.assistant import KICampusAssistant
from src.llm.LLMs import Models

app = FastAPI()

# Langfuse tracing
langfuse_handler = LlamaIndexCallbackHandler()
Settings.callback_manager = CallbackManager([langfuse_handler])

# authentication with OAuth2
api_key_hearder = APIKeyHeader(name="Api-Key")


async def api_key_auth(api_key: Annotated[str, Depends(api_key_hearder)]):
    ALLOWED_API_KEYS = ["example_todelete_123"]

    if api_key not in ALLOWED_API_KEYS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")


# APIs
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
    model: Models = Field(
        default=Models.GPT4,
        description="The LLM to use for the conversation.",
        examples=[Models.GPT4, Models.MISTRAL],
    )

    @field_validator("messages", mode="after")
    @classmethod
    def final_message_is_user(cls, messages: list[SerializableChatMessage]) -> list[SerializableChatMessage]:
        if messages[-1].role != MessageRole.USER:
            raise ValueError("The last message must be a user message.")
        return messages

    def get_chat_history(self) -> list[ChatMessage]:
        return [message.to_chat_message() for message in self.messages[:-1]]

    def get_user_query(self) -> str:
        return self.messages[-1].content


class ChatResponse(BaseModel):
    message: SerializableChatMessage = Field(
        description="The assistant response to the user message.",
        examples=["I can help you with that. What is the assignment about?"],
    )
    response_id: str = Field(description="An ID for the response, that is needed for using the feedback endpoint.")


@app.post("/api/chat", dependencies=[Depends(api_key_auth)])
@observe()
def chat(chat_request: ChatRequest) -> ChatResponse:
    """Returns the response to the user message in one response (no streaming)."""
    assistant = KICampusAssistant()

    # if chat_request.course_id:
    #     llm_response = assistant.chat_with_course(
    #         query=chat_request.get_user_query(), chat_history=chat_request.get_chat_history(), model=chat_request.model, course=chat_request.course_id, module=chat_request.module_id
    #     )
    # else:
    llm_response = assistant.chat(
        query=chat_request.get_user_query(), chat_history=chat_request.get_chat_history(), model=chat_request.model
    )

    trace_id = langfuse_context.get_current_trace_id()
    if not trace_id:
        trace_id = "TRACING_UNAVAILABLE"
    chat_response = ChatResponse(message=SerializableChatMessage.from_chat_message(llm_response), response_id=trace_id)
    return chat_response


class FeedbackRequest(BaseModel):
    response_id: str = Field(description="The ID of the response that the feedback belongs to.")
    feedback: str = Field(description="Feedback on the conversation.")
    score: int = Field(default="Score between 0 and 1, where 1 is good and 0 is bad.")

    @field_validator("score", mode="after")
    @classmethod
    def validate_score(cls, score: int) -> int:
        if 0 <= score <= 1:
            raise ValueError("Score must be between 0 and 1.")
        return score


@app.post("/api/feedback", dependencies=[Depends(api_key_auth)])
def track_feedback(feedback_request: FeedbackRequest):
    """Update feedback in langfuse logs."""
    Langfuse().score(
        trace_id=feedback_request.response_id,
        name="user-explicit-feedback",
        value=feedback_request.score,
        comment=feedback_request.feedback,
    )
