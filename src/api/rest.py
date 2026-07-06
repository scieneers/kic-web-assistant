from typing import Annotated

import json
import logging
import queue
import threading
import uuid

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe
from llama_index.core.llms import ChatMessage, MessageRole
from pydantic import BaseModel, Field, field_validator, model_validator

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.env import env
from src.llm.assistant import KICampusAssistant
from src.llm.objects.LLMs import Models
from src.vectordb.azure_search import VectorDBAzureSearch
from src.llm.streaming import TokenCallbackContext

# Set DEBUG level for all src.llm.* loggers when DEBUG_MODE is enabled.
# Third-party libraries stay at WARNING to avoid noise.
if env.DEBUG_MODE:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)-8s - %(name)s - %(message)s",
    )
    logging.getLogger("src.llm").setLevel(logging.DEBUG)
else:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)-8s - %(name)s - %(message)s",
    )

# Lazy singletons - initialized on first request to avoid blocking app startup
_vector_db: VectorDBAzureSearch | None = None
_assistant: KICampusAssistant | None = None


def get_vector_db() -> VectorDBAzureSearch:
    global _vector_db
    if _vector_db is None:
        _vector_db = VectorDBAzureSearch()
    return _vector_db


def get_assistant() -> KICampusAssistant:
    global _assistant
    if _assistant is None:
        _assistant = KICampusAssistant()
    return _assistant


app = FastAPI()
# authentication with OAuth2
api_key_header = APIKeyHeader(name="Api-Key")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ki-campus.staging.piipe.de",
        "https://ki-campus.org",
        "https://moodle.ki-campus.org",
        "https://ki-campus.loom.de",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def api_key_auth(api_key: Annotated[str, Depends(api_key_header)]):
    ALLOWED_API_KEYS = env.REST_API_KEYS

    if api_key not in ALLOWED_API_KEYS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")


# APIs
@app.get("/health")
def health() -> str:
    """Health check endpoint. Returns 'OK' if the service is running."""
    return "OK"

# Definert aber nicht genutzt!
class RetrievalRequest(BaseModel):
    message: str = Field(
        description="The query to find the most fitting sources to.",
        examples=["Which course is teaching Ethics?"],
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
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="module_id is required when course_id is set.",
            )
        return self


class ChatRequest(BaseModel):
    user_query: SerializableChatMessage = Field(
        description="The new user message. Should contain exactly one USER message.",
        examples=[
            SerializableChatMessage(content="I need help with my assignment", role=MessageRole.USER)
        ],
    )
    thread_id: str | None = Field(
        default=None,
        description="Thread ID for persistent conversations. If provided, chat history is loaded from backend checkpoint. If None, a new thread is created.",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
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
        default=Models.GEMMA4_31B,
        description="The LLM to use for the conversation.",
        examples=[Models.GEMMA4_31B, Models.AZURE_FALLBACK],
    )

    def get_user_query(self) -> str:
        """Extract the query string from user_query SerializableChatMessage."""
        return self.user_query.content

    @model_validator(mode="after")
    def validate_module_id(self):
        if self.module_id and not self.course_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="module_id is required when course_id is set.",
            )
        if self.module_id is not None:
            if not get_vector_db().check_if_module_exists(self.module_id):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"no module found with the given id: {self.module_id}.",
                )
        return self

    @model_validator(mode="after")
    def validate_course_id(self):
        if self.course_id is not None:
            if not get_vector_db().check_if_course_exists(self.course_id):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"no course found with the given id: {self.course_id}.",
                )
        return self


class ChatResponse(BaseModel):
    message: str = Field(
        description="The assistant response to the user message.",
        examples=["I can help you with that. What is the assignment about?"],
    )
    response_id: str = Field(description="An ID for the response, that is needed for using the feedback endpoint.")
    thread_id: str = Field(
        description="The thread ID. Use this for subsequent requests to continue the conversation.",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )


@app.post("/api/chat", dependencies=[Depends(api_key_auth)])
@observe()
def chat(chat_request: ChatRequest) -> ChatResponse:
    """Returns the response to the user message in one response (no streaming)."""
    # Use singleton assistant for performance
    
    if chat_request.course_id is not None:
        # Chat with course content (with or without module filter)
        llm_response, thread_id = get_assistant().chat_with_course(
            query=chat_request.get_user_query(),
            model=chat_request.model,
            course_id=chat_request.course_id,
            module_id=chat_request.module_id,  # Can be None
            thread_id=chat_request.thread_id,
        )
    else:
        # General chat (Drupal content)
        llm_response, thread_id = get_assistant().chat(
            query=chat_request.get_user_query(),
            model=chat_request.model,
            thread_id=chat_request.thread_id,
        )

    trace_id = langfuse_context.get_current_trace_id()
    if not trace_id:
        trace_id = "TRACING_UNAVAILABLE"
    
    # llm_response is SerializableChatMessage, extract content string
    chat_response = ChatResponse(
        message=llm_response.content,
        response_id=trace_id,
        thread_id=thread_id
    )
    return chat_response


@app.post("/api/chat/stream", dependencies=[Depends(api_key_auth)])
def chat_stream(chat_request: ChatRequest) -> StreamingResponse:
    """Stream the assistant response token-by-token as NDJSON.

    Response (application/x-ndjson):
      - {"type":"meta", "thread_id":..., "response_id":...}
      - {"type":"token", "token":...}  (many)
      - {"type":"final", "message":..., "thread_id":..., "response_id":...}
      - {"type":"error", "message":...} (optional)
    """

    # Ensure we control the thread_id that will be used for checkpointing.
    thread_id = chat_request.thread_id or str(uuid.uuid4())

    # Create a stable trace_id that will be used as Langfuse trace id.
    # We set it as "root trace id" before invoking @observe-decorated code.
    trace_id = str(uuid.uuid4())

    q: queue.Queue[object] = queue.Queue()
    DONE = object()

    def token_callback(token: str) -> None:
        q.put({"type": "token", "token": token})

    def worker() -> None:
        try:
            # Make all nested @observe spans use our predefined trace_id.
            # This is important because the streaming response needs the ID early.
            langfuse_context._set_root_trace_id(trace_id)

            with TokenCallbackContext(token_callback):
                if chat_request.course_id is not None:
                    llm_response, _thread_id = get_assistant().chat_with_course(
                        query=chat_request.get_user_query(),
                        model=chat_request.model,
                        course_id=chat_request.course_id,
                        module_id=chat_request.module_id,
                        thread_id=thread_id,
                    )
                else:
                    llm_response, _thread_id = get_assistant().chat(
                        query=chat_request.get_user_query(),
                        model=chat_request.model,
                        thread_id=thread_id,
                    )

            q.put(
                {
                    "type": "final",
                    "message": llm_response.content,
                    "response_id": trace_id,
                    "thread_id": _thread_id,
                }
            )
        except Exception as e:
            q.put({"type": "error", "message": str(e), "response_id": trace_id, "thread_id": thread_id})
        finally:
            q.put(DONE)

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        # Send metadata first so the client can store ids immediately.
        yield (json.dumps({"type": "meta", "thread_id": thread_id, "response_id": trace_id}) + "\n").encode(
            "utf-8"
        )
        while True:
            item = q.get()
            if item is DONE:
                break
            yield (json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8")

    return StreamingResponse(gen(), media_type="application/x-ndjson")


class ChatHistoryResponse(BaseModel):
    thread_id: str = Field(description="Die Thread-ID der Konversation.")
    messages: list[SerializableChatMessage] = Field(description="Alle gespeicherten Nachrichten der Konversation.")


@app.get("/api/chat/history/{thread_id}", dependencies=[Depends(api_key_auth)])
def get_chat_history(thread_id: str) -> ChatHistoryResponse:
    """Gibt den gespeicherten Gesprächsverlauf zurück (z.B. nach Page-Reload in Moodle).
    Gibt eine leere Liste zurück wenn die Session abgelaufen oder unbekannt ist."""
    messages = get_assistant().get_chat_history(thread_id)
    return ChatHistoryResponse(thread_id=thread_id, messages=messages)


class FeedbackRequest(BaseModel):
    response_id: str = Field(description="The ID of the response that the feedback belongs to.")
    feedback: str | None = Field(description="Feedback on the conversation.", default=None)
    score: int = Field(description="Score between 0 and 1, where 1 is good and 0 is bad.")

    @field_validator("score", mode="after")
    @classmethod
    def validate_score(cls, score: int) -> int:
        if not 0 <= score <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Score must be between 0 and 1.",
            )
        return score


@app.post("/api/feedback", dependencies=[Depends(api_key_auth)])
def track_feedback(feedback_request: FeedbackRequest) -> None:
    """Update feedback in langfuse logs."""
    Langfuse().score(
        trace_id=feedback_request.response_id,
        name="user-explicit-feedback",
        value=feedback_request.score,
        comment=feedback_request.feedback,
    )
