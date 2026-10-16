from pydantic import BaseModel, Field
from typing import Any
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from agent import close_agents
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from stream import create_chat_streaming_response, stop_active_run
from langgraph.types import Command
import uuid
from pathlib import Path
from agent import get_models
from database import init_db, list_conversations, get_chat_history


class ChatRequest(BaseModel):
    thread_id: str = Field(description="Unique thread ID for storage services")
    user_input: str = Field(min_length=1, max_length=300, description="User's message")
    model: str = Field(description="Model name to use for the chatbot")


class ChatResumeRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    resume: Any
    interrupt_id: str | None = None
    model: str = Field(description="Model name to use for the chatbot")


class ChatStopRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        await close_agents()


app = FastAPI(
    lifespan=lifespan,
)

Path("uploads").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

init_db()

origins = [
    # "http://localhost:3000",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/conversations")
async def conversations():
    return list_conversations()


@app.get("/history/{thread_id}")
async def history(thread_id: str):
    return get_chat_history(thread_id)


# Generate a unique thread ID for each new conversation
@app.post(
    "/threads",
    tags=["threads"],
    summary="Create a new thread ID for a conversation",
)
def thread_id():
    return str(uuid.uuid4())


@app.post("/models", tags=["models"], summary="Get list of available models")
def models():
    return get_models()


@app.post(
    "/chat",
    tags=["chat"],
    summary="Chat with the agentic chatbot",
)
async def chat_with_thread(
    request_body: ChatRequest,
    request: Request,
) -> StreamingResponse:
    return await create_chat_streaming_response(
        request=request,
        graph_input={
            "messages": [
                HumanMessage(
                    content=request_body.user_input,
                )
            ]
        },
        thread_id=request_body.thread_id,
        operation="invoke",
        error_message="The chatbot response stream failed.",
        model=request_body.model,
    )


@app.post(
    "/chat/resume",
    tags=["chat"],
    summary="Resume an interrupted chatbot workflow",
)
async def resume_chat_with_thread(
    request_body: ChatResumeRequest,
    request: Request,
) -> StreamingResponse:
    return await create_chat_streaming_response(
        request=request,
        graph_input=Command(
            resume=request_body.resume,
        ),
        thread_id=request_body.thread_id,
        operation="resume",
        interrupt_id=request_body.interrupt_id,
        error_message="The chatbot workflow could not be resumed.",
        model=request_body.model,
    )


@app.post(
    "/chat/stop",
    tags=["chat"],
    summary="Stop an active chatbot run",
)
async def stop_chat_run(
    request: ChatStopRequest,
) -> dict[str, Any]:
    status = await stop_active_run(
        thread_id=request.thread_id,
        run_id=request.run_id,
    )

    if status == "not_found":
        raise HTTPException(
            status_code=404,
            detail="Active run not found.",
        )

    return {
        "status": status,
        "thread_id": request.thread_id,
        "run_id": request.run_id,
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True, log_level="debug")
