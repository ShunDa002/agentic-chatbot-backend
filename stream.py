import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Literal
from uuid import uuid4
from dataclasses import dataclass
from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, ToolMessage
from collections.abc import Iterator
from agent import get_agent
from database import create_or_update_conversation, save_chat_message
from tools import reset_current_thread_id, set_current_thread_id
from tools import reset_current_thread_id, set_current_thread_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

STREAM_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Content-Type-Options": "nosniff",
    "X-Accel-Buffering": "no",
}


@dataclass
class ActiveRun:
    run_id: str
    thread_id: str
    task: asyncio.Task[None]
    cancel_event: asyncio.Event


_active_runs: dict[str, ActiveRun] = {}
_active_runs_lock = asyncio.Lock()


@dataclass(frozen=True)
class StreamEnd:
    pass


STREAM_END = StreamEnd()
QueueItem = str | StreamEnd


def finalize_tool_calls(
    tool_calls_by_key: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []

    for stored_tool_call in tool_calls_by_key.values():
        tool_call = dict(stored_tool_call)

        arguments_text = tool_call.pop(
            "argumentsText",
            "",
        )

        if arguments_text:
            try:
                arguments = json.loads(arguments_text)
            except json.JSONDecodeError:
                arguments = {
                    "raw": arguments_text,
                }
        else:
            arguments = {}

        tool_call["arguments"] = arguments
        finalized.append(tool_call)

    return finalized


def attach_tool_result(
    tool_calls_by_key: dict[str, dict[str, Any]],
    event: dict[str, Any],
) -> None:
    key = get_tool_call_key(event)

    tool_call = tool_calls_by_key.setdefault(
        key,
        {
            "id": event.get("tool_call_id"),
            "name": event.get("tool_name"),
            "argumentsText": "",
            "result": None,
            "status": "running",
        },
    )

    if event.get("tool_call_id"):
        tool_call["id"] = event["tool_call_id"]

    if event.get("tool_name"):
        tool_call["name"] = event["tool_name"]

    tool_call["result"] = event.get("content")
    tool_call["status"] = event.get("status") or "completed"


def accumulate_tool_call_delta(
    tool_calls_by_key: dict[str, dict[str, Any]],
    event: dict[str, Any],
) -> None:
    key = get_tool_call_key(event)

    tool_call = tool_calls_by_key.setdefault(
        key,
        {
            "id": event.get("tool_call_id"),
            "name": event.get("tool_name"),
            "argumentsText": "",
            "result": None,
            "status": "running",
        },
    )

    if event.get("tool_call_id"):
        tool_call["id"] = event["tool_call_id"]

    if event.get("tool_name"):
        tool_call["name"] = event["tool_name"]

    tool_call["argumentsText"] += event.get("args_delta") or ""


def get_tool_call_key(event: dict[str, Any]) -> str:
    """
    Prefer the tool-call ID, falling back to its stream index.
    """
    tool_call_id = event.get("tool_call_id")

    if tool_call_id:
        return str(tool_call_id)

    tool_call_index = event.get("tool_call_index")

    return f"index:{tool_call_index}"


def parse_ndjson_event(event: str) -> dict[str, Any] | None:
    """
    Decode one NDJSON event for server-side accumulation.

    The original event string remains unchanged and can still be
    forwarded to the frontend.
    """
    try:
        parsed = json.loads(event.strip())
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "Unable to decode generated NDJSON event: %r",
            event,
        )
        return None

    if not isinstance(parsed, dict):
        logger.warning(
            "Expected an NDJSON object but received: %s",
            type(parsed).__name__,
        )
        return None

    return parsed


async def stop_active_run(
    *,
    thread_id: str,
    run_id: str,
) -> Literal["stopping", "already_stopped", "not_found"]:
    async with _active_runs_lock:
        active_run = _active_runs.get(run_id)

        if active_run is None:
            return "already_stopped"

        if active_run.thread_id != thread_id:
            return "not_found"

        active_run.cancel_event.set()
        active_run.task.cancel()

        return "stopping"


def to_ndjson(data: dict[str, Any]) -> str:
    return (
        json.dumps(
            jsonable_encoder(data),
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )
        + "\n"
    )


def extract_text(content: Any) -> str:
    """Normalize message content into text for the chatbot UI."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)

        return "".join(text_parts)

    return ""


def get_field(
    value: Any,
    field: str,
    default: Any = None,
) -> Any:
    """
    Read a field from either a dictionary-like value or an object.

    Tool-call chunks may behave as dictionaries or typed objects,
    depending on the installed LangChain version and provider.
    """
    # Check if the value is a dictionary
    if isinstance(value, dict):
        return value.get(field, default)
    # Otherwise, treat it as an object and use getattr
    return getattr(value, field, default)


def serialize_interrupt(
    interrupt_value: Any,
    thread_id: str,
) -> dict[str, Any]:
    """
    Convert a LangGraph Interrupt object into a safe NDJSON event.
    """
    return {
        "type": "interrupt",
        "thread_id": thread_id,
        "interrupt_id": get_field(interrupt_value, "id"),
        "value": get_field(interrupt_value, "value"),
        "resumable": True,
    }


def process_message_chunk(
    *,
    data: Any,
    thread_id: str,
) -> Iterator[str]:
    """
    Convert one LangGraph `messages` stream part into zero or more
    frontend NDJSON events.
    """
    if not isinstance(data, tuple) or len(data) != 2:
        logger.warning(
            "Unexpected messages stream data: " "thread_id=%s data_type=%s",
            thread_id,
            type(data).__name__,
        )
        return

    token, metadata = data
    metadata = metadata or {}

    node = metadata.get("langgraph_node")

    if isinstance(token, AIMessageChunk):
        text = extract_text(token.content)

        if text:
            yield to_ndjson(
                {
                    "type": "token",
                    "content": text,
                    "node": node,
                }
            )

        for tool_chunk in token.tool_call_chunks or []:
            yield to_ndjson(
                {
                    "type": "tool_call_delta",
                    "tool_name": get_field(
                        tool_chunk,
                        "name",
                    ),
                    "tool_call_id": get_field(
                        tool_chunk,
                        "id",
                    ),
                    "tool_call_index": get_field(
                        tool_chunk,
                        "index",
                    ),
                    "args_delta": (
                        get_field(
                            tool_chunk,
                            "args",
                            "",
                        )
                        or ""
                    ),
                    "node": node,
                }
            )

    elif isinstance(token, ToolMessage):
        yield to_ndjson(
            {
                "type": "tool_result",
                "tool_name": token.name,
                "tool_call_id": token.tool_call_id,
                "content": token.content,
                "status": getattr(
                    token,
                    "status",
                    None,
                ),
                "node": node,
            }
        )


def process_update_chunk(
    *,
    data: Any,
    thread_id: str,
) -> tuple[list[str], bool]:
    """
    Convert one LangGraph `updates` stream part into NDJSON events.

    Returns:
        A tuple containing:
        - Serialized NDJSON interrupt events
        - Whether an interrupt was detected
    """
    if not isinstance(data, dict):
        logger.warning(
            "Unexpected updates stream data: " "thread_id=%s data_type=%s",
            thread_id,
            type(data).__name__,
        )
        return [], False

    interrupts = data.get("__interrupt__")

    if not interrupts:
        # Normal node updates are intentionally ignored because
        # assistant text, tool calls, and tool results are already
        # emitted through stream_mode="messages".
        return [], False

    events = [
        to_ndjson(
            serialize_interrupt(
                interrupt_value,
                thread_id,
            )
        )
        for interrupt_value in interrupts
    ]

    return events, True


def create_chat_config(
    *,
    thread_id: str,
    operation: str,
    interrupt_id: str | None = None,
) -> dict[str, Any]:
    """
    Build the LangGraph configuration for an initial invocation
    or an interrupted workflow resume.
    """
    metadata: dict[str, Any] = {
        "thread_id": thread_id,
        "operation": operation,
    }

    if interrupt_id is not None:
        metadata["interrupt_id"] = interrupt_id

    return {
        "configurable": {
            "thread_id": thread_id,
        },
        "metadata": metadata,
        "run_name": ("chat_trace" if operation == "invoke" else "chat_resume_trace"),
    }


async def produce_chat_events(
    *,
    graph_input: Any,
    config: dict[str, Any],
    thread_id: str,
    run_id: str,
    operation: Literal["invoke", "resume"],
    cancel_event: asyncio.Event,
    queue: asyncio.Queue[QueueItem],
    error_message: str,
    model: str | None,
) -> None:
    workflow_interrupted = False
    character_message_saved = False
    character_message_id = str(uuid4())
    character_text_parts: list[str] = []
    tool_calls_by_key: dict[str, dict[str, Any]] = {}
    interrupt_id: str | None = None
    stream_started = False
    thread_context_token = None

    try:
        await queue.put(
            to_ndjson(
                {
                    "type": "start",
                    "thread_id": thread_id,
                    "run_id": run_id,
                    "operation": operation,
                }
            )
        )

        logger.info(
            "Creating agent: thread_id=%s run_id=%s model=%s",
            thread_id,
            run_id,
            model,
        )

        thread_context_token = set_current_thread_id(thread_id)
        agent = await get_agent(model_name=model)

        if operation == "invoke":
            visitor_message = graph_input["messages"][0]
            visitor_text = extract_text(visitor_message.content)

            create_or_update_conversation(
                thread_id,
                visitor_text,
            )

            await asyncio.to_thread(
                save_chat_message,
                thread_id=thread_id,
                author="visitor",
                text=visitor_text,
                status="completed",
            )
        else:
            create_or_update_conversation(thread_id)

        logger.info(
            "Agent created successfully: thread_id=%s run_id=%s agent_type=%s",
            thread_id,
            run_id,
            type(agent).__name__,
        )

        logger.info(
            "Starting agent stream: thread_id=%s run_id=%s",
            thread_id,
            run_id,
        )

        stream_started = True

        async for chunk in agent.astream(
            graph_input,
            config=config,
            stream_mode=["messages", "updates"],
            version="v2",
        ):
            # 合作式停止检查
            if cancel_event.is_set():
                raise asyncio.CancelledError

            chunk_type = chunk.get("type")
            data = chunk.get("data")

            if chunk_type == "messages":
                for event in process_message_chunk(
                    data=data,
                    thread_id=thread_id,
                ):
                    if cancel_event.is_set():
                        raise asyncio.CancelledError

                    # Send the original NDJSON string to the frontend unchanged.
                    await queue.put(event)

                    # Decode a copy for database accumulation.
                    parsed_event = parse_ndjson_event(event)

                    if parsed_event is None:
                        continue

                    event_type = parsed_event.get("type")

                    if event_type == "token":
                        content = parsed_event.get("content")

                        if isinstance(content, str) and content:
                            character_text_parts.append(content)

                    elif event_type == "tool_call_delta":
                        accumulate_tool_call_delta(
                            tool_calls_by_key,
                            parsed_event,
                        )

                    elif event_type == "tool_result":
                        attach_tool_result(
                            tool_calls_by_key,
                            parsed_event,
                        )

            elif chunk_type == "updates":
                interrupt_events, was_interrupted = process_update_chunk(
                    data=data,
                    thread_id=thread_id,
                )

                if was_interrupted:
                    workflow_interrupted = True

                for event in interrupt_events:
                    # Stream the original interrupt event unchanged.
                    await queue.put(event)

                    # Decode a copy to obtain the interrupt ID.
                    parsed_event = parse_ndjson_event(event)

                    if parsed_event is None:
                        continue

                    candidate_interrupt_id = (
                        parsed_event.get("interrupt_id")
                        or parsed_event.get("interruptId")
                        or parsed_event.get("id")
                    )

                    if candidate_interrupt_id:
                        interrupt_id = str(candidate_interrupt_id)

        character_text = "".join(character_text_parts)

        tool_calls = finalize_tool_calls(tool_calls_by_key)

        if not workflow_interrupted:
            await asyncio.to_thread(
                save_chat_message,
                message_id=character_message_id,
                thread_id=thread_id,
                author="character",
                text=character_text,
                status="completed",
                tool_calls=tool_calls or None,
            )

            character_message_saved = True

            await queue.put(
                to_ndjson(
                    {
                        "type": "done",
                        "thread_id": thread_id,
                        "run_id": run_id,
                        "operation": operation,
                    }
                )
            )

        else:
            if not interrupt_id:
                raise RuntimeError(
                    "Workflow was interrupted, but the interrupt "
                    "event did not contain an interrupt ID."
                )

            await asyncio.to_thread(
                save_chat_message,
                message_id=character_message_id,
                thread_id=thread_id,
                author="character",
                text=character_text,
                status="interrupted",
                tool_calls=tool_calls or None,
                interrupt_id=interrupt_id,
            )

            character_message_saved = True

    except asyncio.CancelledError:
        logger.info(
            "Chat run cancelled: " "thread_id=%s run_id=%s operation=%s",
            thread_id,
            run_id,
            operation,
        )

        partial_text = "".join(character_text_parts)

        tool_calls = finalize_tool_calls(tool_calls_by_key)

        if not character_message_saved and (partial_text or tool_calls):
            try:
                await asyncio.to_thread(
                    save_chat_message,
                    message_id=character_message_id,
                    thread_id=thread_id,
                    author="character",
                    text=partial_text,
                    status="cancelled",
                    tool_calls=tool_calls or None,
                )

                character_message_saved = True

            except Exception:
                logger.exception(
                    "Failed to save cancelled character message: "
                    "thread_id=%s run_id=%s",
                    thread_id,
                    run_id,
                )

        with contextlib.suppress(Exception):
            await queue.put(
                to_ndjson(
                    {
                        "type": "stopped",
                        "thread_id": thread_id,
                        "run_id": run_id,
                        "operation": operation,
                    }
                )
            )

        raise

    except Exception:
        logger.exception(
            "Chat workflow failed: " "thread_id=%s run_id=%s operation=%s",
            thread_id,
            run_id,
            operation,
        )

        partial_text = "".join(character_text_parts)

        tool_calls = finalize_tool_calls(tool_calls_by_key)

        if not character_message_saved and (
            stream_started or character_text_parts or tool_calls_by_key
        ):
            try:
                await asyncio.to_thread(
                    save_chat_message,
                    message_id=character_message_id,
                    thread_id=thread_id,
                    author="character",
                    text=partial_text or error_message,
                    status="error",
                    tool_calls=tool_calls or None,
                )

                character_message_saved = True

            except Exception:
                logger.exception(
                    "Failed to save errored character message: "
                    "thread_id=%s run_id=%s",
                    thread_id,
                    run_id,
                )

        await queue.put(
            to_ndjson(
                {
                    "type": "error",
                    "thread_id": thread_id,
                    "run_id": run_id,
                    "operation": operation,
                    "message": error_message,
                }
            )
        )

    finally:
        if thread_context_token is not None:
            reset_current_thread_id(thread_context_token)
        await queue.put(STREAM_END)

        async with _active_runs_lock:
            current = _active_runs.get(run_id)

            # 防止误删已经被替换的新任务
            if current is not None and current.task is asyncio.current_task():
                _active_runs.pop(run_id, None)


async def consume_chat_events(
    *,
    queue: asyncio.Queue[QueueItem],
    task: asyncio.Task[None],
    request: Request,
    cancel_event: asyncio.Event,
) -> AsyncIterator[str]:
    try:
        while True:
            if await request.is_disconnected():
                cancel_event.set()
                task.cancel()
                break

            try:
                item = await asyncio.wait_for(
                    queue.get(),
                    timeout=0.5,
                )
            except TimeoutError:
                continue

            if item is STREAM_END:
                break

            yield item

    except asyncio.CancelledError:
        cancel_event.set()
        task.cancel()
        raise

    finally:
        # 浏览器关闭、切换页面或 AbortController.abort()
        # 都会进入清理流程。
        if not task.done():
            cancel_event.set()
            task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task


async def create_chat_streaming_response(
    *,
    request: Request,
    graph_input: Any,
    thread_id: str,
    operation: Literal["invoke", "resume"],
    interrupt_id: str | None = None,
    error_message: str,
    model: str | None = None,
) -> StreamingResponse:
    run_id = str(uuid4())
    queue: asyncio.Queue[QueueItem] = asyncio.Queue(maxsize=100)
    cancel_event = asyncio.Event()

    config = create_chat_config(
        thread_id=thread_id,
        operation=operation,
        interrupt_id=interrupt_id,
    )

    # 将 run_id 写入 tracing metadata。
    config.setdefault("metadata", {})["run_id"] = run_id

    task = asyncio.create_task(
        produce_chat_events(
            graph_input=graph_input,
            config=config,
            thread_id=thread_id,
            run_id=run_id,
            operation=operation,
            cancel_event=cancel_event,
            queue=queue,
            error_message=error_message,
            model=model,
        ),
        name=f"chat-run:{run_id}",
    )

    async with _active_runs_lock:
        _active_runs[run_id] = ActiveRun(
            run_id=run_id,
            thread_id=thread_id,
            task=task,
            cancel_event=cancel_event,
        )

    return StreamingResponse(
        consume_chat_events(
            queue=queue,
            task=task,
            request=request,
            cancel_event=cancel_event,
        ),
        media_type="application/x-ndjson",
        headers={
            **STREAM_HEADERS,
            "X-Chat-Run-Id": run_id,
        },
    )
