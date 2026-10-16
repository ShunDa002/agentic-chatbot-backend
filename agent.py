import os
from pathlib import Path
from dotenv import load_dotenv
import certifi
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from prompt import REM_SYSTEM_PROMPT
from tools import tools
from typing import Literal
import aiosqlite
import asyncio

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


Path("data").mkdir(exist_ok=True)

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "free-model")

GROQ_MODELS = {"gpt-oss-20b": "openai/gpt-oss-20b", "qwen3.8-27b": "qwen/qwen3.8-27b"}

OPENROUTER_MODELS = {
    "nemotron-3.5-lightning": "nvidia/nemotron-3.5-lightning:free",
    "gemma-4-31b-it": "google/gemma-4-31b-it:free",
    "free-model": "openrouter/free",
}

Provider = Literal["groq", "openrouter"]


def model_and_provider(model_map: str) -> tuple[str, Provider]:
    """
    Validate selected model from frontend.
    If model is missing or invalid, fallback to DEFAULT_MODEL.
    """
    model_name = model_map.strip().lower() if model_map else DEFAULT_MODEL

    if model_name in GROQ_MODELS:
        return (GROQ_MODELS[model_name], "groq")

    if model_name in OPENROUTER_MODELS:
        return (OPENROUTER_MODELS[model_name], "openrouter")

    # Invalid frontend selection: resolve the configured default model.
    if DEFAULT_MODEL in GROQ_MODELS:
        return GROQ_MODELS[DEFAULT_MODEL], "groq"

    if DEFAULT_MODEL in OPENROUTER_MODELS:
        return OPENROUTER_MODELS[DEFAULT_MODEL], "openrouter"

    raise ValueError(
        f"DEFAULT_MODEL {DEFAULT_MODEL!r} is not configured in any provider."
    )


async def build_agent(model_name: str, provider: Provider):
    """
    Build a LangGraph agent for the selected model.
    The SQLite connection remains open while the compiled agent is stored in the agent cache.
    """

    if provider == "groq":
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=model_name,
        )
    elif provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        llm = ChatOpenRouter(
            model=model_name,
            temperature=0.8,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider!r}")

    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: MessagesState):
        messages = [
            SystemMessage(content=REM_SYSTEM_PROMPT),
            *state["messages"],
        ]
        response = await llm_with_tools.ainvoke(messages)

        return {"messages": [response]}

    tool_node = ToolNode(tools)

    chatbot = StateGraph(MessagesState)

    chatbot.add_node("chat", chat_node)
    chatbot.add_node("tools", tool_node)

    chatbot.add_edge(START, "chat")
    chatbot.add_conditional_edges("chat", tools_condition)
    chatbot.add_edge("tools", "chat")

    conn = await aiosqlite.connect("data/langgraph_checkpoints.sqlite")

    checkpointer = AsyncSqliteSaver(conn)

    await checkpointer.setup()

    agent = chatbot.compile(
        checkpointer=checkpointer,
    )

    return agent, conn


# Build agent at the first time and save it in cache, preventing re-building the agent every time the chatbot is called.
_AGENT_CACHE: dict[tuple[Provider, str], object] = {}

_CONNECTION_CACHE: dict[
    tuple[Provider, str],
    aiosqlite.Connection,
] = {}

_AGENT_CACHE_LOCK = asyncio.Lock()


async def get_agent(model_name: str | None = None):
    """
    Return cached LangGraph agent for selected model.
    If not created yet, create it once and reuse it.
    """

    selected_model, provider = model_and_provider(model_name)

    cache_key = (
        provider,
        selected_model,
    )

    cached_agent = _AGENT_CACHE.get(cache_key)

    if cached_agent is not None:
        return cached_agent

    async with _AGENT_CACHE_LOCK:
        # Check again after acquiring the lock because another request
        # may have created the agent while this request was*waiting.
        cached_agent = _AGENT_CACHE.get(cache_key)

        if cached_agent is not None:
            return cached_agent

        agent, connection = await build_agent(
            model_name=selected_model,
            provider=provider,
        )

        _AGENT_CACHE[cache_key] = agent
        _CONNECTION_CACHE[cache_key] = connection

        return agent


async def close_agents() -> None:
    """
    Close all cached asynchronous SQLit* connections.

    Call this funct*on during FastAPI application shut*own.
    """
    async with _AGENT_CACHE_LOCK:
        connections = list(_CONNECTION_CACHE.values())

        _AGENT_CACHE.clear()
        _CONNECTION_CACHE.clear()

    for connection in connections:
        await connection.close()


def get_models():
    """
    Return a list of all available models name.
    """
    return list(GROQ_MODELS.keys()) + list(OPENROUTER_MODELS.keys())
