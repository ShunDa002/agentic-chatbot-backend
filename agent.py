import os, asyncio, aiosqlite, certifi
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from prompt import REM_SYSTEM_PROMPT
from tools import tools
from rag import has_uploaded_documents, retrieve_from_rag

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


def model_and_provider(value: str) -> tuple[str, Provider]:
    name = value.strip().lower() if value else DEFAULT_MODEL
    if name in GROQ_MODELS:
        return GROQ_MODELS[name], "groq"
    if name in OPENROUTER_MODELS:
        return OPENROUTER_MODELS[name], "openrouter"
    if DEFAULT_MODEL in GROQ_MODELS:
        return GROQ_MODELS[DEFAULT_MODEL], "groq"
    if DEFAULT_MODEL in OPENROUTER_MODELS:
        return OPENROUTER_MODELS[DEFAULT_MODEL], "openrouter"
    raise ValueError(f"DEFAULT_MODEL {DEFAULT_MODEL!r} is not configured")


def _latest_human_query(messages) -> str | None:
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            return message.content
    return None


async def build_agent(model_name: str, provider: Provider):
    if provider == "groq":
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=model_name)
    else:
        from langchain_openrouter import ChatOpenRouter

        llm = ChatOpenRouter(model=model_name, temperature=0.3)
    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: MessagesState, config: RunnableConfig):
        thread_id = config.get("configurable", {}).get("thread_id")
        messages = list(state["messages"])
        query = _latest_human_query(messages)
        document_context = None
        # Deterministic file-first policy: every new user question in a thread with
        # uploaded files gets retrieval before the LLM can choose web search.
        if (
            isinstance(thread_id, str)
            and query
            and await asyncio.to_thread(has_uploaded_documents, thread_id)
        ):
            document_context = await asyncio.to_thread(
                retrieve_from_rag, query, thread_id, 6
            )
        systems = [SystemMessage(content=REM_SYSTEM_PROMPT)]
        if document_context:
            systems.append(
                SystemMessage(
                    content="UPLOADED DOCUMENT CONTEXT (primary source):\n\n"
                    + document_context
                    + "\n\nAnswer from this context when supported. If unsupported, do not automatically search the web; state the limitation unless the user explicitly requested web/current information."
                )
            )
        response = await llm_with_tools.ainvoke([*systems, *messages])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")
    conn = await aiosqlite.connect("data/langgraph_checkpoints.sqlite")
    saver = AsyncSqliteSaver(conn)
    await saver.setup()
    return graph.compile(checkpointer=saver), conn


_AGENT_CACHE = {}
_CONNECTION_CACHE = {}
_AGENT_CACHE_LOCK = asyncio.Lock()


async def get_agent(model_name: str | None = None):
    selected, provider = model_and_provider(model_name)
    key = (provider, selected)
    if key in _AGENT_CACHE:
        return _AGENT_CACHE[key]
    async with _AGENT_CACHE_LOCK:
        if key not in _AGENT_CACHE:
            agent, conn = await build_agent(selected, provider)
            _AGENT_CACHE[key] = agent
            _CONNECTION_CACHE[key] = conn
        return _AGENT_CACHE[key]


async def close_agents():
    async with _AGENT_CACHE_LOCK:
        conns = list(_CONNECTION_CACHE.values())
        _AGENT_CACHE.clear()
        _CONNECTION_CACHE.clear()
    for conn in conns:
        await conn.close()


def get_models():
    return list(GROQ_MODELS) + list(OPENROUTER_MODELS)
