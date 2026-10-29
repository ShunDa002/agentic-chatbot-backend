from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
import requests, math
from contextvars import ContextVar
from langgraph.types import interrupt
from database import save_memory, search_memory
from rag import retrieve_from_rag

load_dotenv()
CURRENT_THREAD_ID: ContextVar[str | None] = ContextVar(
    "current_thread_id", default=None
)


def set_current_thread_id(thread_id: str):
    if not thread_id or not thread_id.strip():
        raise ValueError("thread_id must be non-empty")
    return CURRENT_THREAD_ID.set(thread_id.strip())


def reset_current_thread_id(token):
    CURRENT_THREAD_ID.reset(token)


def _require_thread_id() -> str:
    value = CURRENT_THREAD_ID.get()
    if not value:
        raise RuntimeError("No conversation thread is active for this tool call.")
    return value


web_search = TavilySearch(max_results=3, topic="general", search_depth="advanced")


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=HTM6MLWOXVJCK77J"
        r = requests.get(url)
        return r.json()
    except requests.RequestException as exc:
        return {"error": "Stock service unavailable.", "details": str(exc)}


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchased {quantity} shares of {symbol} was placed.",
            "symbol": symbol,
            "quantity": quantity,
        }
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression."""
    try:
        return str(
            eval(
                expression,
                {"__builtins__": {}},
                {
                    "math": math,
                    "abs": abs,
                    "round": round,
                    "min": min,
                    "max": max,
                    "sum": sum,
                },
            )
        )
    except Exception as exc:
        return f"Calculation error: {exc}"


@tool
def search_uploaded_documents(query: str) -> str:
    """Search files uploaded in the active thread. Prefer this over web search when files exist."""
    return retrieve_from_rag(query=query, thread_id=_require_thread_id())


@tool
def remember_this(memory: str) -> str:
    """Save an explicitly requested memory."""
    return save_memory(thread_id=_require_thread_id(), memory=memory)


@tool
def recall_memory(query: str) -> str:
    """Recall saved long-term memories."""
    return search_memory(thread_id=_require_thread_id(), query=query)


tools = [
    web_search,
    get_stock_price,
    purchase_stock,
    calculator,
    search_uploaded_documents,
    remember_this,
    recall_memory,
]
