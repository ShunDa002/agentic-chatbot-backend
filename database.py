import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

# -------------------------------------------------------------------
# Database setup
# -------------------------------------------------------------------

Path("data").mkdir(exist_ok=True)

DATABASE_URL = "sqlite:///data/chatbot_memory.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 10},
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

Base = declarative_base()


def utc_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(UTC)


# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(
        String(64),
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    thread_id = Column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
    )

    title = Column(
        String(255),
        nullable=False,
        default="New Chat",
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(
        String(64),
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    thread_id = Column(
        String(64),
        ForeignKey(
            "conversations.thread_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    author = Column(
        String(16),
        nullable=False,
    )

    text = Column(
        Text,
        nullable=False,
        default="",
    )

    status = Column(
        String(32),
        nullable=False,
        default="completed",
    )

    # SQLite stores this as serialized JSON text.
    tool_calls = Column(
        Text,
        nullable=True,
    )

    interrupt_id = Column(
        String(64),
        nullable=True,
    )

    interrupt_decision = Column(
        String(8),
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )

    __table_args__ = (
        CheckConstraint(
            "author IN ('visitor', 'character')",
            name="ck_message_author",
        ),
        CheckConstraint(
            "status IN " "('completed', 'cancelled', 'interrupted', 'error')",
            name="ck_message_status",
        ),
        CheckConstraint(
            "interrupt_decision IS NULL " "OR interrupt_decision IN ('yes', 'no')",
            name="ck_interrupt_decision",
        ),
        Index(
            "ix_messages_thread_created",
            "thread_id",
            "created_at",
        ),
    )


class LongTermMemory(Base):
    __tablename__ = "long_term_memory"

    id = Column(
        String(64),
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    thread_id = Column(
        String(64),
        ForeignKey(
            "conversations.thread_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    memory = Column(
        Text,
        nullable=False,
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )


# -------------------------------------------------------------------
# Database initialization
# -------------------------------------------------------------------


def init_db() -> None:
    """Create tables that do not already exist."""
    Base.metadata.create_all(bind=engine)


# -------------------------------------------------------------------
# Serialization
# -------------------------------------------------------------------


def serialize_message(message: ConversationMessage) -> dict[str, Any]:
    """
    Convert a database message into the camelCase object expected by
    the frontend Message interface.
    """

    tool_calls = None

    if message.tool_calls:
        try:
            tool_calls = json.loads(message.tool_calls)
        except json.JSONDecodeError:
            tool_calls = None

    return {
        "id": str(message.id),
        "author": message.author,
        "text": message.text,
        "status": message.status,
        "toolCalls": tool_calls,
        "interruptId": message.interrupt_id,
        "interruptDecision": message.interrupt_decision,
        "createdAt": (message.created_at.isoformat() if message.created_at else None),
    }


def serialize_conversation(conversation: Conversation) -> dict[str, Any]:
    """Convert a conversation record into an API-friendly object."""

    return {
        "id": str(conversation.id),
        "threadId": conversation.thread_id,
        "title": conversation.title,
        "createdAt": (
            conversation.created_at.isoformat() if conversation.created_at else None
        ),
        "updatedAt": (
            conversation.updated_at.isoformat() if conversation.updated_at else None
        ),
    }


# -------------------------------------------------------------------
# Conversation operations
# -------------------------------------------------------------------


def create_or_update_conversation(
    thread_id: str,
    first_message: str | None = None,
) -> dict[str, Any]:
    """
    Create a conversation when it does not exist.

    If it already exists, update its last-activity timestamp.
    """

    db = SessionLocal()

    try:
        conversation = (
            db.query(Conversation).filter(Conversation.thread_id == thread_id).first()
        )

        now = utc_now()

        if conversation is None:
            title = "New Chat"

            if first_message and first_message.strip():
                cleaned_message = first_message.strip()
                title = cleaned_message[:40]

                if len(cleaned_message) > 40:
                    title += "..."

            conversation = Conversation(
                thread_id=thread_id,
                title=title,
                created_at=now,
                updated_at=now,
            )

            db.add(conversation)
        else:
            conversation.updated_at = now

        db.commit()
        db.refresh(conversation)

        return serialize_conversation(conversation)

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


def list_conversations() -> list[dict[str, Any]]:
    """Return conversations ordered by most recent activity."""

    db = SessionLocal()

    try:
        conversations = (
            db.query(Conversation).order_by(Conversation.updated_at.desc()).all()
        )

        return [serialize_conversation(conversation) for conversation in conversations]

    finally:
        db.close()


# -------------------------------------------------------------------
# Message operations
# -------------------------------------------------------------------


def save_chat_message(
    thread_id: str,
    author: str,
    text: str,
    status: str = "completed",
    tool_calls: list[dict[str, Any]] | None = None,
    interrupt_id: str | None = None,
    interrupt_decision: str | None = None,
    message_id: str | None = None,
) -> dict[str, Any]:
    """
    Save a message and return it in the frontend Message shape.

    The message ID can be generated by the frontend and passed into
    this function, or generated by the backend when omitted.
    """

    valid_authors = {"visitor", "character"}
    valid_statuses = {
        "completed",
        "cancelled",
        "interrupted",
        "error",
    }
    valid_decisions = {None, "yes", "no"}

    if author not in valid_authors:
        raise ValueError("author must be either 'visitor' or 'character'")

    if status not in valid_statuses:
        raise ValueError(
            "status must be 'completed', 'cancelled', " "'interrupted', or 'error'"
        )

    if interrupt_decision not in valid_decisions:
        raise ValueError("interrupt_decision must be 'yes', 'no', or None")

    if status == "interrupted" and not interrupt_id:
        raise ValueError("interrupt_id is required when status is 'interrupted'")

    db = SessionLocal()

    try:
        conversation = (
            db.query(Conversation).filter(Conversation.thread_id == thread_id).first()
        )

        if conversation is None:
            conversation = Conversation(
                thread_id=thread_id,
                title="New Chat",
            )
            db.add(conversation)
            db.flush()

        message = ConversationMessage(
            id=message_id or str(uuid4()),
            thread_id=thread_id,
            author=author,
            text=text,
            status=status,
            tool_calls=(
                json.dumps(tool_calls, ensure_ascii=False)
                if tool_calls is not None
                else None
            ),
            interrupt_id=interrupt_id,
            interrupt_decision=interrupt_decision,
            created_at=utc_now(),
        )

        db.add(message)

        conversation.updated_at = utc_now()

        db.commit()
        db.refresh(message)

        return serialize_message(message)

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


def get_chat_history(
    thread_id: str,
) -> list[dict[str, Any]]:
    """
    Return messages in chronological order and in the shape expected
    by <MessageLog messages={messages} />.
    """

    db = SessionLocal()

    try:
        messages = (
            db.query(ConversationMessage)
            .filter(ConversationMessage.thread_id == thread_id)
            .order_by(
                ConversationMessage.created_at.asc(),
                ConversationMessage.id.asc(),
            )
            .all()
        )

        return [serialize_message(message) for message in messages]

    finally:
        db.close()


def get_chat_message(
    message_id: str,
) -> dict[str, Any] | None:
    """Retrieve one message by its string ID."""

    db = SessionLocal()

    try:
        message = (
            db.query(ConversationMessage)
            .filter(ConversationMessage.id == message_id)
            .first()
        )

        if message is None:
            return None

        return serialize_message(message)

    finally:
        db.close()


def update_message_status(
    message_id: str,
    status: str,
    text: str | None = None,
) -> dict[str, Any] | None:
    """
    Update a message after streaming completes, fails, or is cancelled.
    """

    valid_statuses = {
        "completed",
        "cancelled",
        "interrupted",
        "error",
    }

    if status not in valid_statuses:
        raise ValueError(f"Invalid message status: {status}")

    db = SessionLocal()

    try:
        message = (
            db.query(ConversationMessage)
            .filter(ConversationMessage.id == message_id)
            .first()
        )

        if message is None:
            return None

        message.status = status

        if text is not None:
            message.text = text

        conversation = (
            db.query(Conversation)
            .filter(Conversation.thread_id == message.thread_id)
            .first()
        )

        if conversation:
            conversation.updated_at = utc_now()

        db.commit()
        db.refresh(message)

        return serialize_message(message)

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


def update_interrupt_decision(
    message_id: str,
    decision: str,
) -> dict[str, Any] | None:
    """Persist a user's HITL approval or rejection."""

    if decision not in {"yes", "no"}:
        raise ValueError("decision must be either 'yes' or 'no'")

    db = SessionLocal()

    try:
        message = (
            db.query(ConversationMessage)
            .filter(ConversationMessage.id == message_id)
            .first()
        )

        if message is None:
            return None

        if message.status != "interrupted":
            raise ValueError(
                "Only an interrupted message can receive " "an interrupt decision"
            )

        message.interrupt_decision = decision

        conversation = (
            db.query(Conversation)
            .filter(Conversation.thread_id == message.thread_id)
            .first()
        )

        if conversation:
            conversation.updated_at = utc_now()

        db.commit()
        db.refresh(message)

        return serialize_message(message)

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


# -------------------------------------------------------------------
# Long-term memory operations
# -------------------------------------------------------------------


def save_memory(thread_id: str, memory: str) -> str:
    """Save a long-term memory for a conversation."""

    if not memory.strip():
        raise ValueError("memory cannot be empty")

    db = SessionLocal()

    try:
        item = LongTermMemory(
            thread_id=thread_id,
            memory=memory.strip(),
            created_at=utc_now(),
        )

        db.add(item)
        db.commit()

        return "Memory saved successfully."

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


def search_memory(
    thread_id: str,
    query: str,
    limit: int = 20,
) -> str:
    """
    Perform a basic case-insensitive keyword search.

    This is not semantic or embedding-based retrieval.
    """

    db = SessionLocal()

    try:
        memory_query = db.query(LongTermMemory).filter(
            LongTermMemory.thread_id == thread_id
        )

        cleaned_query = query.strip()

        if cleaned_query:
            memory_query = memory_query.filter(
                LongTermMemory.memory.ilike(f"%{cleaned_query}%")
            )

        memories = (
            memory_query.order_by(LongTermMemory.created_at.desc()).limit(limit).all()
        )

        if not memories:
            return "No saved memory found."

        return "\n".join(f"- {memory.memory}" for memory in memories)

    finally:
        db.close()
