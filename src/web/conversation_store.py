"""
Conversation Store for Agent OS Chat

Provides persistent storage for chat conversations using SQLite.
Integrates with Seshat memory system for semantic search capabilities.

Features:
- SQLite-based persistence (survives restarts)
- Conversation metadata and search
- Message threading and replies
- Export/import for backup
- Optional Seshat integration for semantic memory
"""

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    """Message role in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class StoredMessage:
    """A persistable chat message."""
    id: str
    conversation_id: str
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_message_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "parent_message_id": self.parent_message_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredMessage":
        return cls(
            id=data["id"],
            conversation_id=data["conversation_id"],
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            parent_message_id=data.get("parent_message_id"),
        )


@dataclass
class Conversation:
    """A chat conversation with metadata."""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    archived: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": self.message_count,
            "metadata": self.metadata,
            "archived": self.archived,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        return cls(
            id=data["id"],
            title=data["title"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            message_count=data.get("message_count", 0),
            metadata=data.get("metadata", {}),
            archived=data.get("archived", False),
        )


class ConversationStore:
    """
    Persistent storage for chat conversations.

    Uses SQLite for reliable, local storage that survives restarts.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        auto_title: bool = True,
    ):
        """
        Initialize the conversation store.

        Args:
            db_path: Path to SQLite database. None for default location.
            auto_title: Automatically generate titles from first message.
        """
        if db_path is None:
            # Default to data directory
            data_dir = Path.home() / ".agent-os" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = data_dir / "conversations.db"

        self.db_path = db_path
        self.auto_title = auto_title
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the store and create tables."""
        try:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._conn.row_factory = sqlite3.Row
            self._create_tables()
            self._initialized = True
            logger.info(f"Conversation store initialized at {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize conversation store: {e}")
            return False

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
        self._initialized = False

    def _create_tables(self) -> None:
        """Create database tables."""
        with self._lock:
            cursor = self._conn.cursor()

            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    message_count INTEGER DEFAULT 0,
                    metadata_json TEXT,
                    archived INTEGER DEFAULT 0
                )
            """)

            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata_json TEXT,
                    parent_message_id TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                        ON DELETE CASCADE
                )
            """)

            # Indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp
                ON messages(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_updated
                ON conversations(updated_at DESC)
            """)

            # Full-text search for messages (optional)
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    content,
                    content=messages,
                    content_rowid=rowid
                )
            """)

            self._conn.commit()

    # =========================================================================
    # Conversation Operations
    # =========================================================================

    def create_conversation(
        self,
        conversation_id: Optional[str] = None,
        title: str = "New Conversation",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Conversation:
        """Create a new conversation."""
        now = datetime.utcnow()
        conv = Conversation(
            id=conversation_id or str(uuid.uuid4()),
            title=title,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )

        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                INSERT INTO conversations (id, title, created_at, updated_at, message_count, metadata_json, archived)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                conv.id,
                conv.title,
                conv.created_at.isoformat(),
                conv.updated_at.isoformat(),
                0,
                json.dumps(conv.metadata),
                0,
            ))
            self._conn.commit()

        logger.debug(f"Created conversation: {conv.id}")
        return conv

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            return self._row_to_conversation(row)

    def get_or_create_conversation(
        self,
        conversation_id: str,
        title: str = "New Conversation",
    ) -> Conversation:
        """Get existing conversation or create new one."""
        conv = self.get_conversation(conversation_id)
        if conv:
            return conv
        return self.create_conversation(conversation_id, title)

    def update_conversation(
        self,
        conversation_id: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        archived: Optional[bool] = None,
    ) -> bool:
        """Update conversation metadata."""
        updates = []
        params = []

        if title is not None:
            updates.append("title = ?")
            params.append(title)
        if metadata is not None:
            updates.append("metadata_json = ?")
            params.append(json.dumps(metadata))
        if archived is not None:
            updates.append("archived = ?")
            params.append(1 if archived else 0)

        if not updates:
            return False

        updates.append("updated_at = ?")
        params.append(datetime.utcnow().isoformat())
        params.append(conversation_id)

        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                f"UPDATE conversations SET {', '.join(updates)} WHERE id = ?",
                params
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            self._conn.commit()
            deleted = cursor.rowcount > 0

        if deleted:
            logger.info(f"Deleted conversation: {conversation_id}")
        return deleted

    def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
        include_archived: bool = False,
        search: Optional[str] = None,
    ) -> List[Conversation]:
        """List conversations with optional filtering."""
        with self._lock:
            cursor = self._conn.cursor()

            query = "SELECT * FROM conversations"
            params = []
            conditions = []

            if not include_archived:
                conditions.append("archived = 0")

            if search:
                conditions.append("title LIKE ?")
                params.append(f"%{search}%")

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            return [self._row_to_conversation(row) for row in cursor.fetchall()]

    # =========================================================================
    # Message Operations
    # =========================================================================

    def add_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_message_id: Optional[str] = None,
    ) -> StoredMessage:
        """Add a message to a conversation."""
        # Ensure conversation exists
        conv = self.get_or_create_conversation(conversation_id)

        msg = StoredMessage(
            id=message_id or str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
            parent_message_id=parent_message_id,
        )

        with self._lock:
            cursor = self._conn.cursor()

            # Insert message
            cursor.execute("""
                INSERT INTO messages (id, conversation_id, role, content, timestamp, metadata_json, parent_message_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                msg.id,
                msg.conversation_id,
                msg.role.value,
                msg.content,
                msg.timestamp.isoformat(),
                json.dumps(msg.metadata),
                msg.parent_message_id,
            ))

            # Update conversation
            new_count = conv.message_count + 1
            cursor.execute("""
                UPDATE conversations
                SET updated_at = ?, message_count = ?
                WHERE id = ?
            """, (
                msg.timestamp.isoformat(),
                new_count,
                conversation_id,
            ))

            # Auto-title from first user message
            if self.auto_title and new_count == 1 and role == MessageRole.USER:
                title = content[:50] + ("..." if len(content) > 50 else "")
                cursor.execute(
                    "UPDATE conversations SET title = ? WHERE id = ?",
                    (title, conversation_id)
                )

            # Update FTS index
            cursor.execute("""
                INSERT INTO messages_fts(rowid, content)
                SELECT rowid, content FROM messages WHERE id = ?
            """, (msg.id,))

            self._conn.commit()

        return msg

    def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
    ) -> List[StoredMessage]:
        """Get messages from a conversation."""
        with self._lock:
            cursor = self._conn.cursor()

            query = "SELECT * FROM messages WHERE conversation_id = ?"
            params: List[Any] = [conversation_id]

            if before:
                query += " AND timestamp < ?"
                params.append(before.isoformat())
            if after:
                query += " AND timestamp > ?"
                params.append(after.isoformat())

            query += " ORDER BY timestamp ASC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            return [self._row_to_message(row) for row in cursor.fetchall()]

    def get_recent_messages(
        self,
        conversation_id: str,
        count: int = 10,
    ) -> List[StoredMessage]:
        """Get the most recent messages from a conversation."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                SELECT * FROM (
                    SELECT * FROM messages
                    WHERE conversation_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ) ORDER BY timestamp ASC
            """, (conversation_id, count))
            return [self._row_to_message(row) for row in cursor.fetchall()]

    def search_messages(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[StoredMessage]:
        """Search messages using full-text search."""
        with self._lock:
            cursor = self._conn.cursor()

            if conversation_id:
                cursor.execute("""
                    SELECT m.* FROM messages m
                    JOIN messages_fts fts ON m.rowid = fts.rowid
                    WHERE messages_fts MATCH ? AND m.conversation_id = ?
                    ORDER BY rank
                    LIMIT ?
                """, (query, conversation_id, limit))
            else:
                cursor.execute("""
                    SELECT m.* FROM messages m
                    JOIN messages_fts fts ON m.rowid = fts.rowid
                    WHERE messages_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (query, limit))

            return [self._row_to_message(row) for row in cursor.fetchall()]

    def delete_message(self, message_id: str) -> bool:
        """Delete a specific message."""
        with self._lock:
            cursor = self._conn.cursor()

            # Get conversation_id first
            cursor.execute(
                "SELECT conversation_id FROM messages WHERE id = ?",
                (message_id,)
            )
            row = cursor.fetchone()
            if not row:
                return False

            conversation_id = row[0]

            # Delete message
            cursor.execute("DELETE FROM messages WHERE id = ?", (message_id,))

            # Update conversation count
            cursor.execute("""
                UPDATE conversations
                SET message_count = message_count - 1,
                    updated_at = ?
                WHERE id = ?
            """, (datetime.utcnow().isoformat(), conversation_id))

            self._conn.commit()
            return True

    # =========================================================================
    # Export / Import
    # =========================================================================

    def export_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Export a conversation with all messages."""
        conv = self.get_conversation(conversation_id)
        if not conv:
            return None

        messages = self.get_messages(conversation_id)

        return {
            "conversation": conv.to_dict(),
            "messages": [m.to_dict() for m in messages],
            "exported_at": datetime.utcnow().isoformat(),
            "version": "1.0",
        }

    def import_conversation(
        self,
        data: Dict[str, Any],
        overwrite: bool = False,
    ) -> Optional[str]:
        """Import a conversation from exported data."""
        try:
            conv_data = data["conversation"]
            messages_data = data["messages"]

            conversation_id = conv_data["id"]

            # Check if exists
            existing = self.get_conversation(conversation_id)
            if existing and not overwrite:
                logger.warning(f"Conversation {conversation_id} already exists")
                return None

            if existing:
                self.delete_conversation(conversation_id)

            # Create conversation
            conv = Conversation.from_dict(conv_data)
            with self._lock:
                cursor = self._conn.cursor()
                cursor.execute("""
                    INSERT INTO conversations (id, title, created_at, updated_at, message_count, metadata_json, archived)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    conv.id, conv.title,
                    conv.created_at.isoformat(),
                    conv.updated_at.isoformat(),
                    0, json.dumps(conv.metadata),
                    1 if conv.archived else 0,
                ))
                self._conn.commit()

            # Import messages
            for msg_data in messages_data:
                msg = StoredMessage.from_dict(msg_data)
                self.add_message(
                    conversation_id=msg.conversation_id,
                    role=msg.role,
                    content=msg.content,
                    message_id=msg.id,
                    metadata=msg.metadata,
                    parent_message_id=msg.parent_message_id,
                )

            logger.info(f"Imported conversation: {conversation_id}")
            return conversation_id

        except Exception as e:
            logger.error(f"Failed to import conversation: {e}")
            return None

    def export_all(self) -> Dict[str, Any]:
        """Export all conversations."""
        conversations = self.list_conversations(limit=10000, include_archived=True)
        exported = []

        for conv in conversations:
            export = self.export_conversation(conv.id)
            if export:
                exported.append(export)

        return {
            "conversations": exported,
            "exported_at": datetime.utcnow().isoformat(),
            "version": "1.0",
            "count": len(exported),
        }

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics."""
        with self._lock:
            cursor = self._conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM conversations WHERE archived = 0")
            active_convs = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM conversations WHERE archived = 1")
            archived_convs = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM messages")
            total_messages = cursor.fetchone()[0]

            cursor.execute("""
                SELECT MAX(updated_at) FROM conversations
            """)
            last_activity = cursor.fetchone()[0]

            return {
                "active_conversations": active_convs,
                "archived_conversations": archived_convs,
                "total_messages": total_messages,
                "last_activity": last_activity,
                "database_path": str(self.db_path),
            }

    # =========================================================================
    # Helpers
    # =========================================================================

    def _row_to_conversation(self, row: sqlite3.Row) -> Conversation:
        """Convert a database row to Conversation."""
        return Conversation(
            id=row["id"],
            title=row["title"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            message_count=row["message_count"],
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
            archived=bool(row["archived"]),
        )

    def _row_to_message(self, row: sqlite3.Row) -> StoredMessage:
        """Convert a database row to StoredMessage."""
        return StoredMessage(
            id=row["id"],
            conversation_id=row["conversation_id"],
            role=MessageRole(row["role"]),
            content=row["content"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
            parent_message_id=row["parent_message_id"],
        )


# =============================================================================
# Global Store Instance
# =============================================================================

_store: Optional[ConversationStore] = None


def get_conversation_store(
    db_path: Optional[Path] = None,
    initialize: bool = True,
) -> ConversationStore:
    """
    Get or create the global conversation store.

    Args:
        db_path: Optional custom database path
        initialize: Whether to initialize if not already done

    Returns:
        ConversationStore instance
    """
    global _store

    if _store is None:
        _store = ConversationStore(db_path=db_path)
        if initialize:
            _store.initialize()

    return _store


def close_conversation_store() -> None:
    """Close the global conversation store."""
    global _store
    if _store:
        _store.close()
        _store = None
