"""
Migration 0002: Add Conversation Metadata

Adds additional metadata columns to the conversations table
for improved search and organization.
"""

from datetime import datetime
from typing import List

from ..base import Migration


class AddConversationMetadataMigration(Migration):
    """Add metadata columns to conversations table."""

    version = "0002"
    name = "add_conversation_metadata"
    description = (
        "Adds last_active_at, message_count, and tags columns to the "
        "conversations table for improved organization and querying."
    )

    databases: List[str] = ["conversations"]
    depends_on: List[str] = ["0001"]

    def upgrade(self) -> None:
        """Add new columns to conversations table."""
        db_name = "conversations"
        db_path = self.context.get_db_path(db_name)

        if not db_path.exists():
            return

        # Check if conversations table exists
        if not self.table_exists(db_name, "conversations"):
            return

        # Add last_active_at column if not exists
        if not self.column_exists(db_name, "conversations", "last_active_at"):
            self.execute(
                db_name,
                "ALTER TABLE conversations ADD COLUMN last_active_at TEXT",
            )

            # Populate from existing messages
            self.execute(
                db_name,
                """
                UPDATE conversations
                SET last_active_at = (
                    SELECT MAX(timestamp)
                    FROM messages
                    WHERE messages.conversation_id = conversations.id
                )
            """,
            )

        # Add message_count column if not exists
        if not self.column_exists(db_name, "conversations", "message_count"):
            self.execute(
                db_name,
                "ALTER TABLE conversations ADD COLUMN message_count INTEGER DEFAULT 0",
            )

            # Populate from existing messages
            self.execute(
                db_name,
                """
                UPDATE conversations
                SET message_count = (
                    SELECT COUNT(*)
                    FROM messages
                    WHERE messages.conversation_id = conversations.id
                )
            """,
            )

        # Add tags column if not exists (JSON array)
        if not self.column_exists(db_name, "conversations", "tags"):
            self.execute(
                db_name,
                "ALTER TABLE conversations ADD COLUMN tags TEXT DEFAULT '[]'",
            )

        # Create index for last_active_at
        self.execute(
            db_name,
            """
            CREATE INDEX IF NOT EXISTS idx_conversations_last_active
            ON conversations(last_active_at)
        """,
        )

        # Record schema update
        self.execute(
            db_name,
            """
            INSERT INTO schema_version (version, migration_version, applied_at, description)
            VALUES (?, ?, ?, ?)
        """,
            (
                "1.1.0",
                self.version,
                datetime.utcnow().isoformat(),
                self.description,
            ),
        )

        self.commit(db_name)

    def downgrade(self) -> None:
        """
        Remove added columns.

        Note: SQLite doesn't support DROP COLUMN in older versions,
        so we recreate the table without the new columns.
        """
        db_name = "conversations"

        if not self.context.get_db_path(db_name).exists():
            return

        if not self.table_exists(db_name, "conversations"):
            return

        # SQLite 3.35+ supports ALTER TABLE DROP COLUMN
        # For older versions, we'd need to recreate the table

        # Drop index
        self.execute(
            db_name,
            "DROP INDEX IF EXISTS idx_conversations_last_active",
        )

        # Remove columns (SQLite 3.35+)
        for column in ["last_active_at", "message_count", "tags"]:
            if self.column_exists(db_name, "conversations", column):
                try:
                    self.execute(
                        db_name,
                        f"ALTER TABLE conversations DROP COLUMN {column}",
                    )
                except Exception:
                    # Older SQLite - would need table recreation
                    raise NotImplementedError(
                        "SQLite version doesn't support DROP COLUMN. "
                        "Manual table recreation required."
                    )

        # Remove schema version record
        self.execute(
            db_name,
            "DELETE FROM schema_version WHERE migration_version = ?",
            (self.version,),
        )

        self.commit(db_name)

    def validate(self) -> bool:
        """Verify columns were added correctly."""
        db_name = "conversations"

        if not self.context.get_db_path(db_name).exists():
            return True

        if not self.table_exists(db_name, "conversations"):
            return True

        required_columns = ["last_active_at", "message_count", "tags"]
        for column in required_columns:
            if not self.column_exists(db_name, "conversations", column):
                return False

        return True
