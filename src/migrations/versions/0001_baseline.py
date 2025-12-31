"""
Migration 0001: Baseline

Establishes baseline schema for existing databases.
This migration creates schema_version tables in all databases
to track individual database schema versions going forward.
"""

from datetime import datetime
from typing import List

from ..base import Migration


class BaselineMigration(Migration):
    """Establish baseline schema versioning for all databases."""

    version = "0001"
    name = "baseline"
    description = (
        "Establishes baseline schema version tracking in all Agent OS databases. "
        "This is the foundation migration that all future migrations depend on."
    )

    databases: List[str] = [
        "conversations",
        "context",
        "rules",
        "audit",
        "intent_log",
    ]

    def upgrade(self) -> None:
        """Create schema_version tables in all databases."""
        for db_name in self.databases:
            db_path = self.context.get_db_path(db_name)

            # Skip if database doesn't exist yet
            if not db_path.exists():
                continue

            # Create schema_version table if not exists
            self.execute(
                db_name,
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    id INTEGER PRIMARY KEY,
                    version TEXT NOT NULL,
                    migration_version TEXT NOT NULL,
                    applied_at TEXT NOT NULL,
                    description TEXT
                )
            """,
            )

            # Record baseline version
            self.execute(
                db_name,
                """
                INSERT INTO schema_version (version, migration_version, applied_at, description)
                VALUES (?, ?, ?, ?)
            """,
                ("1.0.0", self.version, datetime.utcnow().isoformat(), "Baseline schema"),
            )

            self.commit(db_name)

    def downgrade(self) -> None:
        """Remove schema_version tables."""
        for db_name in self.databases:
            db_path = self.context.get_db_path(db_name)

            if not db_path.exists():
                continue

            if self.table_exists(db_name, "schema_version"):
                self.execute(db_name, "DROP TABLE schema_version")
                self.commit(db_name)

    def validate(self) -> bool:
        """Verify schema_version tables exist in all databases."""
        for db_name in self.databases:
            db_path = self.context.get_db_path(db_name)

            if not db_path.exists():
                continue

            if not self.table_exists(db_name, "schema_version"):
                return False

        return True
