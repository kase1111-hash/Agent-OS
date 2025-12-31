"""
Base Migration Classes

Provides the foundation for writing database migrations.
"""

import logging
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MigrationContext:
    """Context provided to migrations during execution."""

    data_dir: Path
    dry_run: bool = False
    verbose: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_db_path(self, name: str) -> Path:
        """Get the path to a database file."""
        return self.data_dir / f"{name}.db"

    def get_connection(self, name: str) -> sqlite3.Connection:
        """Get a connection to a database file."""
        db_path = self.get_db_path(name)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn


class Migration(ABC):
    """
    Base class for database migrations.

    Each migration should implement:
    - version: Unique version identifier (e.g., "0001", "0002")
    - name: Human-readable migration name
    - description: Detailed description of changes
    - upgrade(): Apply the migration
    - downgrade(): Revert the migration (optional)
    """

    # Override in subclasses
    version: str = ""
    name: str = ""
    description: str = ""

    # Which database stores this migration affects
    databases: List[str] = []

    # Dependencies on other migrations (by version)
    depends_on: List[str] = []

    def __init__(self, context: MigrationContext):
        self.context = context
        self._connections: Dict[str, sqlite3.Connection] = {}

    def get_connection(self, db_name: str) -> sqlite3.Connection:
        """Get or create a connection to the specified database."""
        if db_name not in self._connections:
            self._connections[db_name] = self.context.get_connection(db_name)
        return self._connections[db_name]

    def execute(self, db_name: str, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute SQL on the specified database."""
        conn = self.get_connection(db_name)
        if self.context.verbose:
            logger.debug(f"[{db_name}] {sql[:100]}...")
        if self.context.dry_run:
            logger.info(f"[DRY RUN] Would execute on {db_name}: {sql[:100]}...")
            return conn.cursor()
        return conn.execute(sql, params)

    def executemany(
        self, db_name: str, sql: str, params_list: List[tuple]
    ) -> sqlite3.Cursor:
        """Execute SQL for multiple parameter sets."""
        conn = self.get_connection(db_name)
        if self.context.dry_run:
            logger.info(
                f"[DRY RUN] Would execute {len(params_list)} times on {db_name}"
            )
            return conn.cursor()
        return conn.executemany(sql, params_list)

    def commit(self, db_name: str) -> None:
        """Commit changes to the specified database."""
        if db_name in self._connections and not self.context.dry_run:
            self._connections[db_name].commit()

    def commit_all(self) -> None:
        """Commit all database connections."""
        for db_name in self._connections:
            self.commit(db_name)

    def rollback(self, db_name: str) -> None:
        """Rollback changes to the specified database."""
        if db_name in self._connections:
            self._connections[db_name].rollback()

    def rollback_all(self) -> None:
        """Rollback all database connections."""
        for db_name in self._connections:
            self.rollback(db_name)

    def close_all(self) -> None:
        """Close all database connections."""
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()

    def table_exists(self, db_name: str, table_name: str) -> bool:
        """Check if a table exists in the database."""
        cursor = self.execute(
            db_name,
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return cursor.fetchone() is not None

    def column_exists(self, db_name: str, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table."""
        cursor = self.execute(db_name, f"PRAGMA table_info({table_name})")
        return any(row["name"] == column_name for row in cursor.fetchall())

    def get_table_columns(self, db_name: str, table_name: str) -> List[Dict[str, Any]]:
        """Get column information for a table."""
        cursor = self.execute(db_name, f"PRAGMA table_info({table_name})")
        return [dict(row) for row in cursor.fetchall()]

    @abstractmethod
    def upgrade(self) -> None:
        """
        Apply the migration.

        Raises:
            Exception: If migration fails
        """
        pass

    def downgrade(self) -> None:
        """
        Revert the migration.

        Override this method to provide rollback capability.
        By default, raises NotImplementedError.

        Raises:
            NotImplementedError: If downgrade is not supported
        """
        raise NotImplementedError(
            f"Migration {self.version} does not support downgrade"
        )

    def validate(self) -> bool:
        """
        Validate that the migration was applied correctly.

        Override to provide custom validation logic.

        Returns:
            True if validation passes
        """
        return True

    def __repr__(self) -> str:
        return f"<Migration {self.version}: {self.name}>"


@dataclass
class MigrationRecord:
    """Record of an applied migration."""

    version: str
    name: str
    applied_at: datetime
    checksum: str
    success: bool
    execution_time_ms: float
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "version": self.version,
            "name": self.name,
            "applied_at": self.applied_at.isoformat(),
            "checksum": self.checksum,
            "success": self.success,
            "execution_time_ms": self.execution_time_ms,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MigrationRecord":
        """Create from dictionary."""
        return cls(
            version=data["version"],
            name=data["name"],
            applied_at=datetime.fromisoformat(data["applied_at"]),
            checksum=data["checksum"],
            success=data["success"],
            execution_time_ms=data["execution_time_ms"],
            error_message=data.get("error_message"),
        )
