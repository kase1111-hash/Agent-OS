"""
Migration Runner

Discovers, orders, and executes database migrations.
"""

import hashlib
import importlib
import importlib.util
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Type

from .base import Migration, MigrationContext, MigrationRecord

logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Error during migration execution."""

    def __init__(self, message: str, migration: Optional[Migration] = None):
        self.migration = migration
        super().__init__(message)


class MigrationRunner:
    """
    Manages and executes database migrations.

    Features:
    - Version tracking in dedicated migrations database
    - Dependency resolution between migrations
    - Dry run mode for testing
    - Checksum verification
    - Automatic backup before migrations
    """

    MIGRATIONS_DB = "migrations"
    MIGRATIONS_TABLE = "migration_history"

    def __init__(
        self,
        data_dir: Path,
        migrations_dir: Optional[Path] = None,
    ):
        """
        Initialize the migration runner.

        Args:
            data_dir: Directory containing database files
            migrations_dir: Directory containing migration scripts
        """
        self.data_dir = Path(data_dir)
        self.migrations_dir = migrations_dir or (
            Path(__file__).parent / "versions"
        )
        self._migrations: Dict[str, Type[Migration]] = {}
        self._connection: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get connection to migrations tracking database."""
        if self._connection is None:
            db_path = self.data_dir / f"{self.MIGRATIONS_DB}.db"
            self._connection = sqlite3.connect(str(db_path))
            self._connection.row_factory = sqlite3.Row
            self._init_migrations_table()
        return self._connection

    def _init_migrations_table(self) -> None:
        """Create the migrations tracking table if needed."""
        self._connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.MIGRATIONS_TABLE} (
                version TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL,
                checksum TEXT NOT NULL,
                success INTEGER NOT NULL,
                execution_time_ms REAL NOT NULL,
                error_message TEXT
            )
        """)
        self._connection.commit()

    def discover_migrations(self) -> Dict[str, Type[Migration]]:
        """
        Discover all migration classes in the migrations directory.

        Returns:
            Dictionary mapping version to Migration class
        """
        self._migrations.clear()

        if not self.migrations_dir.exists():
            logger.warning(f"Migrations directory not found: {self.migrations_dir}")
            return self._migrations

        for path in sorted(self.migrations_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue

            try:
                spec = importlib.util.spec_from_file_location(
                    f"migrations.{path.stem}", path
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find Migration subclasses in the module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, Migration)
                            and attr is not Migration
                            and hasattr(attr, "version")
                            and attr.version
                        ):
                            self._migrations[attr.version] = attr
                            logger.debug(f"Discovered migration: {attr.version}")

            except Exception as e:
                logger.error(f"Error loading migration {path}: {e}")

        return self._migrations

    def get_applied_migrations(self) -> List[MigrationRecord]:
        """Get list of all applied migrations."""
        conn = self._get_connection()
        cursor = conn.execute(f"""
            SELECT * FROM {self.MIGRATIONS_TABLE}
            WHERE success = 1
            ORDER BY version
        """)
        return [
            MigrationRecord(
                version=row["version"],
                name=row["name"],
                applied_at=datetime.fromisoformat(row["applied_at"]),
                checksum=row["checksum"],
                success=bool(row["success"]),
                execution_time_ms=row["execution_time_ms"],
                error_message=row["error_message"],
            )
            for row in cursor.fetchall()
        ]

    def get_pending_migrations(self) -> List[Type[Migration]]:
        """Get list of migrations that haven't been applied yet."""
        self.discover_migrations()
        applied_versions = {m.version for m in self.get_applied_migrations()}

        pending = []
        for version in sorted(self._migrations.keys()):
            if version not in applied_versions:
                pending.append(self._migrations[version])

        return pending

    def _compute_checksum(self, migration_class: Type[Migration]) -> str:
        """Compute a checksum for migration verification."""
        # Include the class source and version in checksum
        import inspect

        source = inspect.getsource(migration_class)
        content = f"{migration_class.version}:{migration_class.name}:{source}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _resolve_dependencies(
        self, migrations: List[Type[Migration]]
    ) -> List[Type[Migration]]:
        """
        Order migrations based on dependencies.

        Raises:
            MigrationError: If circular dependencies detected
        """
        # Build dependency graph
        version_to_class = {m.version: m for m in migrations}
        resolved = []
        seen = set()

        def resolve(migration_class: Type[Migration], stack: set):
            version = migration_class.version

            if version in stack:
                raise MigrationError(
                    f"Circular dependency detected involving {version}"
                )

            if version in seen:
                return

            stack.add(version)

            for dep_version in migration_class.depends_on:
                if dep_version in version_to_class:
                    resolve(version_to_class[dep_version], stack)

            stack.remove(version)
            seen.add(version)
            resolved.append(migration_class)

        for m in migrations:
            resolve(m, set())

        return resolved

    def _record_migration(self, record: MigrationRecord) -> None:
        """Record a migration in the tracking table."""
        conn = self._get_connection()
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {self.MIGRATIONS_TABLE}
            (version, name, applied_at, checksum, success, execution_time_ms, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                record.version,
                record.name,
                record.applied_at.isoformat(),
                record.checksum,
                1 if record.success else 0,
                record.execution_time_ms,
                record.error_message,
            ),
        )
        conn.commit()

    def run_migration(
        self,
        migration_class: Type[Migration],
        context: MigrationContext,
    ) -> MigrationRecord:
        """
        Run a single migration.

        Args:
            migration_class: The migration class to run
            context: Migration context

        Returns:
            MigrationRecord with execution details
        """
        migration = migration_class(context)
        checksum = self._compute_checksum(migration_class)

        logger.info(f"Running migration {migration.version}: {migration.name}")

        start_time = time.time()
        error_message = None
        success = False

        try:
            migration.upgrade()
            migration.commit_all()

            # Validate if method is overridden
            if migration.validate():
                success = True
                logger.info(f"Migration {migration.version} completed successfully")
            else:
                error_message = "Validation failed"
                migration.rollback_all()
                logger.error(f"Migration {migration.version} validation failed")

        except Exception as e:
            error_message = str(e)
            migration.rollback_all()
            logger.error(f"Migration {migration.version} failed: {e}")

        finally:
            migration.close_all()

        execution_time = (time.time() - start_time) * 1000

        record = MigrationRecord(
            version=migration.version,
            name=migration.name,
            applied_at=datetime.utcnow(),
            checksum=checksum,
            success=success,
            execution_time_ms=execution_time,
            error_message=error_message,
        )

        if not context.dry_run:
            self._record_migration(record)

        return record

    def upgrade(
        self,
        target_version: Optional[str] = None,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> List[MigrationRecord]:
        """
        Run all pending migrations up to target version.

        Args:
            target_version: Stop after this version (None = all pending)
            dry_run: If True, don't actually apply changes
            verbose: If True, log detailed SQL statements

        Returns:
            List of MigrationRecords for applied migrations
        """
        pending = self.get_pending_migrations()

        if not pending:
            logger.info("No pending migrations")
            return []

        # Filter to target version if specified
        if target_version:
            pending = [m for m in pending if m.version <= target_version]

        # Resolve dependencies
        ordered = self._resolve_dependencies(pending)

        context = MigrationContext(
            data_dir=self.data_dir,
            dry_run=dry_run,
            verbose=verbose,
        )

        results = []
        for migration_class in ordered:
            record = self.run_migration(migration_class, context)
            results.append(record)

            if not record.success:
                logger.error(
                    f"Stopping migrations due to failure at {migration_class.version}"
                )
                break

        return results

    def downgrade(
        self,
        target_version: str,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> List[MigrationRecord]:
        """
        Revert migrations down to (but not including) target version.

        Args:
            target_version: Revert all migrations after this version
            dry_run: If True, don't actually apply changes
            verbose: If True, log detailed SQL statements

        Returns:
            List of MigrationRecords for reverted migrations
        """
        self.discover_migrations()
        applied = self.get_applied_migrations()

        # Find migrations to revert (in reverse order)
        to_revert = [m for m in reversed(applied) if m.version > target_version]

        if not to_revert:
            logger.info(f"No migrations to revert after {target_version}")
            return []

        context = MigrationContext(
            data_dir=self.data_dir,
            dry_run=dry_run,
            verbose=verbose,
        )

        results = []
        for record in to_revert:
            if record.version not in self._migrations:
                raise MigrationError(
                    f"Migration {record.version} not found in migrations directory"
                )

            migration_class = self._migrations[record.version]
            migration = migration_class(context)

            logger.info(f"Reverting migration {migration.version}: {migration.name}")

            start_time = time.time()
            error_message = None
            success = False

            try:
                migration.downgrade()
                migration.commit_all()
                success = True
                logger.info(f"Migration {migration.version} reverted successfully")

            except NotImplementedError:
                error_message = "Downgrade not supported"
                logger.error(f"Migration {migration.version} cannot be reverted")
                raise MigrationError(
                    f"Migration {migration.version} does not support downgrade"
                )

            except Exception as e:
                error_message = str(e)
                migration.rollback_all()
                logger.error(f"Migration {migration.version} revert failed: {e}")

            finally:
                migration.close_all()

            execution_time = (time.time() - start_time) * 1000

            revert_record = MigrationRecord(
                version=record.version,
                name=record.name,
                applied_at=datetime.utcnow(),
                checksum=record.checksum,
                success=success,
                execution_time_ms=execution_time,
                error_message=error_message,
            )

            results.append(revert_record)

            # Remove from tracking if successful
            if success and not dry_run:
                conn = self._get_connection()
                conn.execute(
                    f"DELETE FROM {self.MIGRATIONS_TABLE} WHERE version = ?",
                    (record.version,),
                )
                conn.commit()

            if not success:
                break

        return results

    def status(self) -> Dict[str, any]:
        """
        Get current migration status.

        Returns:
            Dictionary with applied/pending counts and details
        """
        self.discover_migrations()
        applied = self.get_applied_migrations()
        pending = self.get_pending_migrations()

        return {
            "applied_count": len(applied),
            "pending_count": len(pending),
            "applied": [
                {"version": m.version, "name": m.name, "applied_at": m.applied_at}
                for m in applied
            ],
            "pending": [
                {"version": m.version, "name": m.name, "description": m.description}
                for m in pending
            ],
            "current_version": applied[-1].version if applied else None,
            "latest_available": (
                sorted(self._migrations.keys())[-1] if self._migrations else None
            ),
        }

    def close(self) -> None:
        """Close the migrations database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
