# Data Migration Strategy

This document describes the data migration framework for Agent OS, including backup/restore procedures and guidelines for writing migrations.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Migration CLI](#migration-cli)
- [Writing Migrations](#writing-migrations)
- [Backup and Restore](#backup-and-restore)
- [Best Practices](#best-practices)
- [Database Schema](#database-schema)

## Overview

Agent OS uses a versioned migration system to manage database schema changes across multiple SQLite databases:

| Database | Purpose |
|----------|---------|
| `conversations.db` | Chat history and messages |
| `context.db` | User and agent context |
| `rules.db` | Constitutional rules |
| `audit.db` | Audit logging |
| `intent_log.db` | AI decision logging |
| `migrations.db` | Migration tracking |

### Key Features

- **Version tracking**: Each migration has a unique version ID
- **Dependency resolution**: Migrations can depend on others
- **Automatic backup**: Backups created before migrations
- **Dry run mode**: Preview changes without applying
- **Checksum verification**: Detect migration file changes
- **Rollback support**: Optional downgrade capability

## Quick Start

### Check Migration Status

```bash
python -m src.migrations.cli status
```

### Run Pending Migrations

```bash
python -m src.migrations.cli upgrade
```

### Create a Backup

```bash
python -m src.migrations.cli backup
```

### Restore from Backup

```bash
# List available backups
python -m src.migrations.cli restore

# Restore specific backup
python -m src.migrations.cli restore backup_20250101_120000_abc123
```

## Migration CLI

### Commands

```bash
# Show migration status
python -m src.migrations.cli status

# Run all pending migrations
python -m src.migrations.cli upgrade

# Run migrations up to a specific version
python -m src.migrations.cli upgrade --target 0002

# Preview migrations without applying (dry run)
python -m src.migrations.cli upgrade --dry-run

# Revert to a specific version
python -m src.migrations.cli downgrade --target 0001

# Create a backup
python -m src.migrations.cli backup

# List all backups
python -m src.migrations.cli list-backups

# Restore from backup
python -m src.migrations.cli restore <backup_id>
```

### Options

| Option | Description |
|--------|-------------|
| `--data-dir PATH` | Specify data directory (default: ~/.agent-os/data) |
| `--verbose` | Enable detailed logging |
| `--dry-run` | Preview changes without applying |
| `--no-backup` | Skip automatic pre-migration backup |
| `--force` | Proceed even if backup fails |

## Writing Migrations

### Migration File Structure

Migrations are Python files in `src/migrations/versions/`. Each file defines a Migration subclass:

```python
# src/migrations/versions/0003_example_migration.py
"""
Migration 0003: Example Migration

Brief description of what this migration does.
"""

from datetime import datetime
from typing import List
from ..base import Migration


class ExampleMigration(Migration):
    """Detailed description of the migration."""

    version = "0003"  # Unique version ID
    name = "example_migration"  # Short name
    description = "Full description of schema changes."

    # Which databases this migration affects
    databases: List[str] = ["conversations"]

    # Dependencies on other migrations
    depends_on: List[str] = ["0002"]

    def upgrade(self) -> None:
        """Apply the migration."""
        # Add a new column
        if not self.column_exists("conversations", "conversations", "new_field"):
            self.execute(
                "conversations",
                "ALTER TABLE conversations ADD COLUMN new_field TEXT"
            )

        # Create an index
        self.execute(
            "conversations",
            "CREATE INDEX IF NOT EXISTS idx_new_field ON conversations(new_field)"
        )

        # Record schema version
        self.execute(
            "conversations",
            """
            INSERT INTO schema_version
            (version, migration_version, applied_at, description)
            VALUES (?, ?, ?, ?)
            """,
            ("1.2.0", self.version, datetime.utcnow().isoformat(), self.description)
        )

        self.commit("conversations")

    def downgrade(self) -> None:
        """Revert the migration (optional)."""
        self.execute("conversations", "DROP INDEX IF EXISTS idx_new_field")
        # Note: SQLite 3.35+ required for DROP COLUMN
        self.execute(
            "conversations",
            "ALTER TABLE conversations DROP COLUMN new_field"
        )
        self.commit("conversations")

    def validate(self) -> bool:
        """Verify migration applied correctly (optional)."""
        return self.column_exists("conversations", "conversations", "new_field")
```

### Naming Convention

Migration files should follow this pattern:

```
XXXX_short_description.py
```

Where:
- `XXXX` is a 4-digit version number (0001, 0002, etc.)
- `short_description` briefly describes the change

### Available Helper Methods

The `Migration` base class provides:

```python
# Database operations
self.execute(db_name, sql, params=())  # Execute SQL
self.executemany(db_name, sql, params_list)  # Batch execute
self.commit(db_name)  # Commit changes
self.rollback(db_name)  # Rollback changes
self.commit_all()  # Commit all databases
self.rollback_all()  # Rollback all databases

# Schema inspection
self.table_exists(db_name, table_name)  # Check if table exists
self.column_exists(db_name, table_name, column_name)  # Check column
self.get_table_columns(db_name, table_name)  # Get column info

# Context
self.context.data_dir  # Data directory path
self.context.dry_run  # True if dry run mode
self.context.verbose  # True if verbose mode
self.context.get_db_path(name)  # Get full database path
```

## Backup and Restore

### Backup Contents

A full backup includes:

- All SQLite databases
- Encrypted blob storage (`blobs/`)
- Blob metadata (`metadata/`)
- Backup manifest with checksums

### Backup Structure

```
backup_YYYYMMDD_HHMMSS_xxxx/
├── manifest.json          # Backup metadata and checksums
├── databases/
│   ├── conversations.db
│   ├── context.db
│   └── ...
├── blobs/                 # Encrypted blob storage
│   └── ...
└── metadata/              # Blob metadata files
    └── ...
```

### Programmatic Backup

```python
from pathlib import Path
from src.migrations import BackupManager

# Create backup manager
manager = BackupManager(Path("~/.agent-os/data").expanduser())

# Create full backup
manifest = manager.create_backup(
    include_blobs=True,
    compress=True,
    metadata={"reason": "Pre-upgrade backup"}
)

print(f"Backup created: {manifest.backup_id}")

# List available backups
backups = manager.list_backups()
for backup in backups:
    print(f"{backup.backup_id}: {backup.created_at}")

# Restore from backup
manager.restore_backup(
    backup_id="backup_20250101_120000_abc123",
    verify_checksums=True
)

# Cleanup old backups (keep last 5)
removed = manager.cleanup_old_backups(keep_count=5)
```

## Best Practices

### General Guidelines

1. **Always test migrations** on a copy of production data
2. **Keep migrations small** - one logical change per migration
3. **Document thoroughly** - explain why changes are needed
4. **Provide downgrades** when possible for reversibility
5. **Use transactions** - migrations auto-commit on success

### SQLite Considerations

1. **ALTER TABLE limitations**: SQLite has limited ALTER TABLE support
   - Adding columns: Supported
   - Dropping columns: Requires SQLite 3.35+ or table recreation
   - Renaming columns: Requires SQLite 3.25+

2. **Table recreation pattern** for complex changes:
   ```python
   def upgrade(self):
       # Create new table with desired schema
       self.execute(db, "CREATE TABLE conversations_new (...)")

       # Copy data
       self.execute(db, "INSERT INTO conversations_new SELECT ... FROM conversations")

       # Swap tables
       self.execute(db, "DROP TABLE conversations")
       self.execute(db, "ALTER TABLE conversations_new RENAME TO conversations")

       # Recreate indexes
       self.execute(db, "CREATE INDEX ...")
   ```

3. **Foreign keys**: Enabled by default, use CASCADE appropriately

### Data Preservation

1. **Never delete data** without user consent
2. **Preserve encrypted blobs** - they contain user data
3. **Maintain consent records** - required for compliance
4. **Keep audit trails** - immutable intent logs

### Testing Migrations

```python
import tempfile
from pathlib import Path
from src.migrations import MigrationRunner, MigrationContext

# Create test environment
with tempfile.TemporaryDirectory() as tmpdir:
    data_dir = Path(tmpdir)

    # Copy test databases
    # ...

    # Run migrations
    runner = MigrationRunner(data_dir)
    results = runner.upgrade(dry_run=True)

    for r in results:
        assert r.success, f"Migration {r.version} failed: {r.error_message}"
```

## Database Schema

### Migration Tracking Table

Each database has a `schema_version` table:

```sql
CREATE TABLE schema_version (
    id INTEGER PRIMARY KEY,
    version TEXT NOT NULL,           -- Schema version (e.g., "1.2.0")
    migration_version TEXT NOT NULL, -- Migration that applied this (e.g., "0002")
    applied_at TEXT NOT NULL,        -- ISO timestamp
    description TEXT                 -- Human-readable description
);
```

### Central Migration History

The `migrations.db` database tracks all migrations:

```sql
CREATE TABLE migration_history (
    version TEXT PRIMARY KEY,        -- Migration version (e.g., "0002")
    name TEXT NOT NULL,              -- Migration name
    applied_at TEXT NOT NULL,        -- When applied
    checksum TEXT NOT NULL,          -- File content hash
    success INTEGER NOT NULL,        -- 1 = success, 0 = failed
    execution_time_ms REAL NOT NULL, -- Execution duration
    error_message TEXT               -- Error if failed
);
```

## Troubleshooting

### Migration Failed

1. Check the error message in migration output
2. Review logs: `--verbose` flag shows detailed SQL
3. Restore from automatic backup if needed

### Checksum Mismatch

Occurs when migration file changed after being applied:

```bash
# Force re-run (use carefully!)
python -m src.migrations.cli upgrade --force
```

### Database Locked

SQLite may lock during concurrent access:

1. Stop running Agent OS instances
2. Check for stale lock files
3. Retry migration

### Backup Restoration Failed

1. Verify backup integrity: `--skip-verify` to skip checksums
2. Check disk space
3. Ensure no processes are using databases
