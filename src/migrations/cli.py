"""
Migration CLI

Command-line interface for managing database migrations.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .backup import BackupManager, create_pre_migration_backup
from .runner import MigrationRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_default_data_dir() -> Path:
    """Get the default Agent OS data directory."""
    import os

    data_dir = os.environ.get("AGENT_OS_DATA_DIR")
    if data_dir:
        return Path(data_dir)

    # Default to ~/.agent-os/data
    return Path.home() / ".agent-os" / "data"


def cmd_status(args: argparse.Namespace) -> int:
    """Show migration status."""
    runner = MigrationRunner(args.data_dir)
    status = runner.status()
    runner.close()

    print(f"\nMigration Status")
    print("=" * 50)
    print(f"Current version: {status['current_version'] or 'None'}")
    print(f"Latest available: {status['latest_available'] or 'None'}")
    print(f"Applied: {status['applied_count']}")
    print(f"Pending: {status['pending_count']}")

    if status["applied"]:
        print(f"\nApplied Migrations:")
        for m in status["applied"]:
            print(f"  [{m['version']}] {m['name']} (applied: {m['applied_at']})")

    if status["pending"]:
        print(f"\nPending Migrations:")
        for m in status["pending"]:
            print(f"  [{m['version']}] {m['name']}")
            if m["description"]:
                print(f"      {m['description'][:60]}...")

    return 0


def cmd_upgrade(args: argparse.Namespace) -> int:
    """Run pending migrations."""
    runner = MigrationRunner(args.data_dir)

    # Create backup before migrating
    if not args.no_backup:
        try:
            logger.info("Creating pre-migration backup...")
            manifest = create_pre_migration_backup(args.data_dir)
            logger.info(f"Backup created: {manifest.backup_id}")
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            if not args.force:
                logger.error("Use --force to proceed without backup")
                return 1

    results = runner.upgrade(
        target_version=args.target,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    runner.close()

    if not results:
        print("No migrations to apply")
        return 0

    success_count = sum(1 for r in results if r.success)
    failed_count = len(results) - success_count

    print(f"\nMigration Results")
    print("=" * 50)

    for r in results:
        status = "✓" if r.success else "✗"
        print(f"  [{status}] {r.version}: {r.name} ({r.execution_time_ms:.1f}ms)")
        if r.error_message:
            print(f"      Error: {r.error_message}")

    print(f"\nApplied: {success_count}, Failed: {failed_count}")

    return 0 if failed_count == 0 else 1


def cmd_downgrade(args: argparse.Namespace) -> int:
    """Revert migrations."""
    if not args.target:
        logger.error("--target is required for downgrade")
        return 1

    runner = MigrationRunner(args.data_dir)

    # Create backup before downgrading
    if not args.no_backup:
        try:
            logger.info("Creating pre-downgrade backup...")
            manifest = create_pre_migration_backup(args.data_dir)
            logger.info(f"Backup created: {manifest.backup_id}")
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            if not args.force:
                return 1

    results = runner.downgrade(
        target_version=args.target,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    runner.close()

    if not results:
        print("No migrations to revert")
        return 0

    success_count = sum(1 for r in results if r.success)
    print(f"\nReverted {success_count} migration(s)")

    return 0


def cmd_backup(args: argparse.Namespace) -> int:
    """Create a backup."""
    manager = BackupManager(args.data_dir)

    manifest = manager.create_backup(
        include_blobs=not args.no_blobs,
        compress=not args.no_compress,
    )

    print(f"\nBackup Created")
    print("=" * 50)
    print(f"ID: {manifest.backup_id}")
    print(f"Created: {manifest.created_at}")
    print(f"Databases: {', '.join(manifest.databases)}")
    print(f"Blob storage: {manifest.blob_storage}")
    print(f"Size: {manifest.total_size_bytes / 1024 / 1024:.2f} MB")

    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    """Restore from a backup."""
    manager = BackupManager(args.data_dir)

    if not args.backup_id:
        # List available backups
        backups = manager.list_backups()
        if not backups:
            print("No backups available")
            return 1

        print("\nAvailable Backups:")
        print("=" * 50)
        for b in backups:
            print(f"  {b.backup_id} ({b.created_at})")
            print(f"    Databases: {', '.join(b.databases)}")
            print(f"    Size: {b.total_size_bytes / 1024 / 1024:.2f} MB")
        return 0

    manifest = manager.restore_backup(
        backup_id=args.backup_id,
        verify_checksums=not args.skip_verify,
    )

    print(f"\nRestore Complete")
    print("=" * 50)
    print(f"Restored from: {manifest.backup_id}")
    print(f"Databases: {', '.join(manifest.databases)}")

    return 0


def cmd_list_backups(args: argparse.Namespace) -> int:
    """List all backups."""
    manager = BackupManager(args.data_dir)
    backups = manager.list_backups()

    if not backups:
        print("No backups found")
        return 0

    print(f"\nBackups ({len(backups)} total)")
    print("=" * 50)

    for b in backups:
        print(f"\n{b.backup_id}")
        print(f"  Created: {b.created_at}")
        print(f"  Version: {b.agent_os_version}")
        print(f"  Databases: {', '.join(b.databases)}")
        print(f"  Blob storage: {b.blob_storage}")
        print(f"  Size: {b.total_size_bytes / 1024 / 1024:.2f} MB")
        if b.metadata:
            print(f"  Metadata: {b.metadata}")

    return 0


def main(argv: Optional[list] = None) -> int:
    """Main entry point for migration CLI."""
    parser = argparse.ArgumentParser(
        description="Agent OS Database Migration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=get_default_data_dir(),
        help="Agent OS data directory",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show migration status")
    status_parser.set_defaults(func=cmd_status)

    # Upgrade command
    upgrade_parser = subparsers.add_parser("upgrade", help="Run pending migrations")
    upgrade_parser.add_argument(
        "--target",
        help="Stop at this migration version",
    )
    upgrade_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    upgrade_parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip automatic backup before migration",
    )
    upgrade_parser.add_argument(
        "--force",
        action="store_true",
        help="Proceed even if backup fails",
    )
    upgrade_parser.set_defaults(func=cmd_upgrade)

    # Downgrade command
    downgrade_parser = subparsers.add_parser("downgrade", help="Revert migrations")
    downgrade_parser.add_argument(
        "--target",
        required=True,
        help="Revert to this migration version",
    )
    downgrade_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    downgrade_parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip automatic backup",
    )
    downgrade_parser.add_argument(
        "--force",
        action="store_true",
        help="Proceed even if backup fails",
    )
    downgrade_parser.set_defaults(func=cmd_downgrade)

    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create a backup")
    backup_parser.add_argument(
        "--no-blobs",
        action="store_true",
        help="Exclude blob storage from backup",
    )
    backup_parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Don't compress the backup",
    )
    backup_parser.set_defaults(func=cmd_backup)

    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from backup")
    restore_parser.add_argument(
        "backup_id",
        nargs="?",
        help="Backup ID to restore (omit to list available)",
    )
    restore_parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip checksum verification",
    )
    restore_parser.set_defaults(func=cmd_restore)

    # List backups command
    list_parser = subparsers.add_parser("list-backups", help="List all backups")
    list_parser.set_defaults(func=cmd_list_backups)

    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.command:
        parser.print_help()
        return 1

    # Ensure data directory exists
    args.data_dir.mkdir(parents=True, exist_ok=True)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
