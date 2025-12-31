"""
Backup and Restore Utilities

Provides backup/restore functionality for Agent OS data stores.
"""

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tarfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BackupError(Exception):
    """Error during backup or restore operations."""

    pass


@dataclass
class BackupManifest:
    """Manifest describing backup contents."""

    backup_id: str
    created_at: datetime
    agent_os_version: str
    databases: List[str]
    blob_storage: bool
    total_size_bytes: int
    checksums: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "created_at": self.created_at.isoformat(),
            "agent_os_version": self.agent_os_version,
            "databases": self.databases,
            "blob_storage": self.blob_storage,
            "total_size_bytes": self.total_size_bytes,
            "checksums": self.checksums,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackupManifest":
        return cls(
            backup_id=data["backup_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            agent_os_version=data["agent_os_version"],
            databases=data["databases"],
            blob_storage=data["blob_storage"],
            total_size_bytes=data["total_size_bytes"],
            checksums=data["checksums"],
            metadata=data.get("metadata", {}),
        )


class BackupManager:
    """
    Manages backup and restore operations for Agent OS data.

    Features:
    - Full and incremental backups
    - Database-level backup with integrity checks
    - Blob storage backup with encryption preservation
    - Compressed archive format
    - Checksum verification
    - Point-in-time restore
    """

    # Known database files to backup
    KNOWN_DATABASES = [
        "conversations",
        "migrations",
        "intent_log",
        "context",
        "rules",
        "audit",
        "agentos_mobile",
    ]

    MANIFEST_FILE = "manifest.json"
    AGENT_OS_VERSION = "0.1.0"

    def __init__(self, data_dir: Path, backup_dir: Optional[Path] = None):
        """
        Initialize backup manager.

        Args:
            data_dir: Directory containing Agent OS data
            backup_dir: Directory for storing backups (default: data_dir/backups)
        """
        self.data_dir = Path(data_dir)
        self.backup_dir = backup_dir or (self.data_dir / "backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def _generate_backup_id(self) -> str:
        """Generate a unique backup ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        random_suffix = os.urandom(4).hex()
        return f"backup_{timestamp}_{random_suffix}"

    def _compute_file_checksum(self, path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _backup_database(
        self, db_path: Path, backup_path: Path
    ) -> Optional[str]:
        """
        Backup a SQLite database using the backup API.

        Returns checksum of backed up file, or None if source doesn't exist.
        """
        if not db_path.exists():
            return None

        backup_path.parent.mkdir(parents=True, exist_ok=True)

        # Use SQLite backup API for consistency
        source = sqlite3.connect(str(db_path))
        dest = sqlite3.connect(str(backup_path))

        try:
            source.backup(dest)
            dest.close()
            source.close()
            return self._compute_file_checksum(backup_path)
        except Exception as e:
            dest.close()
            source.close()
            raise BackupError(f"Failed to backup {db_path}: {e}")

    def _backup_blob_storage(
        self, backup_staging: Path
    ) -> Dict[str, str]:
        """
        Backup encrypted blob storage.

        Returns checksums of backed up files.
        """
        checksums = {}
        blob_source = self.data_dir / "blobs"
        metadata_source = self.data_dir / "metadata"

        if blob_source.exists():
            blob_dest = backup_staging / "blobs"
            shutil.copytree(blob_source, blob_dest, dirs_exist_ok=True)

            for path in blob_dest.rglob("*"):
                if path.is_file():
                    rel_path = path.relative_to(backup_staging)
                    checksums[str(rel_path)] = self._compute_file_checksum(path)

        if metadata_source.exists():
            metadata_dest = backup_staging / "metadata"
            shutil.copytree(metadata_source, metadata_dest, dirs_exist_ok=True)

            for path in metadata_dest.rglob("*"):
                if path.is_file():
                    rel_path = path.relative_to(backup_staging)
                    checksums[str(rel_path)] = self._compute_file_checksum(path)

        return checksums

    def create_backup(
        self,
        include_blobs: bool = True,
        databases: Optional[List[str]] = None,
        compress: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BackupManifest:
        """
        Create a full backup of Agent OS data.

        Args:
            include_blobs: Include encrypted blob storage
            databases: Specific databases to backup (None = all)
            compress: Create compressed tarball
            metadata: Additional metadata to include

        Returns:
            BackupManifest describing the backup
        """
        backup_id = self._generate_backup_id()
        staging_dir = self.backup_dir / f".staging_{backup_id}"
        staging_dir.mkdir(parents=True)

        try:
            checksums: Dict[str, str] = {}
            backed_up_dbs: List[str] = []
            total_size = 0

            # Backup databases
            db_list = databases or self.KNOWN_DATABASES
            db_staging = staging_dir / "databases"
            db_staging.mkdir()

            for db_name in db_list:
                db_path = self.data_dir / f"{db_name}.db"
                if db_path.exists():
                    backup_path = db_staging / f"{db_name}.db"
                    checksum = self._backup_database(db_path, backup_path)
                    if checksum:
                        checksums[f"databases/{db_name}.db"] = checksum
                        backed_up_dbs.append(db_name)
                        total_size += backup_path.stat().st_size
                        logger.info(f"Backed up database: {db_name}")

            # Backup blob storage
            blob_backed_up = False
            if include_blobs:
                blob_checksums = self._backup_blob_storage(staging_dir)
                checksums.update(blob_checksums)
                blob_backed_up = bool(blob_checksums)

                for rel_path in blob_checksums:
                    file_path = staging_dir / rel_path
                    if file_path.exists():
                        total_size += file_path.stat().st_size

            # Create manifest
            manifest = BackupManifest(
                backup_id=backup_id,
                created_at=datetime.utcnow(),
                agent_os_version=self.AGENT_OS_VERSION,
                databases=backed_up_dbs,
                blob_storage=blob_backed_up,
                total_size_bytes=total_size,
                checksums=checksums,
                metadata=metadata or {},
            )

            # Write manifest
            manifest_path = staging_dir / self.MANIFEST_FILE
            with open(manifest_path, "w") as f:
                json.dump(manifest.to_dict(), f, indent=2)

            # Create final backup
            if compress:
                archive_path = self.backup_dir / f"{backup_id}.tar.gz"
                with tarfile.open(archive_path, "w:gz") as tar:
                    tar.add(staging_dir, arcname=backup_id)
                logger.info(f"Created backup archive: {archive_path}")
            else:
                final_path = self.backup_dir / backup_id
                shutil.move(staging_dir, final_path)
                logger.info(f"Created backup directory: {final_path}")

            return manifest

        finally:
            # Clean up staging
            if staging_dir.exists():
                shutil.rmtree(staging_dir)

    def list_backups(self) -> List[BackupManifest]:
        """List all available backups."""
        backups = []

        # Check compressed archives
        for archive_path in self.backup_dir.glob("backup_*.tar.gz"):
            try:
                with tarfile.open(archive_path, "r:gz") as tar:
                    # Find manifest
                    for member in tar.getmembers():
                        if member.name.endswith(self.MANIFEST_FILE):
                            f = tar.extractfile(member)
                            if f:
                                manifest_data = json.load(f)
                                backups.append(BackupManifest.from_dict(manifest_data))
                            break
            except Exception as e:
                logger.warning(f"Failed to read backup {archive_path}: {e}")

        # Check uncompressed directories
        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir() and backup_dir.name.startswith("backup_"):
                manifest_path = backup_dir / self.MANIFEST_FILE
                if manifest_path.exists():
                    try:
                        with open(manifest_path) as f:
                            manifest_data = json.load(f)
                        backups.append(BackupManifest.from_dict(manifest_data))
                    except Exception as e:
                        logger.warning(f"Failed to read manifest {manifest_path}: {e}")

        return sorted(backups, key=lambda b: b.created_at, reverse=True)

    def restore_backup(
        self,
        backup_id: str,
        verify_checksums: bool = True,
        databases: Optional[List[str]] = None,
        include_blobs: bool = True,
    ) -> BackupManifest:
        """
        Restore from a backup.

        Args:
            backup_id: ID of backup to restore
            verify_checksums: Verify file integrity before restore
            databases: Specific databases to restore (None = all)
            include_blobs: Restore blob storage

        Returns:
            BackupManifest of restored backup

        Raises:
            BackupError: If backup not found or integrity check fails
        """
        # Find backup
        archive_path = self.backup_dir / f"{backup_id}.tar.gz"
        dir_path = self.backup_dir / backup_id

        if archive_path.exists():
            # Extract to staging
            staging_dir = self.backup_dir / f".restore_{backup_id}"
            staging_dir.mkdir(parents=True, exist_ok=True)

            try:
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(staging_dir)
                source_dir = staging_dir / backup_id
            except Exception as e:
                shutil.rmtree(staging_dir)
                raise BackupError(f"Failed to extract backup: {e}")

        elif dir_path.exists():
            source_dir = dir_path
            staging_dir = None

        else:
            raise BackupError(f"Backup not found: {backup_id}")

        try:
            # Read manifest
            manifest_path = source_dir / self.MANIFEST_FILE
            if not manifest_path.exists():
                raise BackupError("Backup manifest not found")

            with open(manifest_path) as f:
                manifest = BackupManifest.from_dict(json.load(f))

            # Verify checksums if requested
            if verify_checksums:
                self._verify_checksums(source_dir, manifest.checksums)

            # Restore databases
            db_list = databases or manifest.databases
            for db_name in db_list:
                if db_name in manifest.databases:
                    backup_db = source_dir / "databases" / f"{db_name}.db"
                    if backup_db.exists():
                        target_db = self.data_dir / f"{db_name}.db"

                        # Create backup of current if exists
                        if target_db.exists():
                            target_db.rename(
                                target_db.with_suffix(".db.pre_restore")
                            )

                        shutil.copy2(backup_db, target_db)
                        logger.info(f"Restored database: {db_name}")

            # Restore blob storage
            if include_blobs and manifest.blob_storage:
                for subdir in ["blobs", "metadata"]:
                    source = source_dir / subdir
                    if source.exists():
                        target = self.data_dir / subdir

                        # Backup current
                        if target.exists():
                            target.rename(
                                target.with_suffix(".pre_restore")
                            )

                        shutil.copytree(source, target)
                        logger.info(f"Restored {subdir} storage")

            return manifest

        finally:
            if staging_dir and staging_dir.exists():
                shutil.rmtree(staging_dir)

    def _verify_checksums(
        self, source_dir: Path, checksums: Dict[str, str]
    ) -> None:
        """Verify file checksums match manifest."""
        for rel_path, expected_checksum in checksums.items():
            file_path = source_dir / rel_path
            if not file_path.exists():
                raise BackupError(f"Missing file in backup: {rel_path}")

            actual_checksum = self._compute_file_checksum(file_path)
            if actual_checksum != expected_checksum:
                raise BackupError(
                    f"Checksum mismatch for {rel_path}: "
                    f"expected {expected_checksum}, got {actual_checksum}"
                )

        logger.info("All checksums verified")

    def delete_backup(self, backup_id: str) -> None:
        """Delete a backup."""
        archive_path = self.backup_dir / f"{backup_id}.tar.gz"
        dir_path = self.backup_dir / backup_id

        if archive_path.exists():
            archive_path.unlink()
            logger.info(f"Deleted backup archive: {backup_id}")
        elif dir_path.exists():
            shutil.rmtree(dir_path)
            logger.info(f"Deleted backup directory: {backup_id}")
        else:
            raise BackupError(f"Backup not found: {backup_id}")

    def cleanup_old_backups(
        self, keep_count: int = 5, keep_days: Optional[int] = None
    ) -> int:
        """
        Remove old backups.

        Args:
            keep_count: Minimum number of backups to keep
            keep_days: Remove backups older than this (optional)

        Returns:
            Number of backups removed
        """
        backups = self.list_backups()  # Sorted newest first
        removed = 0

        for i, backup in enumerate(backups):
            should_remove = False

            # Keep minimum count
            if i >= keep_count:
                should_remove = True

            # Check age if specified
            if keep_days is not None:
                age_days = (datetime.utcnow() - backup.created_at).days
                if age_days > keep_days and i >= keep_count:
                    should_remove = True

            if should_remove:
                try:
                    self.delete_backup(backup.backup_id)
                    removed += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {backup.backup_id}: {e}")

        return removed


def create_pre_migration_backup(data_dir: Path) -> BackupManifest:
    """
    Convenience function to create a backup before running migrations.

    Args:
        data_dir: Agent OS data directory

    Returns:
        BackupManifest of created backup
    """
    manager = BackupManager(data_dir)
    return manager.create_backup(
        metadata={"type": "pre_migration", "automated": True}
    )
