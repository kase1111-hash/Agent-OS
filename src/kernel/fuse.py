"""FUSE Filesystem Wrapper for Conversational Kernel.

Provides filesystem-level policy enforcement using FUSE (Filesystem in Userspace).
Intercepts file operations and applies rules before allowing access.
"""

import errno
import logging
import os
import stat
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .policy import FilePolicy
from .rules import Rule, RuleAction, RuleEffect, RuleRegistry

logger = logging.getLogger(__name__)


class FuseOperation(str, Enum):
    """FUSE filesystem operations."""

    GETATTR = "getattr"
    READDIR = "readdir"
    OPEN = "open"
    READ = "read"
    WRITE = "write"
    CREATE = "create"
    UNLINK = "unlink"
    RMDIR = "rmdir"
    MKDIR = "mkdir"
    RENAME = "rename"
    CHMOD = "chmod"
    CHOWN = "chown"
    TRUNCATE = "truncate"
    LINK = "link"
    SYMLINK = "symlink"
    READLINK = "readlink"
    SETXATTR = "setxattr"
    GETXATTR = "getxattr"
    LISTXATTR = "listxattr"
    ACCESS = "access"


# Map FUSE operations to rule actions
OPERATION_TO_ACTION: Dict[FuseOperation, RuleAction] = {
    FuseOperation.GETATTR: RuleAction.STAT,
    FuseOperation.READDIR: RuleAction.LIST,
    FuseOperation.OPEN: RuleAction.READ,  # May also be WRITE based on flags
    FuseOperation.READ: RuleAction.READ,
    FuseOperation.WRITE: RuleAction.WRITE,
    FuseOperation.CREATE: RuleAction.CREATE,
    FuseOperation.UNLINK: RuleAction.DELETE,
    FuseOperation.RMDIR: RuleAction.DELETE,
    FuseOperation.MKDIR: RuleAction.CREATE,
    FuseOperation.RENAME: RuleAction.RENAME,
    FuseOperation.CHMOD: RuleAction.CHMOD,
    FuseOperation.CHOWN: RuleAction.CHOWN,
    FuseOperation.TRUNCATE: RuleAction.MODIFY,
    FuseOperation.LINK: RuleAction.LINK,
    FuseOperation.SYMLINK: RuleAction.LINK,
}


@dataclass
class FuseConfig:
    """Configuration for FUSE mount."""

    source_path: Path  # The real directory to wrap
    mount_point: Path  # Where to mount the FUSE filesystem
    read_only: bool = False
    allow_other: bool = False
    allow_root: bool = False
    default_permissions: bool = True
    debug: bool = False
    foreground: bool = False
    nothreads: bool = False
    auto_unmount: bool = True
    uid: Optional[int] = None
    gid: Optional[int] = None
    umask: int = 0o022

    def to_fuse_options(self) -> List[str]:
        """Convert to FUSE mount options."""
        options = []

        if self.read_only:
            options.append("ro")
        if self.allow_other:
            options.append("allow_other")
        if self.allow_root:
            options.append("allow_root")
        if self.default_permissions:
            options.append("default_permissions")
        if self.auto_unmount:
            options.append("auto_unmount")
        if self.uid is not None:
            options.append(f"uid={self.uid}")
        if self.gid is not None:
            options.append(f"gid={self.gid}")
        if self.umask:
            options.append(f"umask={oct(self.umask)[2:]}")

        return options


@dataclass
class FuseMount:
    """Information about an active FUSE mount."""

    mount_id: str
    config: FuseConfig
    started_at: datetime = field(default_factory=datetime.now)
    active: bool = True
    pid: Optional[int] = None
    access_count: int = 0
    denied_count: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mount_id": self.mount_id,
            "source_path": str(self.config.source_path),
            "mount_point": str(self.config.mount_point),
            "started_at": self.started_at.isoformat(),
            "active": self.active,
            "pid": self.pid,
            "access_count": self.access_count,
            "denied_count": self.denied_count,
        }


class FuseWrapper:
    """FUSE filesystem wrapper with policy enforcement.

    Intercepts filesystem operations and checks them against
    the rule registry before allowing access.

    Note: Actual FUSE mounting requires the 'fusepy' library
    and appropriate system permissions. This class provides
    the policy enforcement logic that would integrate with FUSE.
    """

    def __init__(
        self,
        rule_registry: RuleRegistry,
        audit_handler: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """Initialize FUSE wrapper.

        Args:
            rule_registry: Registry for rule lookup
            audit_handler: Optional handler for audit events
        """
        self.rule_registry = rule_registry
        self.audit_handler = audit_handler
        self._mounts: Dict[str, FuseMount] = {}
        self._operation_handlers: Dict[FuseOperation, Callable] = {}
        self._lock = threading.Lock()
        self._fuse_available = self._check_fuse_available()

    def _check_fuse_available(self) -> bool:
        """Check if FUSE is available on the system."""
        try:
            # Check for FUSE module/library
            if os.path.exists("/dev/fuse"):
                return True
            # Check fusermount command
            import shutil

            return shutil.which("fusermount") is not None or shutil.which("fusermount3") is not None
        except (OSError, PermissionError, ImportError):
            return False

    @property
    def is_available(self) -> bool:
        """Check if FUSE is available."""
        return self._fuse_available

    def mount(self, config: FuseConfig) -> Optional[FuseMount]:
        """Create a FUSE mount with policy enforcement.

        Args:
            config: Mount configuration

        Returns:
            FuseMount info if successful, None otherwise
        """
        if not self._fuse_available:
            logger.warning("FUSE not available on this system")
            return None

        # Validate paths
        if not config.source_path.exists():
            logger.error(f"Source path does not exist: {config.source_path}")
            return None

        if not config.mount_point.exists():
            try:
                config.mount_point.mkdir(parents=True)
            except Exception as e:
                logger.error(f"Failed to create mount point: {e}")
                return None

        mount_id = f"fuse_{len(self._mounts)}_{config.source_path.name}"

        mount = FuseMount(
            mount_id=mount_id,
            config=config,
        )

        with self._lock:
            self._mounts[mount_id] = mount

        logger.info(f"Prepared FUSE mount: {config.source_path} -> {config.mount_point}")

        # In production, this would start the actual FUSE process
        # For now, we just register the mount configuration
        return mount

    def unmount(self, mount_id: str) -> bool:
        """Unmount a FUSE filesystem.

        Args:
            mount_id: ID of mount to remove

        Returns:
            True if successful
        """
        with self._lock:
            if mount_id not in self._mounts:
                return False

            mount = self._mounts[mount_id]
            mount.active = False

            # In production, this would call fusermount -u
            logger.info(f"Unmounted: {mount.config.mount_point}")

            del self._mounts[mount_id]
            return True

    def check_access(
        self,
        path: str,
        operation: FuseOperation,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[Rule]]:
        """Check if an operation is allowed.

        Args:
            path: The file/directory path
            operation: The FUSE operation
            context: Additional context (uid, gid, pid, etc.)

        Returns:
            Tuple of (allowed, matching_rule)
        """
        context = context or {}

        # Map operation to rule action
        action = OPERATION_TO_ACTION.get(operation)
        if not action:
            # Unknown operation - allow by default
            return True, None

        # Handle OPEN with write flags
        if operation == FuseOperation.OPEN:
            flags = context.get("flags", 0)
            if flags & (os.O_WRONLY | os.O_RDWR):
                action = RuleAction.WRITE

        # Evaluate rules
        effect, rule = self.rule_registry.evaluate(path, action, context)

        allowed = effect in (RuleEffect.ALLOW, RuleEffect.AUDIT)

        # Handle audit effect
        if effect == RuleEffect.AUDIT and self.audit_handler:
            self._emit_audit_event(path, operation, context, rule, allowed)

        # Update stats
        self._update_stats(path, allowed)

        return allowed, rule

    def _emit_audit_event(
        self,
        path: str,
        operation: FuseOperation,
        context: Dict[str, Any],
        rule: Optional[Rule],
        allowed: bool,
    ) -> None:
        """Emit an audit event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "path": path,
            "operation": operation.value,
            "allowed": allowed,
            "rule_id": rule.rule_id if rule else None,
            "uid": context.get("uid"),
            "pid": context.get("pid"),
        }

        if self.audit_handler:
            try:
                self.audit_handler(event)
            except Exception as e:
                logger.warning(f"Audit handler error: {e}")

    def _update_stats(self, path: str, allowed: bool) -> None:
        """Update access statistics."""
        # Find relevant mount
        for mount in self._mounts.values():
            if path.startswith(str(mount.config.source_path)):
                mount.access_count += 1
                if not allowed:
                    mount.denied_count += 1
                break

    # FUSE operation handlers
    # These would be called by the actual FUSE implementation

    def handle_getattr(
        self, path: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Handle getattr operation.

        Args:
            path: File path
            context: Operation context

        Returns:
            File attributes or None if denied
        """
        allowed, rule = self.check_access(path, FuseOperation.GETATTR, context)

        if not allowed:
            return None

        # Get real attributes
        try:
            st = os.stat(path)
            return {
                "st_mode": st.st_mode,
                "st_nlink": st.st_nlink,
                "st_uid": st.st_uid,
                "st_gid": st.st_gid,
                "st_size": st.st_size,
                "st_atime": st.st_atime,
                "st_mtime": st.st_mtime,
                "st_ctime": st.st_ctime,
            }
        except OSError:
            return None

    def handle_readdir(
        self, path: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[List[str]]:
        """Handle readdir operation.

        Args:
            path: Directory path
            context: Operation context

        Returns:
            List of entries or None if denied
        """
        allowed, rule = self.check_access(path, FuseOperation.READDIR, context)

        if not allowed:
            return None

        try:
            entries = [".", ".."]
            for entry in os.listdir(path):
                # Check access to each entry
                entry_path = os.path.join(path, entry)
                entry_allowed, _ = self.check_access(entry_path, FuseOperation.GETATTR, context)
                if entry_allowed:
                    entries.append(entry)
            return entries
        except OSError:
            return None

    def handle_open(
        self,
        path: str,
        flags: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, int]:
        """Handle open operation.

        Args:
            path: File path
            flags: Open flags
            context: Operation context

        Returns:
            Tuple of (allowed, errno)
        """
        context = context or {}
        context["flags"] = flags

        allowed, rule = self.check_access(path, FuseOperation.OPEN, context)

        if not allowed:
            return False, errno.EACCES

        return True, 0

    def handle_read(
        self,
        path: str,
        size: int,
        offset: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[bytes]:
        """Handle read operation.

        Args:
            path: File path
            size: Bytes to read
            offset: Starting offset
            context: Operation context

        Returns:
            Data bytes or None if denied
        """
        allowed, rule = self.check_access(path, FuseOperation.READ, context)

        if not allowed:
            return None

        try:
            with open(path, "rb") as f:
                f.seek(offset)
                return f.read(size)
        except OSError:
            return None

    def handle_write(
        self,
        path: str,
        data: bytes,
        offset: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Handle write operation.

        Args:
            path: File path
            data: Data to write
            offset: Starting offset
            context: Operation context

        Returns:
            Bytes written or -EACCES if denied
        """
        allowed, rule = self.check_access(path, FuseOperation.WRITE, context)

        if not allowed:
            return -errno.EACCES

        try:
            with open(path, "r+b") as f:
                f.seek(offset)
                return f.write(data)
        except OSError as e:
            return -e.errno

    def handle_create(
        self,
        path: str,
        mode: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, int]:
        """Handle create operation.

        Args:
            path: File path
            mode: File mode
            context: Operation context

        Returns:
            Tuple of (allowed, errno)
        """
        allowed, rule = self.check_access(path, FuseOperation.CREATE, context)

        if not allowed:
            return False, errno.EACCES

        return True, 0

    def handle_unlink(self, path: str, context: Optional[Dict[str, Any]] = None) -> int:
        """Handle unlink (delete) operation.

        Args:
            path: File path
            context: Operation context

        Returns:
            0 if allowed, -EACCES if denied
        """
        allowed, rule = self.check_access(path, FuseOperation.UNLINK, context)

        if not allowed:
            return -errno.EACCES

        return 0

    def handle_rename(
        self,
        old_path: str,
        new_path: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Handle rename operation.

        Args:
            old_path: Source path
            new_path: Destination path
            context: Operation context

        Returns:
            0 if allowed, -EACCES if denied
        """
        # Check both source and destination
        allowed_src, _ = self.check_access(old_path, FuseOperation.RENAME, context)
        allowed_dst, _ = self.check_access(new_path, FuseOperation.CREATE, context)

        if not (allowed_src and allowed_dst):
            return -errno.EACCES

        return 0

    def list_mounts(self) -> List[Dict[str, Any]]:
        """List all active mounts."""
        with self._lock:
            return [m.to_dict() for m in self._mounts.values() if m.active]

    def get_mount(self, mount_id: str) -> Optional[FuseMount]:
        """Get a specific mount."""
        return self._mounts.get(mount_id)

    def apply_file_policies(self, policies: List[FilePolicy]) -> int:
        """Apply file policies as FUSE mounts.

        Args:
            policies: File policies to apply

        Returns:
            Number of mounts created
        """
        count = 0

        for policy in policies:
            path = Path(policy.path)

            if not path.exists():
                logger.warning(f"Policy path does not exist: {path}")
                continue

            # Create mount point in tmp
            mount_point = Path(f"/tmp/agentos_fuse/{path.name}")

            config = FuseConfig(
                source_path=path,
                mount_point=mount_point,
                read_only="write" not in policy.operations and "create" not in policy.operations,
            )

            if self.mount(config):
                count += 1

        return count
