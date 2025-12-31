"""File System Monitor for Conversational Kernel.

Provides monitoring of file system events using:
- inotify for real-time file watching
- Audit logging for security events
"""

import json
import logging
import os
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, Flag, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class EventType(Flag):
    """File system event types (matching inotify masks)."""

    ACCESS = auto()  # File accessed
    MODIFY = auto()  # File modified
    ATTRIB = auto()  # Metadata changed
    CLOSE_WRITE = auto()  # File closed after writing
    CLOSE_NOWRITE = auto()  # File closed without writing
    OPEN = auto()  # File opened
    MOVED_FROM = auto()  # File moved from
    MOVED_TO = auto()  # File moved to
    CREATE = auto()  # File/dir created
    DELETE = auto()  # File/dir deleted
    DELETE_SELF = auto()  # Watched item deleted
    MOVE_SELF = auto()  # Watched item moved

    # Convenience combinations
    CLOSE = CLOSE_WRITE | CLOSE_NOWRITE
    MOVE = MOVED_FROM | MOVED_TO
    ALL_EVENTS = (
        ACCESS
        | MODIFY
        | ATTRIB
        | CLOSE_WRITE
        | CLOSE_NOWRITE
        | OPEN
        | MOVED_FROM
        | MOVED_TO
        | CREATE
        | DELETE
        | DELETE_SELF
        | MOVE_SELF
    )

    @classmethod
    def from_inotify_mask(cls, mask: int) -> "EventType":
        """Convert inotify mask to EventType."""
        # inotify constants
        IN_ACCESS = 0x00000001
        IN_MODIFY = 0x00000002
        IN_ATTRIB = 0x00000004
        IN_CLOSE_WRITE = 0x00000008
        IN_CLOSE_NOWRITE = 0x00000010
        IN_OPEN = 0x00000020
        IN_MOVED_FROM = 0x00000040
        IN_MOVED_TO = 0x00000080
        IN_CREATE = 0x00000100
        IN_DELETE = 0x00000200
        IN_DELETE_SELF = 0x00000400
        IN_MOVE_SELF = 0x00000800

        result = EventType(0)

        if mask & IN_ACCESS:
            result |= cls.ACCESS
        if mask & IN_MODIFY:
            result |= cls.MODIFY
        if mask & IN_ATTRIB:
            result |= cls.ATTRIB
        if mask & IN_CLOSE_WRITE:
            result |= cls.CLOSE_WRITE
        if mask & IN_CLOSE_NOWRITE:
            result |= cls.CLOSE_NOWRITE
        if mask & IN_OPEN:
            result |= cls.OPEN
        if mask & IN_MOVED_FROM:
            result |= cls.MOVED_FROM
        if mask & IN_MOVED_TO:
            result |= cls.MOVED_TO
        if mask & IN_CREATE:
            result |= cls.CREATE
        if mask & IN_DELETE:
            result |= cls.DELETE
        if mask & IN_DELETE_SELF:
            result |= cls.DELETE_SELF
        if mask & IN_MOVE_SELF:
            result |= cls.MOVE_SELF

        return result


@dataclass
class MonitorEvent:
    """A file system monitoring event."""

    event_id: str
    event_type: EventType
    path: str
    filename: Optional[str] = None
    is_directory: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    source_path: Optional[str] = None  # For move events
    cookie: int = 0  # For correlating move events
    process_id: Optional[int] = None
    user_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_path(self) -> str:
        """Get full path including filename."""
        if self.filename:
            return os.path.join(self.path, self.filename)
        return self.path

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": (
                self.event_type.name
                if isinstance(self.event_type, EventType)
                else str(self.event_type)
            ),
            "path": self.path,
            "filename": self.filename,
            "full_path": self.full_path,
            "is_directory": self.is_directory,
            "timestamp": self.timestamp.isoformat(),
            "source_path": self.source_path,
            "cookie": self.cookie,
            "process_id": self.process_id,
            "user_id": self.user_id,
            "metadata": self.metadata,
        }


@dataclass
class AuditEntry:
    """An audit log entry."""

    entry_id: str
    timestamp: datetime
    event_type: str
    path: str
    action: str  # allowed, denied, audited
    rule_id: Optional[str] = None
    user_id: Optional[int] = None
    process_id: Optional[int] = None
    process_name: Optional[str] = None
    success: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "path": self.path,
            "action": self.action,
            "rule_id": self.rule_id,
            "user_id": self.user_id,
            "process_id": self.process_id,
            "process_name": self.process_name,
            "success": self.success,
            "details": self.details,
        }


class AuditLog:
    """Persistent audit log storage.

    Stores security-relevant events for compliance and debugging.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize audit log.

        Args:
            db_path: Path to SQLite database (None for in-memory)
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._entry_count = 0
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database."""
        db_str = str(self.db_path) if self.db_path else ":memory:"
        self._conn = sqlite3.connect(db_str, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                entry_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                path TEXT NOT NULL,
                action TEXT NOT NULL,
                rule_id TEXT,
                user_id INTEGER,
                process_id INTEGER,
                process_name TEXT,
                success INTEGER,
                details TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_audit_path ON audit_log(path);
            CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_log(event_type);
            CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);
        """
        )

    def log(self, entry: AuditEntry) -> None:
        """Log an audit entry.

        Args:
            entry: Audit entry to log
        """
        if not self._conn:
            return

        self._conn.execute(
            """
            INSERT INTO audit_log (
                entry_id, timestamp, event_type, path, action,
                rule_id, user_id, process_id, process_name, success, details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.entry_id,
                entry.timestamp.isoformat(),
                entry.event_type,
                entry.path,
                entry.action,
                entry.rule_id,
                entry.user_id,
                entry.process_id,
                entry.process_name,
                1 if entry.success else 0,
                json.dumps(entry.details),
            ),
        )
        self._conn.commit()
        self._entry_count += 1

    def query(
        self,
        path: Optional[str] = None,
        event_type: Optional[str] = None,
        action: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEntry]:
        """Query audit entries.

        Args:
            path: Filter by path prefix
            event_type: Filter by event type
            action: Filter by action (allowed, denied, audited)
            start_time: Filter entries after this time
            end_time: Filter entries before this time
            limit: Maximum entries to return
            offset: Offset for pagination

        Returns:
            List of matching audit entries
        """
        if not self._conn:
            return []

        query = "SELECT * FROM audit_log WHERE 1=1"
        params: List[Any] = []

        if path:
            query += " AND path LIKE ?"
            params.append(f"{path}%")

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if action:
            query += " AND action = ?"
            params.append(action)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = self._conn.execute(query, params)

        return [self._row_to_entry(row) for row in cursor.fetchall()]

    def _row_to_entry(self, row: sqlite3.Row) -> AuditEntry:
        """Convert database row to AuditEntry."""
        return AuditEntry(
            entry_id=row["entry_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            event_type=row["event_type"],
            path=row["path"],
            action=row["action"],
            rule_id=row["rule_id"],
            user_id=row["user_id"],
            process_id=row["process_id"],
            process_name=row["process_name"],
            success=bool(row["success"]),
            details=json.loads(row["details"]) if row["details"] else {},
        )

    def get_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get audit statistics.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Statistics dictionary
        """
        if not self._conn:
            return {}

        where = "WHERE 1=1"
        params: List[Any] = []

        if start_time:
            where += " AND timestamp >= ?"
            params.append(start_time.isoformat())

        if end_time:
            where += " AND timestamp <= ?"
            params.append(end_time.isoformat())

        stats = {
            "total_entries": 0,
            "by_action": {},
            "by_event_type": {},
            "denied_paths": [],
        }

        # Total count
        cursor = self._conn.execute(f"SELECT COUNT(*) as cnt FROM audit_log {where}", params)
        stats["total_entries"] = cursor.fetchone()["cnt"]

        # By action
        cursor = self._conn.execute(
            f"SELECT action, COUNT(*) as cnt FROM audit_log {where} GROUP BY action",
            params,
        )
        for row in cursor.fetchall():
            stats["by_action"][row["action"]] = row["cnt"]

        # By event type
        cursor = self._conn.execute(
            f"SELECT event_type, COUNT(*) as cnt FROM audit_log {where} GROUP BY event_type",
            params,
        )
        for row in cursor.fetchall():
            stats["by_event_type"][row["event_type"]] = row["cnt"]

        # Top denied paths
        cursor = self._conn.execute(
            f"""
            SELECT path, COUNT(*) as cnt FROM audit_log
            {where} AND action = 'denied'
            GROUP BY path ORDER BY cnt DESC LIMIT 10
            """,
            params,
        )
        stats["denied_paths"] = [
            {"path": row["path"], "count": row["cnt"]} for row in cursor.fetchall()
        ]

        return stats

    def cleanup(self, older_than: datetime) -> int:
        """Remove old audit entries.

        Args:
            older_than: Delete entries older than this

        Returns:
            Number of entries deleted
        """
        if not self._conn:
            return 0

        cursor = self._conn.execute(
            "DELETE FROM audit_log WHERE timestamp < ?",
            (older_than.isoformat(),),
        )
        self._conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# Event callback type
EventCallback = Callable[[MonitorEvent], None]


class FileMonitor:
    """File system monitor using inotify.

    Watches directories for file system events and
    invokes callbacks for policy enforcement.
    """

    def __init__(
        self,
        audit_log: Optional[AuditLog] = None,
    ):
        """Initialize file monitor.

        Args:
            audit_log: Optional audit log for recording events
        """
        self.audit_log = audit_log
        self._watches: Dict[int, str] = {}  # wd -> path
        self._paths: Dict[str, int] = {}  # path -> wd
        self._callbacks: List[EventCallback] = []
        self._event_queue: queue.Queue[MonitorEvent] = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._inotify_fd = -1
        self._event_counter = 0
        self._inotify_available = self._check_inotify_available()

    def _check_inotify_available(self) -> bool:
        """Check if inotify is available."""
        try:
            # Try to import inotify bindings or check /proc
            return os.path.exists("/proc/sys/fs/inotify")
        except Exception:
            return False

    @property
    def is_available(self) -> bool:
        """Check if monitoring is available."""
        return self._inotify_available

    def add_watch(
        self,
        path: str,
        events: EventType = EventType.ALL_EVENTS,
        recursive: bool = False,
    ) -> bool:
        """Add a path to watch.

        Args:
            path: Path to watch
            events: Events to monitor
            recursive: Watch subdirectories

        Returns:
            True if successful
        """
        path = os.path.abspath(path)

        if not os.path.exists(path):
            logger.warning(f"Path does not exist: {path}")
            return False

        if path in self._paths:
            logger.debug(f"Path already watched: {path}")
            return True

        # In production, this would use inotify_add_watch
        # For now, simulate with a unique watch descriptor
        wd = len(self._watches) + 1
        self._watches[wd] = path
        self._paths[path] = wd

        logger.info(f"Added watch: {path}")

        # Add subdirectories if recursive
        if recursive and os.path.isdir(path):
            try:
                for entry in os.scandir(path):
                    if entry.is_dir():
                        self.add_watch(entry.path, events, recursive=True)
            except PermissionError:
                logger.warning(f"Permission denied scanning: {path}")

        return True

    def remove_watch(self, path: str) -> bool:
        """Remove a watched path.

        Args:
            path: Path to stop watching

        Returns:
            True if successful
        """
        path = os.path.abspath(path)

        if path not in self._paths:
            return False

        wd = self._paths[path]
        del self._paths[path]
        del self._watches[wd]

        logger.info(f"Removed watch: {path}")
        return True

    def on_event(self, callback: EventCallback) -> None:
        """Register an event callback.

        Args:
            callback: Function to call on events
        """
        self._callbacks.append(callback)

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        self._event_counter += 1
        return f"evt_{self._event_counter}_{int(time.time() * 1000)}"

    def emit_event(self, event: MonitorEvent) -> None:
        """Emit a monitoring event.

        Args:
            event: Event to emit
        """
        # Queue for processing
        self._event_queue.put(event)

        # Call callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

        # Log to audit
        if self.audit_log:
            entry = AuditEntry(
                entry_id=f"audit_{event.event_id}",
                timestamp=event.timestamp,
                event_type=(
                    event.event_type.name
                    if isinstance(event.event_type, EventType)
                    else str(event.event_type)
                ),
                path=event.full_path,
                action="audited",
                process_id=event.process_id,
                user_id=event.user_id,
                details=event.metadata,
            )
            self.audit_log.log(entry)

    def simulate_event(
        self,
        path: str,
        event_type: EventType,
        **kwargs: Any,
    ) -> MonitorEvent:
        """Simulate a file system event (for testing).

        Args:
            path: File path
            event_type: Type of event
            **kwargs: Additional event attributes

        Returns:
            Generated event
        """
        event = MonitorEvent(
            event_id=self._generate_event_id(),
            event_type=event_type,
            path=os.path.dirname(path),
            filename=os.path.basename(path),
            is_directory=os.path.isdir(path) if os.path.exists(path) else False,
            **kwargs,
        )
        self.emit_event(event)
        return event

    def start(self) -> None:
        """Start the monitor thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("File monitor started")

    def stop(self) -> None:
        """Stop the monitor thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("File monitor stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            # In production, this would read from inotify fd
            # For now, just sleep and process queued events
            time.sleep(0.1)

            # Process any events in queue
            while not self._event_queue.empty():
                try:
                    self._event_queue.get_nowait()
                except queue.Empty:
                    break

    def list_watches(self) -> List[Dict[str, Any]]:
        """List all active watches."""
        return [{"wd": wd, "path": path} for wd, path in self._watches.items()]

    def get_pending_events(self) -> List[MonitorEvent]:
        """Get pending events from queue."""
        events = []
        while not self._event_queue.empty():
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def watch_for_violations(
        self,
        paths: List[str],
        rule_actions: List[str],
        callback: Callable[[MonitorEvent, str], None],
    ) -> None:
        """Watch paths for rule violations.

        Args:
            paths: Paths to watch
            rule_actions: Actions to monitor (read, write, delete, etc.)
            callback: Called with (event, action) on potential violation
        """
        # Map rule actions to event types
        action_to_events: Dict[str, EventType] = {
            "read": EventType.ACCESS | EventType.OPEN,
            "write": EventType.MODIFY | EventType.CLOSE_WRITE,
            "delete": EventType.DELETE,
            "create": EventType.CREATE,
            "rename": EventType.MOVE,
            "chmod": EventType.ATTRIB,
        }

        # Determine which events to watch
        events_to_watch = EventType(0)
        for action in rule_actions:
            if action in action_to_events:
                events_to_watch |= action_to_events[action]

        # Add watches
        for path in paths:
            self.add_watch(path, events_to_watch, recursive=True)

        # Register callback wrapper
        def violation_checker(event: MonitorEvent) -> None:
            for action, event_flags in action_to_events.items():
                if action in rule_actions and event.event_type & event_flags:
                    callback(event, action)

        self.on_event(violation_checker)
