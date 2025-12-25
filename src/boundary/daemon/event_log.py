"""
Smith Daemon Immutable Event Log

Append-only, cryptographically chained event log for audit purposes.
Cannot be modified or deleted once written.

This is part of Agent Smith's system-level enforcement mechanism within Agent-OS,
distinct from the external boundary-daemon project.
"""

import hashlib
import json
import os
import threading
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator
import uuid


logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """A single log entry in the event chain."""
    sequence: int
    timestamp: datetime
    event_type: str
    event_data: Dict[str, Any]
    previous_hash: str
    entry_hash: str
    signature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "event_data": self.event_data,
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogEntry":
        """Create from dictionary."""
        return cls(
            sequence=data["sequence"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_type=data["event_type"],
            event_data=data["event_data"],
            previous_hash=data["previous_hash"],
            entry_hash=data["entry_hash"],
            signature=data.get("signature"),
        )


class ImmutableEventLog:
    """
    Immutable, append-only event log with hash chain.

    Every entry is chained to the previous entry via cryptographic
    hash, making tampering detectable. The log cannot be modified
    or deleted once written.
    """

    # Genesis hash for the first entry
    GENESIS_HASH = "0" * 64

    def __init__(
        self,
        log_path: Optional[Path] = None,
        verify_on_load: bool = True,
    ):
        """
        Initialize event log.

        Args:
            log_path: Path to log file (None for in-memory only)
            verify_on_load: Verify chain integrity when loading
        """
        self.log_path = log_path
        self.verify_on_load = verify_on_load

        self._entries: List[LogEntry] = []
        self._lock = threading.Lock()
        self._sequence = 0
        self._last_hash = self.GENESIS_HASH

        # Load existing log if path provided
        if self.log_path and self.log_path.exists():
            self._load_from_file()

    def append(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        sign: bool = False,
    ) -> LogEntry:
        """
        Append an event to the log.

        Args:
            event_type: Type of event
            event_data: Event data
            sign: Whether to sign the entry

        Returns:
            The created LogEntry
        """
        with self._lock:
            self._sequence += 1

            entry = LogEntry(
                sequence=self._sequence,
                timestamp=datetime.now(),
                event_type=event_type,
                event_data=event_data,
                previous_hash=self._last_hash,
                entry_hash="",  # Will be computed
            )

            # Compute hash
            entry.entry_hash = self._compute_hash(entry)

            # Sign if requested
            if sign:
                entry.signature = self._sign_entry(entry)

            # Update chain
            self._last_hash = entry.entry_hash
            self._entries.append(entry)

            # Persist if path set
            if self.log_path:
                self._persist_entry(entry)

            return entry

    def log_event(
        self,
        event_type: str,
        **kwargs,
    ) -> LogEntry:
        """
        Convenience method to log an event.

        Args:
            event_type: Type of event
            **kwargs: Event data

        Returns:
            The created LogEntry
        """
        return self.append(event_type, kwargs)

    def log_tripwire(
        self,
        tripwire_id: str,
        reason: str,
        severity: int,
        **kwargs,
    ) -> LogEntry:
        """Log a tripwire event."""
        return self.append("tripwire", {
            "tripwire_id": tripwire_id,
            "reason": reason,
            "severity": severity,
            **kwargs,
        })

    def log_enforcement(
        self,
        action: str,
        reason: str,
        success: bool,
        **kwargs,
    ) -> LogEntry:
        """Log an enforcement event."""
        return self.append("enforcement", {
            "action": action,
            "reason": reason,
            "success": success,
            **kwargs,
        })

    def log_mode_change(
        self,
        old_mode: str,
        new_mode: str,
        reason: str,
        **kwargs,
    ) -> LogEntry:
        """Log a mode change event."""
        return self.append("mode_change", {
            "old_mode": old_mode,
            "new_mode": new_mode,
            "reason": reason,
            **kwargs,
        })

    def log_policy_decision(
        self,
        request_type: str,
        decision: str,
        reason: str,
        **kwargs,
    ) -> LogEntry:
        """Log a policy decision."""
        return self.append("policy_decision", {
            "request_type": request_type,
            "decision": decision,
            "reason": reason,
            **kwargs,
        })

    def get_entry(self, sequence: int) -> Optional[LogEntry]:
        """Get entry by sequence number."""
        for entry in self._entries:
            if entry.sequence == sequence:
                return entry
        return None

    def get_entries(
        self,
        start_sequence: int = 0,
        end_sequence: Optional[int] = None,
        event_type: Optional[str] = None,
    ) -> List[LogEntry]:
        """
        Get entries with optional filtering.

        Args:
            start_sequence: Starting sequence (inclusive)
            end_sequence: Ending sequence (inclusive)
            event_type: Filter by event type

        Returns:
            List of matching entries
        """
        entries = []
        for entry in self._entries:
            if entry.sequence < start_sequence:
                continue
            if end_sequence and entry.sequence > end_sequence:
                break
            if event_type and entry.event_type != event_type:
                continue
            entries.append(entry)
        return entries

    def get_latest(self, count: int = 10) -> List[LogEntry]:
        """Get the latest N entries."""
        return list(self._entries[-count:])

    def count(self) -> int:
        """Get total entry count."""
        return len(self._entries)

    def verify_integrity(self) -> tuple[bool, List[str]]:
        """
        Verify the integrity of the entire log chain.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        expected_hash = self.GENESIS_HASH

        for entry in self._entries:
            # Check previous hash
            if entry.previous_hash != expected_hash:
                errors.append(
                    f"Entry {entry.sequence}: previous_hash mismatch "
                    f"(expected {expected_hash[:16]}..., got {entry.previous_hash[:16]}...)"
                )

            # Verify entry hash
            computed = self._compute_hash(entry)
            if entry.entry_hash != computed:
                errors.append(
                    f"Entry {entry.sequence}: entry_hash mismatch "
                    f"(expected {computed[:16]}..., got {entry.entry_hash[:16]}...)"
                )

            expected_hash = entry.entry_hash

        return len(errors) == 0, errors

    def export_json(self, path: Optional[Path] = None) -> str:
        """
        Export log to JSON format.

        Args:
            path: Optional path to write to

        Returns:
            JSON string
        """
        data = {
            "exported_at": datetime.now().isoformat(),
            "entry_count": len(self._entries),
            "first_hash": self.GENESIS_HASH,
            "last_hash": self._last_hash,
            "entries": [e.to_dict() for e in self._entries],
        }

        json_str = json.dumps(data, indent=2)

        if path:
            path.write_text(json_str)

        return json_str

    def iter_entries(self) -> Iterator[LogEntry]:
        """Iterate over all entries."""
        yield from self._entries

    def _compute_hash(self, entry: LogEntry) -> str:
        """Compute hash for an entry."""
        # Create canonical representation
        data = {
            "sequence": entry.sequence,
            "timestamp": entry.timestamp.isoformat(),
            "event_type": entry.event_type,
            "event_data": entry.event_data,
            "previous_hash": entry.previous_hash,
        }

        # Compute SHA-256
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _sign_entry(self, entry: LogEntry) -> str:
        """Sign an entry (placeholder for real signing)."""
        # In production, would use actual cryptographic signing
        return hashlib.sha256(
            (entry.entry_hash + str(uuid.uuid4())).encode()
        ).hexdigest()[:32]

    def _persist_entry(self, entry: LogEntry) -> None:
        """Persist entry to file."""
        if not self.log_path:
            return

        try:
            # Append to file
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")

        except Exception as e:
            logger.error(f"Failed to persist log entry: {e}")

    def _load_from_file(self) -> None:
        """Load log from file."""
        if not self.log_path or not self.log_path.exists():
            return

        try:
            with open(self.log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        entry = LogEntry.from_dict(data)
                        self._entries.append(entry)
                        self._sequence = max(self._sequence, entry.sequence)
                        self._last_hash = entry.entry_hash

            if self.verify_on_load and self._entries:
                valid, errors = self.verify_integrity()
                if not valid:
                    logger.error(f"Log integrity errors: {errors}")
                    raise ValueError("Log integrity verification failed")

            logger.info(f"Loaded {len(self._entries)} log entries")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse log file: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load log: {e}")
            raise


def create_event_log(
    log_path: Optional[Path] = None,
    verify_on_load: bool = True,
) -> ImmutableEventLog:
    """Factory function to create an immutable event log."""
    return ImmutableEventLog(
        log_path=log_path,
        verify_on_load=verify_on_load,
    )
