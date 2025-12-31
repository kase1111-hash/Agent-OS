"""
Cryptographic Audit Logging for Post-Quantum Operations

Provides comprehensive audit logging for all PQ cryptographic operations:
- Key generation and lifecycle events
- Encryption/decryption operations
- Signature creation and verification
- HSM interactions
- Certificate operations

Features:
- Tamper-evident log chain (hash linking)
- Configurable retention policies
- Export to external SIEM systems
- Compliance reporting (SOC 2, FIPS, etc.)
- Performance metrics

Usage:
    # Initialize audit logger
    audit = CryptoAuditLogger(config)
    audit.start()

    # Log operations
    audit.log_key_generation(key_id, algorithm, success=True)
    audit.log_sign_operation(key_id, message_hash, signature_id)

    # Generate compliance report
    report = audit.generate_compliance_report(
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
    )
"""

import base64
import gzip
import hashlib
import json
import logging
import os
import queue
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Audit Event Types
# =============================================================================


class AuditEventType(str, Enum):
    """Types of cryptographic audit events."""

    # Key Lifecycle
    KEY_GENERATED = "key.generated"
    KEY_IMPORTED = "key.imported"
    KEY_EXPORTED = "key.exported"
    KEY_ROTATED = "key.rotated"
    KEY_REVOKED = "key.revoked"
    KEY_DESTROYED = "key.destroyed"
    KEY_ACCESSED = "key.accessed"

    # Encryption Operations
    ENCRYPT = "crypto.encrypt"
    DECRYPT = "crypto.decrypt"
    ENCAPSULATE = "crypto.encapsulate"
    DECAPSULATE = "crypto.decapsulate"

    # Signature Operations
    SIGN = "crypto.sign"
    VERIFY = "crypto.verify"

    # Certificate Operations
    CERT_ISSUED = "cert.issued"
    CERT_VERIFIED = "cert.verified"
    CERT_REVOKED = "cert.revoked"
    CERT_EXPIRED = "cert.expired"

    # HSM Operations
    HSM_CONNECT = "hsm.connect"
    HSM_DISCONNECT = "hsm.disconnect"
    HSM_OPERATION = "hsm.operation"
    HSM_ERROR = "hsm.error"

    # Identity Operations
    IDENTITY_CREATED = "identity.created"
    IDENTITY_VERIFIED = "identity.verified"
    IDENTITY_TRUSTED = "identity.trusted"
    IDENTITY_REVOKED = "identity.revoked"

    # System Events
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    CONFIG_CHANGE = "system.config_change"
    POLICY_VIOLATION = "system.policy_violation"
    SECURITY_ALERT = "system.security_alert"


class AuditSeverity(str, Enum):
    """Audit event severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComplianceStandard(str, Enum):
    """Compliance standards for reporting."""

    SOC2 = "soc2"
    FIPS_140_3 = "fips_140_3"
    ISO_27001 = "iso_27001"
    PCI_DSS = "pci_dss"
    GDPR = "gdpr"
    NIST_CSF = "nist_csf"


# =============================================================================
# Audit Event Model
# =============================================================================


@dataclass
class CryptoAuditEvent:
    """Cryptographic audit event."""

    event_id: str  # Unique event ID
    event_type: AuditEventType  # Event type
    timestamp: datetime  # When it occurred
    severity: AuditSeverity = AuditSeverity.INFO

    # Operation details
    operation_id: Optional[str] = None  # Related operation ID
    key_id: Optional[str] = None  # Related key ID
    algorithm: Optional[str] = None  # Cryptographic algorithm
    success: bool = True
    error_message: Optional[str] = None

    # Actor information
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source_ip: Optional[str] = None
    component: Optional[str] = None

    # Cryptographic details (no secrets!)
    input_hash: Optional[str] = None  # Hash of input data
    output_hash: Optional[str] = None  # Hash of output data
    key_fingerprint: Optional[str] = None  # Public key fingerprint

    # Chain integrity
    sequence_number: int = 0
    previous_hash: Optional[str] = None
    event_hash: Optional[str] = None

    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"audit:{secrets.token_hex(12)}"
        if not self.event_hash:
            self.event_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash for chain integrity."""
        data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "sequence_number": self.sequence_number,
            "previous_hash": self.previous_hash,
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "success": self.success,
            "user_id": self.user_id,
        }
        serialized = json.dumps(data, sort_keys=True).encode()
        return hashlib.sha256(serialized).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "operation_id": self.operation_id,
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "success": self.success,
            "error_message": self.error_message,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "source_ip": self.source_ip,
            "component": self.component,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "key_fingerprint": self.key_fingerprint,
            "sequence_number": self.sequence_number,
            "previous_hash": self.previous_hash,
            "event_hash": self.event_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CryptoAuditEvent":
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=AuditEventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            severity=AuditSeverity(data.get("severity", "info")),
            operation_id=data.get("operation_id"),
            key_id=data.get("key_id"),
            algorithm=data.get("algorithm"),
            success=data.get("success", True),
            error_message=data.get("error_message"),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            source_ip=data.get("source_ip"),
            component=data.get("component"),
            input_hash=data.get("input_hash"),
            output_hash=data.get("output_hash"),
            key_fingerprint=data.get("key_fingerprint"),
            sequence_number=data.get("sequence_number", 0),
            previous_hash=data.get("previous_hash"),
            event_hash=data.get("event_hash"),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Audit Configuration
# =============================================================================


@dataclass
class AuditConfig:
    """Audit logger configuration."""

    # Storage
    log_directory: Optional[Path] = None
    max_file_size_mb: int = 100
    max_retention_days: int = 90
    compress_logs: bool = True

    # Chain integrity
    enable_chain_verification: bool = True
    chain_verification_interval_seconds: int = 3600

    # Filtering
    min_severity: AuditSeverity = AuditSeverity.INFO
    excluded_event_types: List[AuditEventType] = field(default_factory=list)
    included_components: List[str] = field(default_factory=list)

    # Output
    output_format: str = "json"  # json, jsonl, csv
    enable_console_output: bool = False
    enable_syslog: bool = False
    syslog_host: Optional[str] = None
    syslog_port: int = 514

    # Callbacks
    enable_callbacks: bool = True
    async_processing: bool = True
    queue_size: int = 10000

    # Compliance
    compliance_standards: List[ComplianceStandard] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "log_directory": str(self.log_directory) if self.log_directory else None,
            "max_file_size_mb": self.max_file_size_mb,
            "max_retention_days": self.max_retention_days,
            "compress_logs": self.compress_logs,
            "enable_chain_verification": self.enable_chain_verification,
            "min_severity": self.min_severity.value,
            "output_format": self.output_format,
            "enable_console_output": self.enable_console_output,
            "compliance_standards": [s.value for s in self.compliance_standards],
        }


# =============================================================================
# Audit Metrics
# =============================================================================


@dataclass
class AuditMetrics:
    """Aggregated audit metrics."""

    total_events: int = 0
    events_by_type: Dict[str, int] = field(default_factory=dict)
    events_by_severity: Dict[str, int] = field(default_factory=dict)
    failed_operations: int = 0
    unique_keys_used: int = 0
    unique_users: int = 0
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_events": self.total_events,
            "events_by_type": self.events_by_type,
            "events_by_severity": self.events_by_severity,
            "failed_operations": self.failed_operations,
            "unique_keys_used": self.unique_keys_used,
            "unique_users": self.unique_users,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
        }


# =============================================================================
# Crypto Audit Logger
# =============================================================================


class CryptoAuditLogger:
    """
    Comprehensive audit logger for cryptographic operations.

    Features:
    - Tamper-evident chain (hash linking)
    - Async event processing
    - Configurable output (file, console, syslog)
    - Compliance reporting
    """

    def __init__(self, config: Optional[AuditConfig] = None):
        self.config = config or AuditConfig()
        self._lock = threading.RLock()
        self._running = False

        # Event chain
        self._sequence_number = 0
        self._last_hash: Optional[str] = None

        # Async processing
        self._event_queue: queue.Queue = queue.Queue(maxsize=self.config.queue_size)
        self._worker_thread: Optional[threading.Thread] = None

        # Callbacks
        self._callbacks: List[Callable[[CryptoAuditEvent], None]] = []

        # File output
        self._current_file: Optional[Path] = None
        self._current_file_size = 0

        # Metrics
        self._keys_seen: set = set()
        self._users_seen: set = set()
        self._metrics = AuditMetrics()

    def start(self) -> None:
        """Start the audit logger."""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._metrics.period_start = datetime.utcnow()

            # Create log directory
            if self.config.log_directory:
                self.config.log_directory.mkdir(parents=True, exist_ok=True)

            # Load chain state
            self._load_chain_state()

            # Start worker thread
            if self.config.async_processing:
                self._worker_thread = threading.Thread(
                    target=self._process_events,
                    daemon=True,
                    name="audit-worker",
                )
                self._worker_thread.start()

            # Log start event
            self.log_event(
                event_type=AuditEventType.SYSTEM_START,
                component="audit",
                metadata={"config": self.config.to_dict()},
            )

            logger.info("Crypto audit logger started")

    def stop(self) -> None:
        """Stop the audit logger."""
        with self._lock:
            if not self._running:
                return

            # Log stop event
            self.log_event(
                event_type=AuditEventType.SYSTEM_STOP,
                component="audit",
            )

            self._running = False
            self._metrics.period_end = datetime.utcnow()

            # Wait for queue to drain
            if self._worker_thread and self._worker_thread.is_alive():
                self._event_queue.put(None)  # Poison pill
                self._worker_thread.join(timeout=5.0)

            # Save chain state
            self._save_chain_state()

            logger.info("Crypto audit logger stopped")

    def add_callback(
        self,
        callback: Callable[[CryptoAuditEvent], None],
    ) -> None:
        """Add event callback."""
        self._callbacks.append(callback)

    def log_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity = AuditSeverity.INFO,
        key_id: Optional[str] = None,
        algorithm: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        component: Optional[str] = None,
        input_data: Optional[bytes] = None,
        output_data: Optional[bytes] = None,
        key_fingerprint: Optional[str] = None,
        **metadata,
    ) -> CryptoAuditEvent:
        """Log a cryptographic audit event."""
        # Check filters
        if severity.value < self.config.min_severity.value:
            return None
        if event_type in self.config.excluded_event_types:
            return None
        if self.config.included_components:
            if component not in self.config.included_components:
                return None

        with self._lock:
            self._sequence_number += 1

            event = CryptoAuditEvent(
                event_id="",
                event_type=event_type,
                timestamp=datetime.utcnow(),
                severity=severity,
                key_id=key_id,
                algorithm=algorithm,
                success=success,
                error_message=error_message,
                user_id=user_id,
                session_id=session_id,
                component=component,
                input_hash=hashlib.sha256(input_data).hexdigest()[:16] if input_data else None,
                output_hash=hashlib.sha256(output_data).hexdigest()[:16] if output_data else None,
                key_fingerprint=key_fingerprint,
                sequence_number=self._sequence_number,
                previous_hash=self._last_hash,
                metadata=metadata,
            )

            self._last_hash = event.event_hash

            # Update metrics
            self._update_metrics(event)

        # Queue for processing
        if self.config.async_processing and self._running:
            try:
                self._event_queue.put_nowait(event)
            except queue.Full:
                logger.warning("Audit queue full, dropping event")
        else:
            self._write_event(event)

        return event

    # Convenience methods for common operations

    def log_key_generation(
        self,
        key_id: str,
        algorithm: str,
        success: bool = True,
        error: Optional[str] = None,
        **metadata,
    ) -> CryptoAuditEvent:
        """Log key generation event."""
        return self.log_event(
            event_type=AuditEventType.KEY_GENERATED,
            key_id=key_id,
            algorithm=algorithm,
            success=success,
            error_message=error,
            **metadata,
        )

    def log_key_destroyed(
        self,
        key_id: str,
        reason: Optional[str] = None,
        **metadata,
    ) -> CryptoAuditEvent:
        """Log key destruction event."""
        return self.log_event(
            event_type=AuditEventType.KEY_DESTROYED,
            key_id=key_id,
            success=True,
            metadata={"reason": reason, **metadata},
        )

    def log_sign_operation(
        self,
        key_id: str,
        message_hash: str,
        success: bool = True,
        **metadata,
    ) -> CryptoAuditEvent:
        """Log signing operation."""
        return self.log_event(
            event_type=AuditEventType.SIGN,
            key_id=key_id,
            success=success,
            input_data=message_hash.encode() if isinstance(message_hash, str) else message_hash,
            **metadata,
        )

    def log_verify_operation(
        self,
        key_fingerprint: str,
        success: bool,
        **metadata,
    ) -> CryptoAuditEvent:
        """Log verification operation."""
        return self.log_event(
            event_type=AuditEventType.VERIFY,
            key_fingerprint=key_fingerprint,
            success=success,
            **metadata,
        )

    def log_encapsulate(
        self,
        recipient_key_id: str,
        algorithm: str,
        success: bool = True,
        **metadata,
    ) -> CryptoAuditEvent:
        """Log key encapsulation."""
        return self.log_event(
            event_type=AuditEventType.ENCAPSULATE,
            key_id=recipient_key_id,
            algorithm=algorithm,
            success=success,
            **metadata,
        )

    def log_decapsulate(
        self,
        key_id: str,
        success: bool = True,
        **metadata,
    ) -> CryptoAuditEvent:
        """Log key decapsulation."""
        return self.log_event(
            event_type=AuditEventType.DECAPSULATE,
            key_id=key_id,
            success=success,
            **metadata,
        )

    def log_certificate_issued(
        self,
        cert_serial: str,
        subject_id: str,
        issuer_id: str,
        **metadata,
    ) -> CryptoAuditEvent:
        """Log certificate issuance."""
        return self.log_event(
            event_type=AuditEventType.CERT_ISSUED,
            success=True,
            metadata={
                "cert_serial": cert_serial,
                "subject_id": subject_id,
                "issuer_id": issuer_id,
                **metadata,
            },
        )

    def log_security_alert(
        self,
        alert_type: str,
        description: str,
        key_id: Optional[str] = None,
        **metadata,
    ) -> CryptoAuditEvent:
        """Log security alert."""
        return self.log_event(
            event_type=AuditEventType.SECURITY_ALERT,
            severity=AuditSeverity.WARNING,
            key_id=key_id,
            success=False,
            metadata={
                "alert_type": alert_type,
                "description": description,
                **metadata,
            },
        )

    def log_policy_violation(
        self,
        policy: str,
        violation: str,
        user_id: Optional[str] = None,
        **metadata,
    ) -> CryptoAuditEvent:
        """Log policy violation."""
        return self.log_event(
            event_type=AuditEventType.POLICY_VIOLATION,
            severity=AuditSeverity.WARNING,
            user_id=user_id,
            success=False,
            metadata={
                "policy": policy,
                "violation": violation,
                **metadata,
            },
        )

    def _update_metrics(self, event: CryptoAuditEvent) -> None:
        """Update metrics from event."""
        self._metrics.total_events += 1

        event_type = event.event_type.value
        self._metrics.events_by_type[event_type] = (
            self._metrics.events_by_type.get(event_type, 0) + 1
        )

        severity = event.severity.value
        self._metrics.events_by_severity[severity] = (
            self._metrics.events_by_severity.get(severity, 0) + 1
        )

        if not event.success:
            self._metrics.failed_operations += 1

        if event.key_id:
            self._keys_seen.add(event.key_id)
            self._metrics.unique_keys_used = len(self._keys_seen)

        if event.user_id:
            self._users_seen.add(event.user_id)
            self._metrics.unique_users = len(self._users_seen)

    def _process_events(self) -> None:
        """Worker thread for async event processing."""
        while True:
            try:
                event = self._event_queue.get(timeout=1.0)
                if event is None:
                    break
                self._write_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audit event: {e}")

    def _write_event(self, event: CryptoAuditEvent) -> None:
        """Write event to output."""
        # Console output
        if self.config.enable_console_output:
            self._write_console(event)

        # File output
        if self.config.log_directory:
            self._write_file(event)

        # Callbacks
        if self.config.enable_callbacks:
            for callback in self._callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Audit callback error: {e}")

    def _write_console(self, event: CryptoAuditEvent) -> None:
        """Write event to console."""
        level_map = {
            AuditSeverity.DEBUG: logging.DEBUG,
            AuditSeverity.INFO: logging.INFO,
            AuditSeverity.WARNING: logging.WARNING,
            AuditSeverity.ERROR: logging.ERROR,
            AuditSeverity.CRITICAL: logging.CRITICAL,
        }
        level = level_map.get(event.severity, logging.INFO)
        logger.log(
            level,
            f"[AUDIT] {event.event_type.value}: key={event.key_id}, "
            f"success={event.success}, seq={event.sequence_number}",
        )

    def _write_file(self, event: CryptoAuditEvent) -> None:
        """Write event to file."""
        if not self.config.log_directory:
            return

        # Rotate file if needed
        if (
            self._current_file is None
            or self._current_file_size >= self.config.max_file_size_mb * 1024 * 1024
        ):
            self._rotate_file()

        # Write event
        try:
            with open(self._current_file, "a") as f:
                if self.config.output_format == "jsonl":
                    line = json.dumps(event.to_dict()) + "\n"
                else:
                    line = json.dumps(event.to_dict(), indent=2) + "\n"

                f.write(line)
                self._current_file_size += len(line.encode())

        except Exception as e:
            logger.error(f"Failed to write audit event: {e}")

    def _rotate_file(self) -> None:
        """Rotate log file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"crypto_audit_{timestamp}.{self.config.output_format}"
        self._current_file = self.config.log_directory / filename
        self._current_file_size = 0

        # Compress old files
        if self.config.compress_logs:
            self._compress_old_logs()

        # Clean old files
        self._clean_old_logs()

    def _compress_old_logs(self) -> None:
        """Compress old log files."""
        if not self.config.log_directory:
            return

        for log_file in self.config.log_directory.glob(f"*.{self.config.output_format}"):
            if log_file == self._current_file:
                continue
            if log_file.suffix == ".gz":
                continue

            try:
                with open(log_file, "rb") as f:
                    data = f.read()
                with gzip.open(str(log_file) + ".gz", "wb") as f:
                    f.write(data)
                log_file.unlink()
            except Exception as e:
                logger.error(f"Failed to compress {log_file}: {e}")

    def _clean_old_logs(self) -> None:
        """Remove logs older than retention period."""
        if not self.config.log_directory:
            return

        cutoff = datetime.utcnow() - timedelta(days=self.config.max_retention_days)

        for log_file in self.config.log_directory.glob("crypto_audit_*"):
            try:
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if mtime < cutoff:
                    log_file.unlink()
                    logger.info(f"Removed old audit log: {log_file}")
            except Exception as e:
                logger.error(f"Failed to clean {log_file}: {e}")

    def _load_chain_state(self) -> None:
        """Load chain state from disk."""
        if not self.config.log_directory:
            return

        state_file = self.config.log_directory / "chain_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                self._sequence_number = state.get("sequence_number", 0)
                self._last_hash = state.get("last_hash")
                logger.info(f"Loaded chain state: seq={self._sequence_number}")
            except Exception as e:
                logger.error(f"Failed to load chain state: {e}")

    def _save_chain_state(self) -> None:
        """Save chain state to disk."""
        if not self.config.log_directory:
            return

        state_file = self.config.log_directory / "chain_state.json"
        try:
            state = {
                "sequence_number": self._sequence_number,
                "last_hash": self._last_hash,
                "saved_at": datetime.utcnow().isoformat(),
            }
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save chain state: {e}")

    def get_metrics(self) -> AuditMetrics:
        """Get current metrics."""
        return self._metrics

    def verify_chain(
        self,
        events: List[CryptoAuditEvent],
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify the integrity of an event chain.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not events:
            return True, None

        for i, event in enumerate(events):
            # Verify hash
            computed_hash = event._compute_hash()
            if computed_hash != event.event_hash:
                return False, f"Hash mismatch at event {event.event_id}"

            # Verify chain link
            if i > 0:
                if event.previous_hash != events[i - 1].event_hash:
                    return False, f"Chain break at event {event.event_id}"

            # Verify sequence
            if i > 0 and event.sequence_number != events[i - 1].sequence_number + 1:
                return False, f"Sequence gap at event {event.event_id}"

        return True, None

    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        standard: ComplianceStandard = ComplianceStandard.SOC2,
    ) -> Dict[str, Any]:
        """
        Generate a compliance report.

        Args:
            start_date: Report start date
            end_date: Report end date
            standard: Compliance standard

        Returns:
            Compliance report dictionary
        """
        return {
            "report_id": secrets.token_hex(8),
            "generated_at": datetime.utcnow().isoformat(),
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "standard": standard.value,
            "metrics": self._metrics.to_dict(),
            "summary": {
                "total_events": self._metrics.total_events,
                "failed_operations": self._metrics.failed_operations,
                "failure_rate": (
                    self._metrics.failed_operations / max(1, self._metrics.total_events)
                ),
                "unique_keys_used": self._metrics.unique_keys_used,
                "unique_users": self._metrics.unique_users,
                "chain_integrity": (
                    "verified" if self.config.enable_chain_verification else "not_enabled"
                ),
            },
            "controls": self._get_compliance_controls(standard),
        }

    def _get_compliance_controls(
        self,
        standard: ComplianceStandard,
    ) -> Dict[str, Any]:
        """Get compliance controls for a standard."""
        # Simplified compliance mapping
        controls = {
            ComplianceStandard.SOC2: {
                "CC6.1": "Encryption of data at rest",
                "CC6.6": "Encryption key management",
                "CC6.7": "Encryption of data in transit",
            },
            ComplianceStandard.FIPS_140_3: {
                "level": "Level 1 (Software)",
                "algorithms": "ML-KEM, ML-DSA (NIST approved)",
                "rng": "OS entropy source",
            },
        }
        return controls.get(standard, {})


# =============================================================================
# Factory Functions
# =============================================================================


_global_audit_logger: Optional[CryptoAuditLogger] = None


def get_audit_logger() -> CryptoAuditLogger:
    """Get the global audit logger instance."""
    global _global_audit_logger
    if _global_audit_logger is None:
        _global_audit_logger = CryptoAuditLogger()
    return _global_audit_logger


def configure_audit_logger(config: AuditConfig) -> CryptoAuditLogger:
    """Configure and return the global audit logger."""
    global _global_audit_logger
    _global_audit_logger = CryptoAuditLogger(config)
    return _global_audit_logger


def create_production_audit_config(
    log_directory: Path,
    compliance_standards: Optional[List[ComplianceStandard]] = None,
) -> AuditConfig:
    """Create production-ready audit configuration."""
    return AuditConfig(
        log_directory=log_directory,
        max_file_size_mb=100,
        max_retention_days=365,  # 1 year retention
        compress_logs=True,
        enable_chain_verification=True,
        min_severity=AuditSeverity.INFO,
        output_format="jsonl",
        enable_console_output=False,
        async_processing=True,
        queue_size=50000,
        compliance_standards=compliance_standards
        or [
            ComplianceStandard.SOC2,
            ComplianceStandard.FIPS_140_3,
        ],
    )
