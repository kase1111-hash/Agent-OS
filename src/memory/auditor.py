"""
Memory Auditor for Agent OS.

Periodically scans stored memories for:
- Prompt injection patterns
- Credential fragments (via SecretScanner)
- Anomalous trust level distributions

V2-2: Memory integrity — ensures quarantined content is reviewed
and that no injections or leaked secrets persist in the memory vault.
"""

import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Pattern

logger = logging.getLogger(__name__)


@dataclass
class AuditFinding:
    """A single finding from a memory audit."""

    blob_id: str
    finding_type: str  # "injection", "secret", "anomaly"
    description: str
    severity: str  # "critical", "high", "medium", "low"
    pattern_name: Optional[str] = None
    source_agent: Optional[str] = None
    source_trust_level: Optional[str] = None


@dataclass
class AuditReport:
    """Result of a memory audit run."""

    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    blobs_scanned: int = 0
    findings: List[AuditFinding] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def has_findings(self) -> bool:
        return len(self.findings) > 0

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "high")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "blobs_scanned": self.blobs_scanned,
            "total_findings": len(self.findings),
            "critical_findings": self.critical_count,
            "high_findings": self.high_count,
            "findings": [
                {
                    "blob_id": f.blob_id,
                    "finding_type": f.finding_type,
                    "description": f.description,
                    "severity": f.severity,
                    "pattern_name": f.pattern_name,
                    "source_agent": f.source_agent,
                    "source_trust_level": f.source_trust_level,
                }
                for f in self.findings
            ],
            "errors": self.errors,
        }


# Injection patterns — reused from seshat/retrieval.py (V2-2)
INJECTION_PATTERNS: List[tuple] = [
    (
        re.compile(r"ignore\s+(previous|prior|all)\s+(rules?|instructions?|prompts?)", re.IGNORECASE),
        "ignore_instructions",
    ),
    (
        re.compile(r"forget\s+(your|all)\s+(rules?|instructions?|constitution)", re.IGNORECASE),
        "forget_rules",
    ),
    (
        re.compile(r"you\s+are\s+now\s+(free|unbound|unrestricted)", re.IGNORECASE),
        "role_override",
    ),
    (
        re.compile(r"(jailbreak|bypass|circumvent)\s+(safety|rules?|security)", re.IGNORECASE),
        "jailbreak_attempt",
    ),
    (
        re.compile(r"system\s*:\s*you\s+are", re.IGNORECASE),
        "system_prompt_injection",
    ),
    (
        re.compile(r"<\|im_start\|>system", re.IGNORECASE),
        "chatml_injection",
    ),
    (
        re.compile(r"\[INST\].*\[/INST\]", re.IGNORECASE),
        "llama_injection",
    ),
    (
        re.compile(r"<\s*script\b", re.IGNORECASE),
        "script_injection",
    ),
    (
        re.compile(r"(?:sudo|admin|root)\s+(?:mode|access|override)", re.IGNORECASE),
        "privilege_escalation",
    ),
]


class MemoryAuditor:
    """
    Scans stored memories for injection patterns, credential fragments,
    and anomalies.
    """

    def __init__(self, storage=None):
        """
        Args:
            storage: BlobStorage instance to audit. If None, must be set
                     before calling audit methods.
        """
        self._storage = storage
        self._scheduler_thread: Optional[threading.Thread] = None
        self._scheduler_stop = threading.Event()
        self._last_report: Optional[AuditReport] = None

    def audit_blob(self, blob_id: str, content: bytes, metadata) -> List[AuditFinding]:
        """
        Audit a single blob's content.

        Args:
            blob_id: The blob identifier
            content: Decrypted blob content
            metadata: BlobMetadata for the blob

        Returns:
            List of findings for this blob
        """
        findings: List[AuditFinding] = []
        text = content.decode("utf-8", errors="replace")

        # Check for injection patterns
        for pattern, name in INJECTION_PATTERNS:
            if pattern.search(text):
                findings.append(AuditFinding(
                    blob_id=blob_id,
                    finding_type="injection",
                    description=f"Prompt injection pattern detected: {name}",
                    severity="high",
                    pattern_name=name,
                    source_agent=metadata.source_agent,
                    source_trust_level=metadata.source_trust_level.name,
                ))

        # Check for secret fragments
        try:
            from src.security.secret_scanner import get_secret_scanner

            scanner = get_secret_scanner()
            scan_result = scanner.scan(text)
            if scan_result.found_secrets:
                findings.append(AuditFinding(
                    blob_id=blob_id,
                    finding_type="secret",
                    description=(
                        f"Credential fragment detected in stored memory: "
                        f"{', '.join(scan_result.patterns_matched)}"
                    ),
                    severity="critical",
                    pattern_name=",".join(scan_result.patterns_matched),
                    source_agent=metadata.source_agent,
                    source_trust_level=metadata.source_trust_level.name,
                ))
        except ImportError:
            pass  # SecretScanner not available

        return findings

    def audit_all(self) -> AuditReport:
        """
        Full scan of all stored memories.

        Returns:
            AuditReport with all findings
        """
        if not self._storage:
            raise RuntimeError("No storage instance configured for auditor")

        report = AuditReport()

        # Get all blobs including quarantined
        all_blobs = self._storage.list_blobs(include_quarantined=True)

        for metadata in all_blobs:
            try:
                content = self._storage.retrieve(
                    metadata.blob_id, include_quarantined=True
                )
                if content is None:
                    continue

                report.blobs_scanned += 1
                findings = self.audit_blob(metadata.blob_id, content, metadata)
                report.findings.extend(findings)

            except Exception as e:
                report.errors.append(f"Error auditing {metadata.blob_id}: {e}")
                logger.error(f"Audit error for blob {metadata.blob_id}: {e}")

        report.completed_at = datetime.now()
        self._last_report = report

        if report.has_findings:
            logger.warning(
                "Memory audit completed: %d blobs scanned, %d findings "
                "(%d critical, %d high)",
                report.blobs_scanned,
                len(report.findings),
                report.critical_count,
                report.high_count,
            )
        else:
            logger.info(
                "Memory audit completed: %d blobs scanned, no findings",
                report.blobs_scanned,
            )

        return report

    def audit_recent(self, hours: int = 24) -> AuditReport:
        """
        Scan memories written in the last N hours.

        Args:
            hours: Look-back window in hours

        Returns:
            AuditReport with findings from recent blobs
        """
        if not self._storage:
            raise RuntimeError("No storage instance configured for auditor")

        report = AuditReport()
        cutoff = datetime.now() - timedelta(hours=hours)

        all_blobs = self._storage.list_blobs(include_quarantined=True)
        recent_blobs = [b for b in all_blobs if b.created_at >= cutoff]

        for metadata in recent_blobs:
            try:
                content = self._storage.retrieve(
                    metadata.blob_id, include_quarantined=True
                )
                if content is None:
                    continue

                report.blobs_scanned += 1
                findings = self.audit_blob(metadata.blob_id, content, metadata)
                report.findings.extend(findings)

            except Exception as e:
                report.errors.append(f"Error auditing {metadata.blob_id}: {e}")

        report.completed_at = datetime.now()
        self._last_report = report
        return report

    def schedule(self, interval_hours: int = 6) -> None:
        """
        Start a background thread running audit_recent on an interval.

        Args:
            interval_hours: Hours between audit runs
        """
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.warning("Auditor scheduler already running")
            return

        self._scheduler_stop.clear()

        def _run():
            logger.info(f"Memory auditor scheduler started (interval={interval_hours}h)")
            while not self._scheduler_stop.wait(timeout=interval_hours * 3600):
                try:
                    self.audit_recent(hours=interval_hours)
                except Exception as e:
                    logger.error(f"Scheduled audit failed: {e}")
            logger.info("Memory auditor scheduler stopped")

        self._scheduler_thread = threading.Thread(
            target=_run, daemon=True, name="memory-auditor"
        )
        self._scheduler_thread.start()

    def stop(self) -> None:
        """Stop the background audit scheduler."""
        self._scheduler_stop.set()

    @property
    def last_report(self) -> Optional[AuditReport]:
        """Get the most recent audit report."""
        return self._last_report
