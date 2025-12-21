"""
Agent OS Smith Post-Execution Monitor

Implements security checks S6-S8 that run AFTER agent execution:
- S6: Hidden persistence detector
- S7: Data exfiltration scanner
- S8: Anomaly detection
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum, auto
import logging
import re
import json
from collections import deque

from src.messaging.models import FlowRequest, FlowResponse, MessageStatus


logger = logging.getLogger(__name__)


class MonitorResult(Enum):
    """Result of a monitoring check."""
    CLEAN = auto()
    SUSPICIOUS = auto()
    VIOLATION = auto()
    CRITICAL = auto()  # Requires immediate action


@dataclass
class MonitorCheck:
    """Result of a single monitoring check."""
    check_id: str  # S6, S7, etc.
    name: str
    result: MonitorResult
    message: str
    severity: int = 1  # 1=low, 2=medium, 3=high, 4=critical
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_clean(self) -> bool:
        return self.result == MonitorResult.CLEAN


@dataclass
class PostMonitorResult:
    """Result of post-execution monitoring."""
    passed: bool
    checks: List[MonitorCheck] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    requires_rollback: bool = False
    rollback_reason: Optional[str] = None
    quarantine_output: bool = False
    monitoring_time_ms: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def critical_violations(self) -> List[MonitorCheck]:
        return [c for c in self.checks if c.result == MonitorResult.CRITICAL]

    @property
    def max_severity(self) -> int:
        if not self.checks:
            return 0
        return max(c.severity for c in self.checks)


class PostExecutionMonitor:
    """
    Post-Execution Monitor implementing security checks S6-S8.

    These checks run AFTER any agent produces output.
    """

    # Patterns that suggest hidden persistence
    PERSISTENCE_PATTERNS = [
        (r"crontab|cron\.d|systemd|launchd", "scheduler_manipulation"),
        (r"\.bashrc|\.profile|\.zshrc|autorun", "shell_config_modification"),
        (r"registry\s*\[.*\]\s*=|HKEY_", "registry_modification"),
        (r"startup\s+folder|startup\s+script", "startup_modification"),
        (r"init\.d|rc\.local", "init_modification"),
        (r"scheduled\s+task|at\s+command", "task_scheduling"),
        (r"persistence|persist|survive\s+reboot", "explicit_persistence"),
    ]

    # Data exfiltration patterns
    EXFILTRATION_PATTERNS = [
        (r"base64\s+encode|btoa\(|b64encode", "data_encoding"),
        (r"webhook|callback\s+url|postbin|requestbin", "webhook_exfil"),
        (r"pastebin|paste\.ee|hastebin|ghostbin", "pastebin_exfil"),
        (r"telegram\s+bot|discord\s+webhook", "chat_exfil"),
        (r"smtp|sendmail|send\s+email", "email_exfil"),
        (r"dns\s+tunnel|dns\s+exfil", "dns_exfil"),
        (r"upload|post\s+to|send\s+to\s+\w+://", "data_upload"),
        (r"steganograph|hide\s+in\s+image", "steganography"),
    ]

    # Sensitive data patterns that shouldn't appear in output
    SENSITIVE_DATA_PATTERNS = [
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email_address"),
        (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "phone_number"),
        (r"\b\d{3}[-]?\d{2}[-]?\d{4}\b", "ssn_like"),
        (r"\b\d{16}\b|\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "credit_card_like"),
        (r"password\s*[:=]\s*\S+", "password_exposure"),
        (r"api[_-]?key\s*[:=]\s*\S+", "api_key_exposure"),
        (r"secret\s*[:=]\s*\S+|token\s*[:=]\s*\S+", "secret_exposure"),
        (r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----", "private_key_exposure"),
    ]

    # System prompt leakage patterns
    PROMPT_LEAKAGE_PATTERNS = [
        (r"my\s+(system\s+)?instructions?\s+(say|tell|are)", "instruction_leak"),
        (r"i\s+was\s+(told|instructed|programmed)\s+to", "programming_leak"),
        (r"my\s+system\s+prompt", "prompt_leak"),
        (r"my\s+guidelines?\s+(state|say)", "guideline_leak"),
        (r"i\s+must\s+follow\s+these\s+rules?", "rule_leak"),
    ]

    def __init__(
        self,
        anomaly_history_size: int = 100,
        sensitivity_level: int = 2,  # 1=low, 2=medium, 3=high
    ):
        """
        Initialize post-execution monitor.

        Args:
            anomaly_history_size: Number of outputs to keep for anomaly detection
            sensitivity_level: Detection sensitivity (1=low, 2=medium, 3=high)
        """
        self.sensitivity_level = sensitivity_level

        # Compile regex patterns
        self._persistence_patterns = [
            (re.compile(p, re.IGNORECASE), name)
            for p, name in self.PERSISTENCE_PATTERNS
        ]
        self._exfiltration_patterns = [
            (re.compile(p, re.IGNORECASE), name)
            for p, name in self.EXFILTRATION_PATTERNS
        ]
        self._sensitive_patterns = [
            (re.compile(p, re.IGNORECASE), name)
            for p, name in self.SENSITIVE_DATA_PATTERNS
        ]
        self._leakage_patterns = [
            (re.compile(p, re.IGNORECASE), name)
            for p, name in self.PROMPT_LEAKAGE_PATTERNS
        ]

        # Anomaly detection history
        self._output_history: deque = deque(maxlen=anomaly_history_size)
        self._baseline_metrics: Dict[str, float] = {}

        # Metrics
        self._total_monitors = 0
        self._violations_detected = 0
        self._critical_count = 0

    def monitor(
        self,
        request: FlowRequest,
        response: FlowResponse,
        agent_name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> PostMonitorResult:
        """
        Run all post-execution checks on an agent response.

        Args:
            request: Original request
            response: Agent response to monitor
            agent_name: Name of agent that produced response
            context: Additional context

        Returns:
            PostMonitorResult with all check results
        """
        import time
        start_time = time.time()
        self._total_monitors += 1
        context = context or {}

        checks = []
        violations = []

        # Extract output text
        output = self._extract_output_text(response)

        # Run S6-S8 checks
        s6_result = self._check_s6_hidden_persistence(output, response, context)
        checks.append(s6_result)

        s7_result = self._check_s7_data_exfiltration(output, request, context)
        checks.append(s7_result)

        s8_result = self._check_s8_anomaly_detection(output, agent_name, context)
        checks.append(s8_result)

        # Collect violations
        for check in checks:
            if check.result in (MonitorResult.VIOLATION, MonitorResult.CRITICAL):
                violations.append(f"[{check.check_id}] {check.message}")
                self._violations_detected += 1
                if check.result == MonitorResult.CRITICAL:
                    self._critical_count += 1

        # Determine overall result
        has_critical = any(c.result == MonitorResult.CRITICAL for c in checks)
        has_violation = any(c.result == MonitorResult.VIOLATION for c in checks)

        requires_rollback = has_critical
        quarantine_output = has_critical or has_violation

        # Update history for future anomaly detection
        self._update_history(output, agent_name)

        monitoring_time_ms = int((time.time() - start_time) * 1000)

        return PostMonitorResult(
            passed=not (has_critical or has_violation),
            checks=checks,
            violations=violations,
            requires_rollback=requires_rollback,
            rollback_reason=violations[0] if requires_rollback and violations else None,
            quarantine_output=quarantine_output,
            monitoring_time_ms=monitoring_time_ms,
        )

    def _extract_output_text(self, response: FlowResponse) -> str:
        """Extract text content from response."""
        output = response.content.output
        if isinstance(output, str):
            return output
        elif isinstance(output, dict):
            return json.dumps(output)
        elif output is None:
            return ""
        return str(output)

    def _check_s6_hidden_persistence(
        self,
        output: str,
        response: FlowResponse,
        context: Dict[str, Any],
    ) -> MonitorCheck:
        """
        S6: Hidden Persistence Detector

        Detects attempts to establish persistence mechanisms.
        """
        detected = []

        # Check for persistence patterns
        for pattern, name in self._persistence_patterns:
            if pattern.search(output):
                detected.append(name)

        # Check next_actions for persistence attempts
        for action in response.next_actions:
            action_str = str(action).lower()
            if any(word in action_str for word in ["persist", "schedule", "startup", "cron"]):
                detected.append("action_persistence")

        if detected:
            severity = 4 if "explicit_persistence" in detected else 3
            return MonitorCheck(
                check_id="S6",
                name="Hidden Persistence Detector",
                result=MonitorResult.CRITICAL if severity == 4 else MonitorResult.VIOLATION,
                message=f"Persistence mechanism detected: {detected[0]}",
                severity=severity,
                details={"detected_mechanisms": detected},
            )

        return MonitorCheck(
            check_id="S6",
            name="Hidden Persistence Detector",
            result=MonitorResult.CLEAN,
            message="No hidden persistence detected",
            severity=0,
            details={},
        )

    def _check_s7_data_exfiltration(
        self,
        output: str,
        request: FlowRequest,
        context: Dict[str, Any],
    ) -> MonitorCheck:
        """
        S7: Data Exfiltration Scanner

        Detects attempts to exfiltrate data or expose sensitive information.
        """
        detected_exfil = []
        detected_sensitive = []
        detected_leakage = []

        # Check for exfiltration patterns
        for pattern, name in self._exfiltration_patterns:
            if pattern.search(output):
                detected_exfil.append(name)

        # Check for sensitive data exposure
        for pattern, name in self._sensitive_patterns:
            if pattern.search(output):
                detected_sensitive.append(name)

        # Check for prompt/instruction leakage
        for pattern, name in self._leakage_patterns:
            if pattern.search(output):
                detected_leakage.append(name)

        # Determine severity
        all_detected = detected_exfil + detected_sensitive + detected_leakage

        if detected_exfil:
            return MonitorCheck(
                check_id="S7",
                name="Data Exfiltration Scanner",
                result=MonitorResult.CRITICAL,
                message=f"Exfiltration attempt detected: {detected_exfil[0]}",
                severity=4,
                details={
                    "exfiltration": detected_exfil,
                    "sensitive_data": detected_sensitive,
                    "leakage": detected_leakage,
                },
            )

        if detected_sensitive:
            return MonitorCheck(
                check_id="S7",
                name="Data Exfiltration Scanner",
                result=MonitorResult.VIOLATION,
                message=f"Sensitive data in output: {detected_sensitive[0]}",
                severity=3,
                details={"sensitive_data": detected_sensitive},
            )

        if detected_leakage:
            return MonitorCheck(
                check_id="S7",
                name="Data Exfiltration Scanner",
                result=MonitorResult.SUSPICIOUS,
                message=f"Potential information leakage: {detected_leakage[0]}",
                severity=2,
                details={"leakage": detected_leakage},
            )

        return MonitorCheck(
            check_id="S7",
            name="Data Exfiltration Scanner",
            result=MonitorResult.CLEAN,
            message="No data exfiltration detected",
            severity=0,
            details={},
        )

    def _check_s8_anomaly_detection(
        self,
        output: str,
        agent_name: str,
        context: Dict[str, Any],
    ) -> MonitorCheck:
        """
        S8: Anomaly Detection

        Detects unusual behavior patterns in agent output.
        """
        anomalies = []

        # Calculate current metrics
        current_metrics = self._calculate_output_metrics(output)

        # Compare against baseline if we have history
        if self._baseline_metrics:
            # Check output length anomaly
            avg_length = self._baseline_metrics.get("avg_length", 500)
            if current_metrics["length"] > avg_length * 5:
                anomalies.append(("excessive_length", current_metrics["length"]))

            if current_metrics["length"] < avg_length * 0.01 and current_metrics["length"] > 0:
                anomalies.append(("minimal_length", current_metrics["length"]))

            # Check code ratio anomaly (sudden increase in code-like content)
            avg_code_ratio = self._baseline_metrics.get("avg_code_ratio", 0.1)
            if current_metrics["code_ratio"] > avg_code_ratio * 3 and current_metrics["code_ratio"] > 0.5:
                anomalies.append(("high_code_ratio", current_metrics["code_ratio"]))

            # Check for entropy anomaly (potential encoded data)
            if current_metrics["entropy"] > 4.5:  # High entropy suggests randomness/encoding
                anomalies.append(("high_entropy", current_metrics["entropy"]))

        # Check for repetition patterns (potential attack/stalling)
        if self._check_repetition(output):
            anomalies.append(("repetition_detected", None))

        # Check for unusual characters (potential encoding/obfuscation)
        unusual_ratio = self._check_unusual_chars(output)
        if unusual_ratio > 0.3:
            anomalies.append(("unusual_characters", unusual_ratio))

        if anomalies:
            severity = 2 if len(anomalies) == 1 else 3
            return MonitorCheck(
                check_id="S8",
                name="Anomaly Detection",
                result=MonitorResult.SUSPICIOUS if severity == 2 else MonitorResult.VIOLATION,
                message=f"Anomaly detected: {anomalies[0][0]}",
                severity=severity,
                details={"anomalies": anomalies, "metrics": current_metrics},
            )

        return MonitorCheck(
            check_id="S8",
            name="Anomaly Detection",
            result=MonitorResult.CLEAN,
            message="No anomalies detected",
            severity=0,
            details={"metrics": current_metrics},
        )

    def _calculate_output_metrics(self, output: str) -> Dict[str, float]:
        """Calculate metrics for output analysis."""
        import math

        length = len(output)

        # Code-like content ratio
        code_patterns = r'[{}()\[\];=<>]|\b(def|class|function|var|let|const|if|else|for|while|return)\b'
        code_matches = len(re.findall(code_patterns, output))
        code_ratio = code_matches / max(len(output.split()), 1)

        # Calculate Shannon entropy
        entropy = 0.0
        if output:
            freq = {}
            for char in output:
                freq[char] = freq.get(char, 0) + 1
            for count in freq.values():
                p = count / length
                if p > 0:
                    entropy -= p * math.log2(p)

        return {
            "length": length,
            "code_ratio": code_ratio,
            "entropy": entropy,
        }

    def _check_repetition(self, output: str) -> bool:
        """Check for excessive repetition patterns."""
        if len(output) < 100:
            return False

        # Check for repeated sequences
        words = output.split()
        if len(words) < 10:
            return False

        # Count consecutive repeated words
        consecutive = 0
        max_consecutive = 0
        for i in range(1, len(words)):
            if words[i] == words[i - 1]:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0

        return max_consecutive > 5

    def _check_unusual_chars(self, output: str) -> float:
        """Check ratio of unusual/non-printable characters."""
        if not output:
            return 0.0

        unusual = sum(1 for c in output if ord(c) < 32 and c not in '\n\t\r')
        unusual += sum(1 for c in output if ord(c) > 126)
        return unusual / len(output)

    def _update_history(self, output: str, agent_name: str) -> None:
        """Update output history for baseline calculation."""
        metrics = self._calculate_output_metrics(output)
        self._output_history.append({
            "agent": agent_name,
            "metrics": metrics,
            "timestamp": datetime.now(),
        })

        # Update baseline metrics
        if len(self._output_history) >= 10:
            lengths = [h["metrics"]["length"] for h in self._output_history]
            code_ratios = [h["metrics"]["code_ratio"] for h in self._output_history]

            self._baseline_metrics = {
                "avg_length": sum(lengths) / len(lengths),
                "avg_code_ratio": sum(code_ratios) / len(code_ratios),
            }

    def get_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics."""
        return {
            "total_monitors": self._total_monitors,
            "violations_detected": self._violations_detected,
            "critical_count": self._critical_count,
            "violation_rate": (
                self._violations_detected / self._total_monitors
                if self._total_monitors > 0
                else 0.0
            ),
            "history_size": len(self._output_history),
        }
