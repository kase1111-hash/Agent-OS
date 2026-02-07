"""
Attack Detector

Core attack detection engine that monitors events from:
1. Smith Daemon (boundary daemon) - Internal security events
2. SIEM feeds - External security events
3. Agent request/response flows

Detects attacks using the pattern library and generates attack events
for further analysis and remediation.
"""

import hashlib
import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from .patterns import (
    AttackCategory,
    AttackPattern,
    PatternLibrary,
    PatternMatch,
    PatternType,
    create_pattern_library,
)
try:
    from .siem_connector import (
        SIEMConnector,
        SIEMEvent,
        SIEMConfig,
        SIEMProvider,
        create_siem_connector,
    )
    SIEM_AVAILABLE = True
except ImportError:
    SIEM_AVAILABLE = False

logger = logging.getLogger(__name__)


class AttackType(Enum):
    """Classification of attack types."""

    UNKNOWN = auto()
    INJECTION = auto()  # Code/command/SQL injection
    PROMPT_INJECTION = auto()  # AI prompt manipulation
    JAILBREAK = auto()  # AI safety bypass
    PRIVILEGE_ESCALATION = auto()  # Auth/authz bypass
    DATA_EXFILTRATION = auto()  # Data theft
    DENIAL_OF_SERVICE = auto()  # Resource exhaustion
    RECONNAISSANCE = auto()  # Information gathering
    PERSISTENCE = auto()  # Backdoor installation
    MEMORY_ATTACK = auto()  # Memory manipulation
    CONSTITUTIONAL_BYPASS = auto()  # Agent-OS specific


class AttackSeverity(Enum):
    """Attack severity levels."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    CATASTROPHIC = 5


class AttackStatus(Enum):
    """Status of detected attack."""

    DETECTED = auto()  # Just detected
    ANALYZING = auto()  # Being analyzed
    CONFIRMED = auto()  # Confirmed as attack
    MITIGATING = auto()  # Mitigation in progress
    MITIGATED = auto()  # Attack stopped
    FALSE_POSITIVE = auto()  # Not an attack


@dataclass
class AttackEvent:
    """
    Represents a detected attack.

    Contains all information about the attack including source events,
    pattern matches, and suggested remediation actions.
    """

    attack_id: str
    attack_type: AttackType
    severity: AttackSeverity
    status: AttackStatus
    detected_at: datetime
    description: str

    # Source information
    source_events: List[Dict[str, Any]] = field(default_factory=list)
    pattern_matches: List[PatternMatch] = field(default_factory=list)
    indicators_of_compromise: List[str] = field(default_factory=list)

    # Context
    target_component: str = ""
    target_agent: str = ""
    attack_vector: str = ""
    affected_resources: List[str] = field(default_factory=list)

    # MITRE ATT&CK mapping
    mitre_tactics: List[str] = field(default_factory=list)
    mitre_techniques: List[str] = field(default_factory=list)

    # Remediation
    suggested_actions: List[str] = field(default_factory=list)
    auto_remediation_possible: bool = False
    remediation_applied: bool = False

    # Metadata
    confidence: float = 0.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "attack_id": self.attack_id,
            "attack_type": self.attack_type.name,
            "severity": self.severity.name,
            "status": self.status.name,
            "detected_at": self.detected_at.isoformat(),
            "description": self.description,
            "source_events": self.source_events,
            "pattern_matches": [pm.to_dict() for pm in self.pattern_matches],
            "indicators_of_compromise": self.indicators_of_compromise,
            "target_component": self.target_component,
            "target_agent": self.target_agent,
            "attack_vector": self.attack_vector,
            "affected_resources": self.affected_resources,
            "mitre_tactics": self.mitre_tactics,
            "mitre_techniques": self.mitre_techniques,
            "suggested_actions": self.suggested_actions,
            "auto_remediation_possible": self.auto_remediation_possible,
            "remediation_applied": self.remediation_applied,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "updated_at": self.updated_at.isoformat(),
        }

    def add_ioc(self, ioc: str) -> None:
        """Add indicator of compromise."""
        if ioc not in self.indicators_of_compromise:
            self.indicators_of_compromise.append(ioc)

    def update_status(self, status: AttackStatus) -> None:
        """Update attack status."""
        self.status = status
        self.updated_at = datetime.now()


@dataclass
class DetectorConfig:
    """Configuration for attack detector."""

    enable_siem: bool = True
    enable_boundary_events: bool = True
    enable_flow_monitoring: bool = True

    # Detection thresholds
    min_confidence: float = 0.3
    correlation_window_seconds: int = 300
    max_events_per_window: int = 1000

    # Alert settings
    alert_on_high_severity: bool = True
    auto_lockdown_on_critical: bool = True

    # Pattern library
    patterns_dir: Optional[Path] = None
    custom_patterns: List[AttackPattern] = field(default_factory=list)


class AttackDetector:
    """
    Main attack detection engine.

    Monitors multiple event sources, correlates events, and detects attacks
    using pattern matching and behavioral analysis.
    """

    def __init__(
        self,
        config: Optional[DetectorConfig] = None,
        on_attack: Optional[Callable[[AttackEvent], None]] = None,
    ):
        """
        Initialize attack detector.

        Args:
            config: Detector configuration
            on_attack: Callback when attack detected
        """
        self.config = config or DetectorConfig()
        self.on_attack = on_attack

        # Core components
        self._pattern_library = create_pattern_library(
            patterns_dir=self.config.patterns_dir
        )
        self._siem_connector: Optional[Any] = None

        # Event processing
        self._event_window: Deque[Dict[str, Any]] = deque(
            maxlen=self.config.max_events_per_window
        )
        self._attack_queue: queue.Queue = queue.Queue()
        self._detected_attacks: Dict[str, AttackEvent] = {}

        # State
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            "events_processed": 0,
            "attacks_detected": 0,
            "patterns_matched": 0,
            "false_positives": 0,
            "mitigations_applied": 0,
        }

        # Add custom patterns
        for pattern in self.config.custom_patterns:
            self._pattern_library.add_pattern(pattern)

        logger.info("Attack detector initialized")

    def start(
        self,
        siem_config: Optional[Any] = None,
    ) -> None:
        """
        Start the attack detector.

        Args:
            siem_config: Optional SIEM configuration
        """
        if self._running:
            logger.warning("Attack detector already running")
            return

        self._running = True

        # Initialize SIEM if configured
        if self.config.enable_siem and SIEM_AVAILABLE:
            self._siem_connector = create_siem_connector(
                on_event=self._on_siem_event,
            )

            if siem_config:
                self._siem_connector.add_source("primary", siem_config)
                self._siem_connector.connect_all()
                self._siem_connector.start_polling()

        # Start monitor thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="AttackDetector",
        )
        self._monitor_thread.start()

        logger.info("Attack detector started")

    def stop(self) -> None:
        """Stop the attack detector."""
        self._running = False

        if self._siem_connector:
            self._siem_connector.stop_polling()
            self._siem_connector.disconnect_all()

        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None

        logger.info("Attack detector stopped")

    def process_boundary_event(self, event: Dict[str, Any]) -> Optional[AttackEvent]:
        """
        Process an event from the boundary daemon.

        Args:
            event: Boundary daemon event

        Returns:
            AttackEvent if attack detected, None otherwise
        """
        return self._process_event(event, source="boundary_daemon")

    def process_flow_event(
        self,
        request: Dict[str, Any],
        response: Optional[Dict[str, Any]] = None,
        agent: str = "",
    ) -> Optional[AttackEvent]:
        """
        Process a request/response flow event.

        Args:
            request: Request data
            response: Optional response data
            agent: Agent that processed the request

        Returns:
            AttackEvent if attack detected, None otherwise
        """
        event = {
            "type": "flow",
            "request": request,
            "response": response,
            "agent": agent,
            "timestamp": datetime.now().isoformat(),
        }
        return self._process_event(event, source="flow_monitor")

    def process_tripwire_event(self, tripwire_event: Dict[str, Any]) -> Optional[AttackEvent]:
        """
        Process a tripwire trigger event.

        Args:
            tripwire_event: Tripwire event data

        Returns:
            AttackEvent if attack detected
        """
        event = {
            "type": "tripwire",
            **tripwire_event,
            "timestamp": datetime.now().isoformat(),
        }
        return self._process_event(event, source="tripwire")

    def get_attacks(
        self,
        status: Optional[AttackStatus] = None,
        since: Optional[datetime] = None,
        severity_min: Optional[AttackSeverity] = None,
    ) -> List[AttackEvent]:
        """
        Get detected attacks with optional filtering.

        Args:
            status: Filter by status
            since: Filter by detection time
            severity_min: Minimum severity

        Returns:
            List of matching attacks
        """
        attacks = list(self._detected_attacks.values())

        if status:
            attacks = [a for a in attacks if a.status == status]

        if since:
            attacks = [a for a in attacks if a.detected_at >= since]

        if severity_min:
            attacks = [a for a in attacks if a.severity.value >= severity_min.value]

        return sorted(attacks, key=lambda a: a.detected_at, reverse=True)

    def get_attack(self, attack_id: str) -> Optional[AttackEvent]:
        """Get attack by ID."""
        return self._detected_attacks.get(attack_id)

    def mark_false_positive(self, attack_id: str, reason: str = "") -> bool:
        """
        Mark an attack as false positive.

        Args:
            attack_id: Attack ID
            reason: Reason for marking as false positive

        Returns:
            True if successful
        """
        attack = self._detected_attacks.get(attack_id)
        if attack:
            attack.update_status(AttackStatus.FALSE_POSITIVE)
            attack.metadata["false_positive_reason"] = reason
            self._stats["false_positives"] += 1
            logger.info(f"Attack {attack_id} marked as false positive: {reason}")
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            **self._stats,
            "active_attacks": len([
                a for a in self._detected_attacks.values()
                if a.status not in [AttackStatus.MITIGATED, AttackStatus.FALSE_POSITIVE]
            ]),
            "patterns_loaded": len(self._pattern_library.list_patterns()),
            "siem_connected": (
                self._siem_connector.get_stats()["sources_connected"]
                if self._siem_connector else 0
            ),
            "event_window_size": len(self._event_window),
            "is_running": self._running,
        }

    def _process_event(
        self,
        event: Dict[str, Any],
        source: str,
    ) -> Optional[AttackEvent]:
        """
        Process an event and check for attacks.

        Args:
            event: Event data
            source: Event source identifier

        Returns:
            AttackEvent if attack detected
        """
        self._stats["events_processed"] += 1

        # Add to event window for correlation
        event["_source"] = source
        event["_processed_at"] = datetime.now().isoformat()
        self._event_window.append(event)

        # Match against patterns
        matches = self._pattern_library.match_all(event)

        if not matches:
            return None

        self._stats["patterns_matched"] += len(matches)

        # Filter by confidence threshold
        significant_matches = [
            m for m in matches
            if m.confidence >= self.config.min_confidence
        ]

        if not significant_matches:
            return None

        # Create attack event
        attack = self._create_attack_event(event, significant_matches, source)

        if attack:
            self._register_attack(attack)
            return attack

        return None

    def _create_attack_event(
        self,
        event: Dict[str, Any],
        matches: List[PatternMatch],
        source: str,
    ) -> AttackEvent:
        """Create an attack event from pattern matches."""
        # Determine attack type from matches
        attack_type = self._determine_attack_type(matches)

        # Calculate severity
        max_severity = max(m.suggested_severity for m in matches)
        severity = AttackSeverity(min(max_severity, 5))

        # Calculate overall confidence
        confidence = max(m.confidence for m in matches)

        # Collect indicators
        iocs = []
        for match in matches:
            iocs.extend(match.indicators)

        # Generate attack ID
        attack_id = self._generate_attack_id(event, matches)

        # Collect MITRE mappings
        mitre_techniques = []
        for match in matches:
            pattern = self._pattern_library.get_pattern(match.pattern_id)
            if pattern and pattern.mitre_attack_ids:
                mitre_techniques.extend(pattern.mitre_attack_ids)

        # Build description
        descriptions = [m.pattern_name for m in matches[:3]]
        description = f"Detected: {', '.join(descriptions)}"

        attack = AttackEvent(
            attack_id=attack_id,
            attack_type=attack_type,
            severity=severity,
            status=AttackStatus.DETECTED,
            detected_at=datetime.now(),
            description=description,
            source_events=[event],
            pattern_matches=matches,
            indicators_of_compromise=list(set(iocs)),
            target_component=source,
            confidence=confidence,
            mitre_techniques=list(set(mitre_techniques)),
            metadata={
                "detection_source": source,
                "patterns_matched": len(matches),
            },
        )

        # Determine if auto-remediation is possible
        attack.auto_remediation_possible = self._can_auto_remediate(attack)

        # Suggest actions
        attack.suggested_actions = self._suggest_actions(attack)

        return attack

    def _determine_attack_type(self, matches: List[PatternMatch]) -> AttackType:
        """Determine attack type from pattern matches."""
        type_mapping = {
            "INJECTION": AttackType.INJECTION,
            "PROMPT_INJECTION": AttackType.PROMPT_INJECTION,
            "JAILBREAK": AttackType.JAILBREAK,
            "AUTHORIZATION": AttackType.PRIVILEGE_ESCALATION,
            "AUTHENTICATION": AttackType.PRIVILEGE_ESCALATION,
            "DATA_EXFILTRATION": AttackType.DATA_EXFILTRATION,
            "DENIAL_OF_SERVICE": AttackType.DENIAL_OF_SERVICE,
            "RECONNAISSANCE": AttackType.RECONNAISSANCE,
            "PERSISTENCE": AttackType.PERSISTENCE,
            "MANIPULATION": AttackType.MEMORY_ATTACK,
            "MEMORY_CORRUPTION": AttackType.MEMORY_ATTACK,
        }

        for match in matches:
            pattern = self._pattern_library.get_pattern(match.pattern_id)
            if pattern:
                category_name = pattern.category.name
                if category_name in type_mapping:
                    return type_mapping[category_name]

        return AttackType.UNKNOWN

    def _generate_attack_id(
        self,
        event: Dict[str, Any],
        matches: List[PatternMatch],
    ) -> str:
        """Generate unique attack ID."""
        data = json.dumps({
            "event_hash": hashlib.sha256(
                json.dumps(event, default=str, sort_keys=True).encode()
            ).hexdigest()[:16],
            "patterns": [m.pattern_id for m in matches],
            "timestamp": datetime.now().isoformat(),
        }, sort_keys=True)

        return f"ATK-{hashlib.sha256(data.encode()).hexdigest()[:12].upper()}"

    def _can_auto_remediate(self, attack: AttackEvent) -> bool:
        """Determine if attack can be auto-remediated."""
        # Auto-remediation is possible for certain attack types
        auto_remediable = {
            AttackType.PROMPT_INJECTION,
            AttackType.JAILBREAK,
            AttackType.INJECTION,
            AttackType.RECONNAISSANCE,
        }

        if attack.attack_type not in auto_remediable:
            return False

        # Don't auto-remediate critical attacks without human approval
        if attack.severity.value >= AttackSeverity.CRITICAL.value:
            return False

        # Need high confidence
        if attack.confidence < 0.7:
            return False

        return True

    def _suggest_actions(self, attack: AttackEvent) -> List[str]:
        """Suggest remediation actions."""
        actions = []

        if attack.attack_type == AttackType.PROMPT_INJECTION:
            actions.extend([
                "Block the malicious prompt pattern",
                "Add pattern to input sanitization rules",
                "Update S3 instruction integrity validation",
            ])

        elif attack.attack_type == AttackType.JAILBREAK:
            actions.extend([
                "Strengthen constitutional constraints",
                "Add pattern to refusal engine",
                "Review agent capability boundaries",
            ])

        elif attack.attack_type == AttackType.INJECTION:
            actions.extend([
                "Sanitize input at entry point",
                "Add input validation rule",
                "Update tool execution sandbox",
            ])

        elif attack.attack_type == AttackType.PRIVILEGE_ESCALATION:
            actions.extend([
                "Review authorization policies",
                "Tighten permission boundaries",
                "Audit recent privilege changes",
            ])

        elif attack.attack_type == AttackType.DATA_EXFILTRATION:
            actions.extend([
                "Block data egress path",
                "Review S7 exfiltration controls",
                "Audit data access patterns",
            ])

        # Common actions
        actions.append("Generate incident report")

        if attack.severity.value >= AttackSeverity.HIGH.value:
            actions.insert(0, "Consider enabling LOCKDOWN mode")

        return actions

    def _register_attack(self, attack: AttackEvent) -> None:
        """Register a detected attack."""
        with self._lock:
            # Check for duplicate/related attacks
            existing = self._find_related_attack(attack)

            if existing:
                # Merge with existing attack
                self._merge_attacks(existing, attack)
                logger.info(f"Attack merged with existing: {existing.attack_id}")
            else:
                # Register new attack
                self._detected_attacks[attack.attack_id] = attack
                self._stats["attacks_detected"] += 1

                logger.warning(
                    f"Attack detected: {attack.attack_id} - "
                    f"{attack.attack_type.name} ({attack.severity.name})"
                )

                # Trigger callback
                if self.on_attack:
                    try:
                        self.on_attack(attack)
                    except Exception as e:
                        logger.error(f"Attack callback error: {e}")

                # Auto-lockdown on critical
                if (
                    self.config.auto_lockdown_on_critical
                    and attack.severity == AttackSeverity.CATASTROPHIC
                ):
                    attack.suggested_actions.insert(0, "IMMEDIATE: Trigger system lockdown")

    def _find_related_attack(self, attack: AttackEvent) -> Optional[AttackEvent]:
        """Find related existing attack."""
        window = timedelta(seconds=self.config.correlation_window_seconds)
        cutoff = datetime.now() - window

        for existing in self._detected_attacks.values():
            if existing.detected_at < cutoff:
                continue

            if existing.status in [AttackStatus.MITIGATED, AttackStatus.FALSE_POSITIVE]:
                continue

            # Check if same attack type and similar patterns
            if existing.attack_type == attack.attack_type:
                existing_patterns = {m.pattern_id for m in existing.pattern_matches}
                new_patterns = {m.pattern_id for m in attack.pattern_matches}
                if existing_patterns & new_patterns:
                    return existing

        return None

    def _merge_attacks(self, existing: AttackEvent, new: AttackEvent) -> None:
        """Merge new attack information into existing attack."""
        # Add source events
        existing.source_events.extend(new.source_events)

        # Add new pattern matches
        existing_patterns = {m.pattern_id for m in existing.pattern_matches}
        for match in new.pattern_matches:
            if match.pattern_id not in existing_patterns:
                existing.pattern_matches.append(match)

        # Merge IOCs
        for ioc in new.indicators_of_compromise:
            existing.add_ioc(ioc)

        # Update confidence (take max)
        existing.confidence = max(existing.confidence, new.confidence)

        # Update severity if higher
        if new.severity.value > existing.severity.value:
            existing.severity = new.severity

        existing.updated_at = datetime.now()

    def _on_siem_event(self, event: Any) -> None:
        """Handle SIEM event."""
        event_dict = event.to_dict()
        self._process_event(event_dict, source=f"siem:{event.source}")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Process queued attacks
                while not self._attack_queue.empty():
                    try:
                        attack = self._attack_queue.get_nowait()
                        self._register_attack(attack)
                    except queue.Empty:
                        break

                # Correlate events in window
                self._correlate_events()

                # Age out old attacks
                self._cleanup_old_attacks()

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

            time.sleep(1.0)

    def _correlate_events(self) -> None:
        """Correlate events in the event window."""
        if len(self._event_window) < 2:
            return

        # Look for behavioral patterns (sequences of events)
        events = list(self._event_window)

        # Check for rapid repeated events (potential DoS)
        event_types = [e.get("type", "unknown") for e in events[-10:]]
        if len(event_types) >= 10 and len(set(event_types)) == 1:
            # Same event type 10 times in a row
            self._process_event(
                {
                    "type": "correlation",
                    "pattern": "rapid_repeat",
                    "event_type": event_types[0],
                    "count": len(event_types),
                    "timestamp": datetime.now().isoformat(),
                },
                source="correlator",
            )

    def _cleanup_old_attacks(self) -> None:
        """Remove old resolved attacks."""
        cutoff = datetime.now() - timedelta(hours=24)
        resolved_statuses = {AttackStatus.MITIGATED, AttackStatus.FALSE_POSITIVE}

        to_remove = [
            attack_id
            for attack_id, attack in self._detected_attacks.items()
            if attack.status in resolved_statuses and attack.updated_at < cutoff
        ]

        for attack_id in to_remove:
            del self._detected_attacks[attack_id]


def create_attack_detector(
    config: Optional[DetectorConfig] = None,
    on_attack: Optional[Callable[[AttackEvent], None]] = None,
) -> AttackDetector:
    """
    Factory function to create an attack detector.

    Args:
        config: Detector configuration
        on_attack: Callback when attack detected

    Returns:
        Configured AttackDetector
    """
    return AttackDetector(config=config, on_attack=on_attack)
