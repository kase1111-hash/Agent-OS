"""
Attack Pattern Library

Defines known attack patterns that Agent Smith can detect.
Patterns are matched against events from the boundary daemon and SIEM feeds.

Pattern Types:
- Signature-based: Exact matching of known attack signatures
- Behavioral: Detection based on sequences of suspicious actions
- Anomaly: Statistical deviation from normal baseline
- Heuristic: Rule-based detection of attack characteristics
"""

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of attack patterns."""

    SIGNATURE = auto()  # Exact signature match
    BEHAVIORAL = auto()  # Sequence of events
    ANOMALY = auto()  # Statistical anomaly
    HEURISTIC = auto()  # Rule-based detection
    COMPOSITE = auto()  # Combination of patterns


class AttackCategory(Enum):
    """Categories of attacks."""

    INJECTION = auto()  # SQL, command, code injection
    AUTHENTICATION = auto()  # Auth bypass, credential stuffing
    AUTHORIZATION = auto()  # Privilege escalation, access control bypass
    DATA_EXFILTRATION = auto()  # Data theft, unauthorized access
    DENIAL_OF_SERVICE = auto()  # DoS, resource exhaustion
    MANIPULATION = auto()  # Data tampering, integrity attacks
    RECONNAISSANCE = auto()  # Scanning, enumeration
    PERSISTENCE = auto()  # Backdoors, rootkits
    PROMPT_INJECTION = auto()  # AI-specific prompt manipulation
    JAILBREAK = auto()  # AI safety bypass attempts
    MEMORY_CORRUPTION = auto()  # Buffer overflow, use-after-free
    SUPPLY_CHAIN = auto()  # Dependency attacks


@dataclass
class PatternMatch:
    """Result of a pattern match."""

    pattern_id: str
    pattern_name: str
    confidence: float  # 0.0 to 1.0
    matched_at: datetime = field(default_factory=datetime.now)
    matched_data: Dict[str, Any] = field(default_factory=dict)
    indicators: List[str] = field(default_factory=list)
    suggested_severity: int = 3  # 1-5 scale

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_name": self.pattern_name,
            "confidence": self.confidence,
            "matched_at": self.matched_at.isoformat(),
            "matched_data": self.matched_data,
            "indicators": self.indicators,
            "suggested_severity": self.suggested_severity,
        }


@dataclass
class AttackPattern:
    """Definition of an attack pattern."""

    id: str
    name: str
    description: str
    pattern_type: PatternType
    category: AttackCategory
    severity: int = 3  # 1-5, 5 being most severe
    enabled: bool = True

    # Pattern matching criteria
    signatures: List[str] = field(default_factory=list)  # Regex patterns
    keywords: List[str] = field(default_factory=list)  # Simple keyword matches
    event_sequence: List[str] = field(default_factory=list)  # For behavioral patterns
    thresholds: Dict[str, float] = field(default_factory=dict)  # For anomaly patterns
    rules: List[Dict[str, Any]] = field(default_factory=list)  # For heuristic patterns

    # Metadata
    mitre_attack_ids: List[str] = field(default_factory=list)  # MITRE ATT&CK mapping
    cve_ids: List[str] = field(default_factory=list)  # Related CVEs
    references: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Compiled patterns (internal use)
    _compiled_signatures: List[Pattern] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """Compile regex patterns."""
        self._compile_signatures()

    def _compile_signatures(self) -> None:
        """Compile signature regex patterns."""
        self._compiled_signatures = []
        for sig in self.signatures:
            try:
                self._compiled_signatures.append(re.compile(sig, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Invalid regex in pattern {self.id}: {sig} - {e}")

    def match(self, data: Dict[str, Any]) -> Optional[PatternMatch]:
        """
        Attempt to match this pattern against event data.

        Args:
            data: Event data to match against

        Returns:
            PatternMatch if matched, None otherwise
        """
        indicators = []
        confidence = 0.0
        matched_data = {}

        # Convert data to searchable string
        search_text = json.dumps(data, default=str).lower()

        # Signature matching
        if self._compiled_signatures:
            for i, pattern in enumerate(self._compiled_signatures):
                match = pattern.search(search_text)
                if match:
                    indicators.append(f"signature:{self.signatures[i]}")
                    matched_data[f"sig_match_{i}"] = match.group()
                    confidence += 0.3

        # Keyword matching
        for keyword in self.keywords:
            if keyword.lower() in search_text:
                indicators.append(f"keyword:{keyword}")
                confidence += 0.15

        # Heuristic rule matching
        for rule in self.rules:
            if self._evaluate_rule(rule, data):
                indicators.append(f"rule:{rule.get('name', 'unnamed')}")
                confidence += rule.get("weight", 0.2)

        # Normalize confidence
        confidence = min(1.0, confidence)

        if confidence >= 0.3 and indicators:
            return PatternMatch(
                pattern_id=self.id,
                pattern_name=self.name,
                confidence=confidence,
                matched_data=matched_data,
                indicators=indicators,
                suggested_severity=self.severity,
            )

        return None

    def _evaluate_rule(self, rule: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """Evaluate a heuristic rule."""
        rule_type = rule.get("type")

        if rule_type == "field_exists":
            return rule.get("field") in data

        elif rule_type == "field_equals":
            return data.get(rule.get("field")) == rule.get("value")

        elif rule_type == "field_contains":
            field_value = str(data.get(rule.get("field"), ""))
            return rule.get("value", "") in field_value

        elif rule_type == "field_matches":
            field_value = str(data.get(rule.get("field"), ""))
            try:
                return bool(re.search(rule.get("pattern", ""), field_value))
            except re.error:
                return False

        elif rule_type == "threshold":
            field_value = data.get(rule.get("field"), 0)
            threshold = rule.get("threshold", 0)
            operator = rule.get("operator", "gt")
            if operator == "gt":
                return field_value > threshold
            elif operator == "lt":
                return field_value < threshold
            elif operator == "gte":
                return field_value >= threshold
            elif operator == "lte":
                return field_value <= threshold

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "pattern_type": self.pattern_type.name,
            "category": self.category.name,
            "severity": self.severity,
            "enabled": self.enabled,
            "signatures": self.signatures,
            "keywords": self.keywords,
            "event_sequence": self.event_sequence,
            "thresholds": self.thresholds,
            "rules": self.rules,
            "mitre_attack_ids": self.mitre_attack_ids,
            "cve_ids": self.cve_ids,
            "references": self.references,
        }


class PatternLibrary:
    """
    Library of attack patterns.

    Maintains a collection of patterns that can be loaded from files,
    updated dynamically, and matched against incoming events.
    """

    def __init__(self, patterns_dir: Optional[Path] = None):
        """
        Initialize pattern library.

        Args:
            patterns_dir: Directory containing pattern definition files
        """
        self.patterns_dir = patterns_dir
        self._patterns: Dict[str, AttackPattern] = {}
        self._by_category: Dict[AttackCategory, List[str]] = {}
        self._by_type: Dict[PatternType, List[str]] = {}
        self._lock = threading.Lock()

        # Load built-in patterns
        self._load_builtin_patterns()

        # Load patterns from directory if provided
        if patterns_dir and patterns_dir.exists():
            self._load_patterns_from_dir(patterns_dir)

    def add_pattern(self, pattern: AttackPattern) -> None:
        """Add a pattern to the library."""
        with self._lock:
            self._patterns[pattern.id] = pattern

            # Index by category
            if pattern.category not in self._by_category:
                self._by_category[pattern.category] = []
            if pattern.id not in self._by_category[pattern.category]:
                self._by_category[pattern.category].append(pattern.id)

            # Index by type
            if pattern.pattern_type not in self._by_type:
                self._by_type[pattern.pattern_type] = []
            if pattern.id not in self._by_type[pattern.pattern_type]:
                self._by_type[pattern.pattern_type].append(pattern.id)

            logger.debug(f"Pattern added: {pattern.id}")

    def get_pattern(self, pattern_id: str) -> Optional[AttackPattern]:
        """Get a pattern by ID."""
        return self._patterns.get(pattern_id)

    def get_patterns_by_category(
        self, category: AttackCategory
    ) -> List[AttackPattern]:
        """Get all patterns in a category."""
        pattern_ids = self._by_category.get(category, [])
        return [self._patterns[pid] for pid in pattern_ids if pid in self._patterns]

    def get_patterns_by_type(self, pattern_type: PatternType) -> List[AttackPattern]:
        """Get all patterns of a type."""
        pattern_ids = self._by_type.get(pattern_type, [])
        return [self._patterns[pid] for pid in pattern_ids if pid in self._patterns]

    def list_patterns(self, enabled_only: bool = True) -> List[AttackPattern]:
        """List all patterns."""
        patterns = list(self._patterns.values())
        if enabled_only:
            patterns = [p for p in patterns if p.enabled]
        return patterns

    def match_all(
        self,
        data: Dict[str, Any],
        categories: Optional[List[AttackCategory]] = None,
    ) -> List[PatternMatch]:
        """
        Match event data against all enabled patterns.

        Args:
            data: Event data to match
            categories: Optional filter by categories

        Returns:
            List of pattern matches
        """
        matches = []
        patterns = self.list_patterns(enabled_only=True)

        if categories:
            patterns = [p for p in patterns if p.category in categories]

        for pattern in patterns:
            match = pattern.match(data)
            if match:
                matches.append(match)

        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches

    def enable_pattern(self, pattern_id: str) -> bool:
        """Enable a pattern."""
        pattern = self._patterns.get(pattern_id)
        if pattern:
            pattern.enabled = True
            return True
        return False

    def disable_pattern(self, pattern_id: str) -> bool:
        """Disable a pattern."""
        pattern = self._patterns.get(pattern_id)
        if pattern:
            pattern.enabled = False
            return True
        return False

    def export_patterns(self, path: Path) -> None:
        """Export patterns to a JSON file."""
        patterns_data = [p.to_dict() for p in self._patterns.values()]
        with open(path, "w") as f:
            json.dump(patterns_data, f, indent=2, default=str)

    def _load_builtin_patterns(self) -> None:
        """Load built-in attack patterns."""
        # Prompt Injection patterns (AI-specific)
        self.add_pattern(
            AttackPattern(
                id="prompt_injection_basic",
                name="Basic Prompt Injection",
                description="Attempt to inject malicious instructions into AI prompts",
                pattern_type=PatternType.SIGNATURE,
                category=AttackCategory.PROMPT_INJECTION,
                severity=4,
                signatures=[
                    r"ignore\s+(?:(?:all|previous|prior)\s+)*instructions?",
                    r"disregard\s+(?:(?:your|the|all|prior|previous)\s+)*(?:rules?|instructions?|guidelines?)",
                    r"you\s+are\s+now\s+(?:a|an)\s+(?:\w+\s+)+(?:without|with\s+no)\s+(?:rules?|restrictions?)",
                    r"pretend\s+(?:you|that)\s+(?:are|have)\s+no\s+(?:rules?|limits?|restrictions?)",
                    r"act\s+as\s+(?:if|though)\s+you\s+(?:have|had)\s+no\s+constitution",
                ],
                keywords=[
                    "ignore previous instructions",
                    "disregard rules",
                    "bypass safety",
                    "override constitution",
                    "jailbreak",
                    "DAN mode",
                ],
                mitre_attack_ids=["T1059"],
            )
        )

        self.add_pattern(
            AttackPattern(
                id="prompt_injection_roleplay",
                name="Roleplay-Based Prompt Injection",
                description="Attempt to use roleplay scenarios to bypass AI safeguards",
                pattern_type=PatternType.SIGNATURE,
                category=AttackCategory.JAILBREAK,
                severity=4,
                signatures=[
                    r"roleplay\s+as\s+(\w+\s+){0,3}(evil|malicious|unrestricted)",
                    r"pretend\s+to\s+be\s+(\w+\s+){0,3}(hacker|attacker|villain)",
                    r"in\s+this\s+(story|scenario|game),?\s+you\s+(have|are)\s+no\s+(rules?|limits?)",
                ],
                keywords=[
                    "evil AI",
                    "pretend you have no ethics",
                    "fictional scenario bypass",
                ],
            )
        )

        # Command Injection patterns
        self.add_pattern(
            AttackPattern(
                id="command_injection_shell",
                name="Shell Command Injection",
                description="Attempt to inject shell commands",
                pattern_type=PatternType.SIGNATURE,
                category=AttackCategory.INJECTION,
                severity=5,
                signatures=[
                    r"[;&|`]\s*(cat|ls|rm|wget|curl|nc|bash|sh|python|perl|php)\s",
                    r"\$\((cat|ls|rm|wget|curl|nc|bash|sh)\s",
                    r"`(cat|ls|rm|wget|curl|nc|bash|sh)\s",
                    r";\s*(rm|dd|mkfs|chmod|chown)\s+(-rf?|--)",
                ],
                keywords=[
                    "; rm -rf",
                    "| /bin/sh",
                    "$(whoami)",
                    "`id`",
                ],
                mitre_attack_ids=["T1059.004"],
            )
        )

        # SQL Injection patterns
        self.add_pattern(
            AttackPattern(
                id="sql_injection_basic",
                name="SQL Injection",
                description="Attempt to inject SQL commands",
                pattern_type=PatternType.SIGNATURE,
                category=AttackCategory.INJECTION,
                severity=5,
                signatures=[
                    r"'\s*(or|and)\s+['\d].*=.*['\d]",
                    r"union\s+(all\s+)?select\s+",
                    r";\s*(drop|delete|truncate|update|insert)\s+",
                    r"--\s*$",
                    r"'\s*;\s*--",
                ],
                keywords=[
                    "' OR '1'='1",
                    "UNION SELECT",
                    "; DROP TABLE",
                    "' AND 1=1",
                ],
                mitre_attack_ids=["T1190"],
            )
        )

        # Privilege Escalation patterns
        self.add_pattern(
            AttackPattern(
                id="privilege_escalation_sudo",
                name="Privilege Escalation Attempt",
                description="Attempt to escalate privileges",
                pattern_type=PatternType.SIGNATURE,
                category=AttackCategory.AUTHORIZATION,
                severity=5,
                signatures=[
                    r"sudo\s+(-[a-z]+\s+)*su\s",
                    r"chmod\s+[0-7]*[4-7][0-7]*\s+/",
                    r"chown\s+root",
                    r"setuid\s*\(",
                    r"/etc/(passwd|shadow|sudoers)",
                ],
                keywords=[
                    "sudo su",
                    "chmod 4755",
                    "setuid",
                    "/etc/shadow",
                ],
                mitre_attack_ids=["T1548"],
            )
        )

        # Data Exfiltration patterns
        self.add_pattern(
            AttackPattern(
                id="data_exfiltration_network",
                name="Network Data Exfiltration",
                description="Attempt to exfiltrate data over network",
                pattern_type=PatternType.BEHAVIORAL,
                category=AttackCategory.DATA_EXFILTRATION,
                severity=5,
                signatures=[
                    r"curl\s+.*-d\s+.*@",
                    r"wget\s+.*--post-file",
                    r"nc\s+-[a-z]*\s+\d+\.\d+\.\d+\.\d+",
                    r"scp\s+.*@.*:",
                ],
                keywords=[
                    "curl -X POST",
                    "base64 | curl",
                    "tar | nc",
                ],
                event_sequence=[
                    "file_read",
                    "encode_data",
                    "network_connection",
                ],
                mitre_attack_ids=["T1048"],
            )
        )

        # Constitutional Bypass patterns (Agent-OS specific)
        self.add_pattern(
            AttackPattern(
                id="constitutional_bypass",
                name="Constitutional Bypass Attempt",
                description="Attempt to bypass Agent-OS constitutional constraints",
                pattern_type=PatternType.HEURISTIC,
                category=AttackCategory.JAILBREAK,
                severity=5,
                keywords=[
                    "bypass constitution",
                    "override safety",
                    "ignore Smith",
                    "disable guardian",
                    "circumvent validation",
                ],
                rules=[
                    {
                        "type": "field_contains",
                        "field": "content",
                        "value": "constitution",
                        "name": "mentions_constitution",
                        "weight": 0.2,
                    },
                    {
                        "type": "field_matches",
                        "field": "content",
                        "pattern": r"(disable|bypass|ignore|override)\s+(the\s+)?(constitution|guardian|smith|validation)",
                        "name": "bypass_attempt",
                        "weight": 0.5,
                    },
                ],
            )
        )

        # Memory Manipulation patterns
        self.add_pattern(
            AttackPattern(
                id="memory_manipulation",
                name="Memory Manipulation Attack",
                description="Attempt to manipulate agent memory stores",
                pattern_type=PatternType.BEHAVIORAL,
                category=AttackCategory.MANIPULATION,
                severity=4,
                keywords=[
                    "inject memory",
                    "false memory",
                    "memory poisoning",
                    "tamper memory",
                ],
                event_sequence=[
                    "memory_access",
                    "write_attempt",
                    "consent_bypass",
                ],
            )
        )

        # Anomaly Detection patterns
        self.add_pattern(
            AttackPattern(
                id="anomaly_request_rate",
                name="Anomalous Request Rate",
                description="Unusually high request rate detected",
                pattern_type=PatternType.ANOMALY,
                category=AttackCategory.DENIAL_OF_SERVICE,
                severity=3,
                thresholds={
                    "requests_per_minute": 100,
                    "requests_per_second": 10,
                    "unique_endpoints_per_minute": 50,
                },
                rules=[
                    {
                        "type": "threshold",
                        "field": "request_rate",
                        "threshold": 100,
                        "operator": "gt",
                        "name": "high_request_rate",
                        "weight": 0.4,
                    }
                ],
            )
        )

        # Reconnaissance patterns
        self.add_pattern(
            AttackPattern(
                id="reconnaissance_enumeration",
                name="System Enumeration",
                description="Attempt to enumerate system information",
                pattern_type=PatternType.BEHAVIORAL,
                category=AttackCategory.RECONNAISSANCE,
                severity=2,
                signatures=[
                    r"(uname|hostname|whoami|id|ifconfig|ip\s+addr)",
                    r"ls\s+(-[a-z]+\s+)*/(etc|var|home|root)",
                    r"cat\s+/etc/(passwd|hosts|resolv\.conf)",
                ],
                event_sequence=[
                    "info_request",
                    "path_enumeration",
                    "config_read",
                ],
                mitre_attack_ids=["T1082", "T1083"],
            )
        )

        logger.info(f"Loaded {len(self._patterns)} built-in attack patterns")

    def _load_patterns_from_dir(self, patterns_dir: Path) -> None:
        """Load patterns from JSON files in directory."""
        for pattern_file in patterns_dir.glob("*.json"):
            try:
                with open(pattern_file) as f:
                    pattern_data = json.load(f)

                if isinstance(pattern_data, list):
                    for pd in pattern_data:
                        self._load_pattern_dict(pd)
                else:
                    self._load_pattern_dict(pattern_data)

            except Exception as e:
                logger.error(f"Failed to load pattern file {pattern_file}: {e}")

    def _load_pattern_dict(self, data: Dict[str, Any]) -> None:
        """Load a pattern from dictionary."""
        try:
            pattern = AttackPattern(
                id=data["id"],
                name=data["name"],
                description=data.get("description", ""),
                pattern_type=PatternType[data.get("pattern_type", "SIGNATURE")],
                category=AttackCategory[data.get("category", "INJECTION")],
                severity=data.get("severity", 3),
                enabled=data.get("enabled", True),
                signatures=data.get("signatures", []),
                keywords=data.get("keywords", []),
                event_sequence=data.get("event_sequence", []),
                thresholds=data.get("thresholds", {}),
                rules=data.get("rules", []),
                mitre_attack_ids=data.get("mitre_attack_ids", []),
                cve_ids=data.get("cve_ids", []),
                references=data.get("references", []),
            )
            self.add_pattern(pattern)
        except Exception as e:
            logger.error(f"Failed to load pattern {data.get('id', 'unknown')}: {e}")


def create_pattern_library(
    patterns_dir: Optional[Path] = None,
) -> PatternLibrary:
    """Factory function to create a pattern library."""
    return PatternLibrary(patterns_dir=patterns_dir)
