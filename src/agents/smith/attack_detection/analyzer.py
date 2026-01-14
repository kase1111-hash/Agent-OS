"""
Attack Analyzer

Analyzes detected attacks to:
1. Identify vulnerable code paths
2. Determine root cause
3. Map attack vectors to code locations
4. Generate vulnerability reports

This module uses static analysis and attack signature correlation
to pinpoint the exact code that needs to be patched.
"""

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .detector import AttackEvent, AttackType, AttackSeverity
from .patterns import AttackCategory, PatternMatch

logger = logging.getLogger(__name__)


class VulnerabilityType(Enum):
    """Types of vulnerabilities."""

    INPUT_VALIDATION = auto()  # Missing/insufficient input validation
    OUTPUT_ENCODING = auto()  # Missing output encoding
    AUTHENTICATION = auto()  # Auth weakness
    AUTHORIZATION = auto()  # Authz weakness
    INJECTION = auto()  # Injection vulnerability
    INSECURE_DESIGN = auto()  # Design flaw
    CONFIGURATION = auto()  # Misconfiguration
    CRYPTOGRAPHIC = auto()  # Crypto weakness
    BOUNDARY_VIOLATION = auto()  # Boundary enforcement gap
    CONSTITUTIONAL_GAP = auto()  # Constitutional coverage gap
    PROMPT_HANDLING = auto()  # Prompt processing weakness


class RiskLevel(Enum):
    """Risk levels for vulnerabilities."""

    INFORMATIONAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


@dataclass
class CodeLocation:
    """
    Represents a location in the codebase.
    """

    file_path: str
    line_start: int
    line_end: int
    column_start: int = 0
    column_end: int = 0
    function_name: str = ""
    class_name: str = ""
    code_snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "column_start": self.column_start,
            "column_end": self.column_end,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "code_snippet": self.code_snippet,
        }

    def __str__(self) -> str:
        if self.function_name:
            return f"{self.file_path}:{self.line_start} ({self.function_name})"
        return f"{self.file_path}:{self.line_start}"


@dataclass
class VulnerabilityFinding:
    """
    A specific vulnerability finding.
    """

    finding_id: str
    vulnerability_type: VulnerabilityType
    risk_level: RiskLevel
    title: str
    description: str
    location: CodeLocation
    attack_id: str  # Related attack

    # Analysis details
    root_cause: str = ""
    attack_vector: str = ""
    exploitability: str = ""

    # Remediation guidance
    remediation_guidance: str = ""
    code_fix_suggestion: str = ""
    references: List[str] = field(default_factory=list)

    # Metadata
    cwe_ids: List[str] = field(default_factory=list)  # CWE identifiers
    confidence: float = 0.0
    discovered_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "finding_id": self.finding_id,
            "vulnerability_type": self.vulnerability_type.name,
            "risk_level": self.risk_level.name,
            "title": self.title,
            "description": self.description,
            "location": self.location.to_dict(),
            "attack_id": self.attack_id,
            "root_cause": self.root_cause,
            "attack_vector": self.attack_vector,
            "exploitability": self.exploitability,
            "remediation_guidance": self.remediation_guidance,
            "code_fix_suggestion": self.code_fix_suggestion,
            "references": self.references,
            "cwe_ids": self.cwe_ids,
            "confidence": self.confidence,
            "discovered_at": self.discovered_at.isoformat(),
        }


@dataclass
class VulnerabilityReport:
    """
    Complete vulnerability analysis report for an attack.
    """

    report_id: str
    attack_id: str
    attack_type: AttackType
    generated_at: datetime

    # Summary
    summary: str = ""
    risk_score: float = 0.0  # 0-10 scale
    total_findings: int = 0

    # Findings
    findings: List[VulnerabilityFinding] = field(default_factory=list)

    # Attack analysis
    attack_chain: List[str] = field(default_factory=list)
    entry_points: List[CodeLocation] = field(default_factory=list)
    affected_components: List[str] = field(default_factory=list)

    # Remediation summary
    critical_fixes: List[str] = field(default_factory=list)
    recommended_patches: List[str] = field(default_factory=list)
    architectural_changes: List[str] = field(default_factory=list)

    # Metadata
    analysis_duration_ms: int = 0
    analyzer_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "attack_id": self.attack_id,
            "attack_type": self.attack_type.name,
            "generated_at": self.generated_at.isoformat(),
            "summary": self.summary,
            "risk_score": self.risk_score,
            "total_findings": self.total_findings,
            "findings": [f.to_dict() for f in self.findings],
            "attack_chain": self.attack_chain,
            "entry_points": [e.to_dict() for e in self.entry_points],
            "affected_components": self.affected_components,
            "critical_fixes": self.critical_fixes,
            "recommended_patches": self.recommended_patches,
            "architectural_changes": self.architectural_changes,
            "analysis_duration_ms": self.analysis_duration_ms,
            "analyzer_version": self.analyzer_version,
        }

    def add_finding(self, finding: VulnerabilityFinding) -> None:
        """Add a finding to the report."""
        self.findings.append(finding)
        self.total_findings = len(self.findings)

        # Update risk score
        self._update_risk_score()

    def _update_risk_score(self) -> None:
        """Update the overall risk score."""
        if not self.findings:
            self.risk_score = 0.0
            return

        # Weighted average based on risk levels
        weights = {
            RiskLevel.INFORMATIONAL: 0.5,
            RiskLevel.LOW: 1.5,
            RiskLevel.MEDIUM: 3.0,
            RiskLevel.HIGH: 5.0,
            RiskLevel.CRITICAL: 8.0,
        }

        total = sum(weights[f.risk_level] * f.confidence for f in self.findings)
        max_possible = len(self.findings) * 8.0
        self.risk_score = min(10.0, (total / max_possible) * 10.0 if max_possible > 0 else 0)


class CodebaseAnalyzer:
    """
    Analyzes the codebase to find vulnerable code patterns.

    Uses AST analysis and pattern matching to identify code locations
    that may be vulnerable to the detected attack.
    """

    # Vulnerability patterns for different attack types
    VULNERABILITY_PATTERNS = {
        AttackType.INJECTION: [
            {
                "pattern": r"subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True",
                "description": "Shell command execution with shell=True",
                "vuln_type": VulnerabilityType.INJECTION,
                "cwe": "CWE-78",
            },
            {
                "pattern": r"os\.system\s*\(",
                "description": "Direct system command execution",
                "vuln_type": VulnerabilityType.INJECTION,
                "cwe": "CWE-78",
            },
            {
                "pattern": r"eval\s*\([^)]+\)",
                "description": "Use of eval() with external input",
                "vuln_type": VulnerabilityType.INJECTION,
                "cwe": "CWE-94",
            },
            {
                "pattern": r"exec\s*\([^)]+\)",
                "description": "Use of exec() with external input",
                "vuln_type": VulnerabilityType.INJECTION,
                "cwe": "CWE-94",
            },
        ],
        AttackType.PROMPT_INJECTION: [
            {
                "pattern": r"prompt\s*=\s*.*\+.*input",
                "description": "Unsanitized user input in prompt",
                "vuln_type": VulnerabilityType.PROMPT_HANDLING,
                "cwe": "CWE-77",
            },
            {
                "pattern": r"f['\"].*\{.*request.*\}",
                "description": "Request data directly in formatted string",
                "vuln_type": VulnerabilityType.INPUT_VALIDATION,
                "cwe": "CWE-20",
            },
        ],
        AttackType.JAILBREAK: [
            {
                "pattern": r"constitution.*skip|bypass.*validate",
                "description": "Potential constitutional check bypass",
                "vuln_type": VulnerabilityType.CONSTITUTIONAL_GAP,
                "cwe": "CWE-285",
            },
        ],
        AttackType.PRIVILEGE_ESCALATION: [
            {
                "pattern": r"@admin_required\s*\n.*def\s+\w+.*request",
                "description": "Admin endpoint with potential bypass",
                "vuln_type": VulnerabilityType.AUTHORIZATION,
                "cwe": "CWE-285",
            },
            {
                "pattern": r"if.*is_admin.*or.*True",
                "description": "Authorization check with bypass condition",
                "vuln_type": VulnerabilityType.AUTHORIZATION,
                "cwe": "CWE-863",
            },
        ],
        AttackType.DATA_EXFILTRATION: [
            {
                "pattern": r"(requests|urllib|http\.client)\.(get|post|put)",
                "description": "Outbound HTTP request",
                "vuln_type": VulnerabilityType.BOUNDARY_VIOLATION,
                "cwe": "CWE-200",
            },
        ],
    }

    def __init__(self, codebase_root: Path):
        """
        Initialize codebase analyzer.

        Args:
            codebase_root: Root directory of the codebase
        """
        self.codebase_root = codebase_root
        self._file_cache: Dict[str, str] = {}
        self._lock = threading.Lock()

    def find_vulnerable_code(
        self,
        attack: AttackEvent,
        patterns: Optional[List[Dict[str, Any]]] = None,
    ) -> List[VulnerabilityFinding]:
        """
        Find code locations vulnerable to the given attack.

        Args:
            attack: The detected attack
            patterns: Custom patterns to search for

        Returns:
            List of vulnerability findings
        """
        findings = []

        # Get patterns for this attack type
        search_patterns = patterns or self.VULNERABILITY_PATTERNS.get(
            attack.attack_type, []
        )

        # Also extract patterns from attack indicators
        for ioc in attack.indicators_of_compromise:
            if ioc.startswith("signature:"):
                sig = ioc.replace("signature:", "")
                search_patterns.append({
                    "pattern": sig,
                    "description": f"Attack signature match: {sig[:50]}",
                    "vuln_type": VulnerabilityType.INPUT_VALIDATION,
                    "cwe": "CWE-20",
                })

        # Search the codebase
        for pattern_def in search_patterns:
            pattern = pattern_def["pattern"]
            matches = self._search_codebase(pattern)

            for file_path, line_num, code_line in matches:
                finding = self._create_finding(
                    attack=attack,
                    pattern_def=pattern_def,
                    file_path=file_path,
                    line_num=line_num,
                    code_line=code_line,
                )
                if finding:
                    findings.append(finding)

        return findings

    def analyze_entry_points(
        self,
        attack: AttackEvent,
    ) -> List[CodeLocation]:
        """
        Identify entry points for the attack.

        Args:
            attack: The detected attack

        Returns:
            List of entry point locations
        """
        entry_points = []

        # Map attack types to likely entry point patterns
        entry_point_patterns = {
            AttackType.PROMPT_INJECTION: [
                (r"def\s+process\s*\(.*request", "Request processing function"),
                (r"class\s+.*Handler.*:", "Request handler class"),
                (r"@app\.(route|post|get)", "Web endpoint"),
            ],
            AttackType.INJECTION: [
                (r"def\s+execute_tool", "Tool execution function"),
                (r"subprocess\.", "Subprocess call"),
                (r"os\.system", "System call"),
            ],
            AttackType.JAILBREAK: [
                (r"class\s+.*Validator", "Validator class"),
                (r"def\s+validate", "Validation function"),
                (r"constitution", "Constitutional check"),
            ],
            AttackType.PRIVILEGE_ESCALATION: [
                (r"def\s+check_permission", "Permission check"),
                (r"@requires_auth", "Auth decorator"),
                (r"is_authorized", "Authorization check"),
            ],
        }

        patterns = entry_point_patterns.get(attack.attack_type, [])

        for pattern, description in patterns:
            matches = self._search_codebase(pattern)
            for file_path, line_num, code_line in matches[:5]:  # Limit to top 5
                entry_points.append(
                    CodeLocation(
                        file_path=file_path,
                        line_start=line_num,
                        line_end=line_num,
                        code_snippet=code_line.strip(),
                    )
                )

        return entry_points

    def get_affected_components(self, attack: AttackEvent) -> List[str]:
        """
        Identify components affected by the attack.

        Args:
            attack: The detected attack

        Returns:
            List of affected component names
        """
        components = set()

        # Based on attack type
        component_mapping = {
            AttackType.PROMPT_INJECTION: ["request_handler", "whisper", "message_bus"],
            AttackType.JAILBREAK: ["smith", "constitution", "refusal_engine"],
            AttackType.INJECTION: ["tool_executor", "sandbox", "boundary_daemon"],
            AttackType.PRIVILEGE_ESCALATION: ["auth", "permissions", "policy_engine"],
            AttackType.DATA_EXFILTRATION: ["memory_vault", "boundary_daemon", "network"],
        }

        components.update(component_mapping.get(attack.attack_type, []))

        # Add target component if specified
        if attack.target_component:
            components.add(attack.target_component)

        if attack.target_agent:
            components.add(attack.target_agent)

        return list(components)

    def _search_codebase(
        self,
        pattern: str,
    ) -> List[Tuple[str, int, str]]:
        """
        Search codebase for a pattern.

        Args:
            pattern: Regex pattern to search

        Returns:
            List of (file_path, line_number, code_line) tuples
        """
        matches = []
        compiled = re.compile(pattern, re.IGNORECASE)

        # Search Python files
        for py_file in self.codebase_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            if ".git" in str(py_file):
                continue

            try:
                content = self._get_file_content(str(py_file))
                for line_num, line in enumerate(content.split("\n"), 1):
                    if compiled.search(line):
                        matches.append((str(py_file), line_num, line))

            except Exception as e:
                logger.debug(f"Error searching {py_file}: {e}")

        return matches

    def _get_file_content(self, file_path: str) -> str:
        """Get file content with caching."""
        with self._lock:
            if file_path not in self._file_cache:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        self._file_cache[file_path] = f.read()
                except Exception:
                    self._file_cache[file_path] = ""
            return self._file_cache[file_path]

    def _create_finding(
        self,
        attack: AttackEvent,
        pattern_def: Dict[str, Any],
        file_path: str,
        line_num: int,
        code_line: str,
    ) -> Optional[VulnerabilityFinding]:
        """Create a vulnerability finding."""
        # Determine risk level based on attack severity
        risk_mapping = {
            AttackSeverity.LOW: RiskLevel.LOW,
            AttackSeverity.MEDIUM: RiskLevel.MEDIUM,
            AttackSeverity.HIGH: RiskLevel.HIGH,
            AttackSeverity.CRITICAL: RiskLevel.CRITICAL,
            AttackSeverity.CATASTROPHIC: RiskLevel.CRITICAL,
        }
        risk_level = risk_mapping.get(attack.severity, RiskLevel.MEDIUM)

        # Get context around the line
        content = self._get_file_content(file_path)
        lines = content.split("\n")
        context_start = max(0, line_num - 3)
        context_end = min(len(lines), line_num + 2)
        code_snippet = "\n".join(lines[context_start:context_end])

        # Generate finding ID
        finding_id = hashlib.sha256(
            f"{file_path}:{line_num}:{attack.attack_id}".encode()
        ).hexdigest()[:12].upper()

        vuln_type = pattern_def.get("vuln_type", VulnerabilityType.INPUT_VALIDATION)

        return VulnerabilityFinding(
            finding_id=f"VLN-{finding_id}",
            vulnerability_type=vuln_type,
            risk_level=risk_level,
            title=pattern_def.get("description", "Potential vulnerability"),
            description=f"Pattern match found at {file_path}:{line_num}",
            location=CodeLocation(
                file_path=file_path,
                line_start=line_num,
                line_end=line_num,
                code_snippet=code_snippet,
            ),
            attack_id=attack.attack_id,
            root_cause=f"Code pattern susceptible to {attack.attack_type.name}",
            cwe_ids=[pattern_def.get("cwe", "")] if pattern_def.get("cwe") else [],
            confidence=attack.confidence * 0.8,  # Slightly lower confidence
        )


class AttackAnalyzer:
    """
    Main attack analyzer that produces vulnerability reports.
    """

    def __init__(
        self,
        codebase_root: Optional[Path] = None,
    ):
        """
        Initialize attack analyzer.

        Args:
            codebase_root: Root of codebase to analyze
        """
        self.codebase_root = codebase_root or Path("/home/user/Agent-OS")
        self._codebase_analyzer = CodebaseAnalyzer(self.codebase_root)
        self._reports: Dict[str, VulnerabilityReport] = {}
        self._lock = threading.Lock()

    def analyze(self, attack: AttackEvent) -> VulnerabilityReport:
        """
        Analyze an attack and produce a vulnerability report.

        Args:
            attack: The attack to analyze

        Returns:
            VulnerabilityReport
        """
        import time
        start_time = time.time()

        # Generate report ID
        report_id = f"RPT-{hashlib.sha256(attack.attack_id.encode()).hexdigest()[:12].upper()}"

        report = VulnerabilityReport(
            report_id=report_id,
            attack_id=attack.attack_id,
            attack_type=attack.attack_type,
            generated_at=datetime.now(),
        )

        # Find vulnerable code
        findings = self._codebase_analyzer.find_vulnerable_code(attack)
        for finding in findings:
            self._enrich_finding(finding, attack)
            report.add_finding(finding)

        # Find entry points
        report.entry_points = self._codebase_analyzer.analyze_entry_points(attack)

        # Get affected components
        report.affected_components = self._codebase_analyzer.get_affected_components(attack)

        # Build attack chain
        report.attack_chain = self._build_attack_chain(attack)

        # Generate summary
        report.summary = self._generate_summary(report)

        # Generate remediation recommendations
        self._generate_remediation_recommendations(report)

        # Record timing
        report.analysis_duration_ms = int((time.time() - start_time) * 1000)

        # Store report
        with self._lock:
            self._reports[report_id] = report

        logger.info(
            f"Attack analysis complete: {report_id} - "
            f"{report.total_findings} findings, risk score: {report.risk_score:.1f}"
        )

        return report

    def get_report(self, report_id: str) -> Optional[VulnerabilityReport]:
        """Get a report by ID."""
        return self._reports.get(report_id)

    def list_reports(
        self,
        attack_type: Optional[AttackType] = None,
    ) -> List[VulnerabilityReport]:
        """List all reports."""
        reports = list(self._reports.values())
        if attack_type:
            reports = [r for r in reports if r.attack_type == attack_type]
        return sorted(reports, key=lambda r: r.generated_at, reverse=True)

    def _enrich_finding(
        self,
        finding: VulnerabilityFinding,
        attack: AttackEvent,
    ) -> None:
        """Enrich a finding with additional context."""
        # Add attack vector
        finding.attack_vector = attack.attack_vector or attack.attack_type.name

        # Add exploitability assessment
        if attack.confidence >= 0.8:
            finding.exploitability = "High - confirmed exploitable"
        elif attack.confidence >= 0.5:
            finding.exploitability = "Medium - likely exploitable"
        else:
            finding.exploitability = "Low - potential vulnerability"

        # Add remediation guidance based on vulnerability type
        guidance = {
            VulnerabilityType.INPUT_VALIDATION: (
                "Implement strict input validation. Use allowlists rather than "
                "blocklists. Validate all user-supplied data before processing."
            ),
            VulnerabilityType.INJECTION: (
                "Use parameterized queries/commands. Avoid shell=True in subprocess. "
                "Never use eval/exec with user input. Use the tool sandbox."
            ),
            VulnerabilityType.PROMPT_HANDLING: (
                "Sanitize user input before including in prompts. Use prompt templates "
                "with strict escaping. Implement S3 instruction integrity validation."
            ),
            VulnerabilityType.AUTHORIZATION: (
                "Implement proper authorization checks. Use the policy engine. "
                "Follow principle of least privilege."
            ),
            VulnerabilityType.CONSTITUTIONAL_GAP: (
                "Review constitutional constraints. Add missing rules to cover "
                "this attack vector. Update Smith's validation checks."
            ),
            VulnerabilityType.BOUNDARY_VIOLATION: (
                "Enforce network boundaries. Use RESTRICTED mode. Add tripwires "
                "for network access attempts."
            ),
        }

        finding.remediation_guidance = guidance.get(
            finding.vulnerability_type,
            "Review the code and apply security best practices."
        )

    def _build_attack_chain(self, attack: AttackEvent) -> List[str]:
        """Build the attack chain description."""
        chain = []

        # Entry
        chain.append(f"1. Attack initiated via {attack.target_component or 'unknown entry point'}")

        # Based on attack type
        if attack.attack_type == AttackType.PROMPT_INJECTION:
            chain.append("2. Malicious prompt injected into agent request")
            chain.append("3. Attempt to override agent instructions")
            chain.append("4. Constitutional validation triggered")

        elif attack.attack_type == AttackType.JAILBREAK:
            chain.append("2. Roleplay/bypass technique employed")
            chain.append("3. Attempt to circumvent safety constraints")
            chain.append("4. Refusal engine activated")

        elif attack.attack_type == AttackType.INJECTION:
            chain.append("2. Command/code injection attempt")
            chain.append("3. Execution attempted in sandbox")
            chain.append("4. Boundary violation detected")

        elif attack.attack_type == AttackType.DATA_EXFILTRATION:
            chain.append("2. Data access attempted")
            chain.append("3. Exfiltration channel identified")
            chain.append("4. S7 exfiltration control triggered")

        # Resolution
        if attack.status.name in ["MITIGATED", "DETECTED"]:
            chain.append(f"5. Attack {attack.status.name.lower()} by Smith")

        return chain

    def _generate_summary(self, report: VulnerabilityReport) -> str:
        """Generate a human-readable summary."""
        if not report.findings:
            return f"Analysis of {report.attack_type.name} attack found no vulnerable code patterns."

        high_risk = len([f for f in report.findings if f.risk_level.value >= RiskLevel.HIGH.value])

        summary = (
            f"Analysis identified {report.total_findings} potential vulnerability(ies) "
            f"related to the {report.attack_type.name} attack. "
        )

        if high_risk > 0:
            summary += f"{high_risk} finding(s) are HIGH or CRITICAL risk. "

        summary += f"Overall risk score: {report.risk_score:.1f}/10. "

        if report.affected_components:
            summary += f"Affected components: {', '.join(report.affected_components)}."

        return summary

    def _generate_remediation_recommendations(self, report: VulnerabilityReport) -> None:
        """Generate remediation recommendations."""
        critical_fixes = []
        patches = []
        architectural = []

        for finding in report.findings:
            if finding.risk_level == RiskLevel.CRITICAL:
                critical_fixes.append(
                    f"[CRITICAL] {finding.title} at {finding.location.file_path}:{finding.location.line_start}"
                )
            elif finding.risk_level == RiskLevel.HIGH:
                patches.append(
                    f"[HIGH] {finding.title}: {finding.remediation_guidance[:100]}"
                )

        # Add type-specific architectural changes
        if report.attack_type == AttackType.PROMPT_INJECTION:
            architectural.append("Consider implementing prompt sandboxing")
            architectural.append("Add pattern-based prompt filtering at entry point")

        elif report.attack_type == AttackType.JAILBREAK:
            architectural.append("Strengthen constitutional enforcement layer")
            architectural.append("Add behavioral analysis to detect bypass attempts")

        elif report.attack_type == AttackType.INJECTION:
            architectural.append("Review and harden tool execution sandbox")
            architectural.append("Implement input sanitization pipeline")

        report.critical_fixes = critical_fixes
        report.recommended_patches = patches[:10]  # Limit to top 10
        report.architectural_changes = architectural


def create_attack_analyzer(
    codebase_root: Optional[Path] = None,
) -> AttackAnalyzer:
    """
    Factory function to create an attack analyzer.

    Args:
        codebase_root: Root of codebase to analyze

    Returns:
        Configured AttackAnalyzer
    """
    return AttackAnalyzer(codebase_root=codebase_root)
