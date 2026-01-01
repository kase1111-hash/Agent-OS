"""
LLM-Powered Attack Analysis

Provides intelligent attack analysis using the Sage reasoning agent.
Enables:
- Deep analysis of attack patterns and intent
- Intelligent vulnerability discovery in code
- Natural language explanations of threats
- MITRE ATT&CK tactic classification
- Contextual fix recommendations

Falls back to pattern-based analysis when Sage is unavailable.
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import Sage
try:
    from src.agents.sage import (
        SageAgent,
        ReasoningChain,
        ReasoningType,
        ConfidenceLevel,
        create_sage_agent,
    )

    SAGE_AVAILABLE = True
except ImportError:
    SAGE_AVAILABLE = False
    SageAgent = None
    ReasoningChain = None


# =============================================================================
# Analysis Prompts
# =============================================================================


ATTACK_ANALYSIS_PROMPT = """Analyze the following security attack that was detected in the system.

## Attack Details
- **Attack ID:** {attack_id}
- **Attack Type:** {attack_type}
- **Severity:** {severity}
- **Source:** {source}
- **Description:** {description}
- **Detected At:** {detected_at}

## Additional Context
{context}

## Analysis Tasks
Please analyze this attack and provide:

1. **Attack Intent Analysis**
   - What is the likely goal of this attack?
   - What data or access is the attacker trying to obtain?
   - Is this part of a larger attack chain?

2. **Attack Vector Analysis**
   - How is the attack being executed?
   - What system components are being exploited?
   - What techniques from MITRE ATT&CK framework apply?

3. **Impact Assessment**
   - What is the potential damage if successful?
   - What data could be compromised?
   - What systems could be affected?

4. **Vulnerability Analysis**
   - What weaknesses allowed this attack?
   - Are there similar vulnerabilities elsewhere in the system?
   - What code patterns enabled this attack?

5. **Recommendations**
   - What immediate mitigations should be applied?
   - What long-term fixes are needed?
   - How can similar attacks be prevented?

Please provide structured analysis with confidence levels for each finding.
"""


CODE_VULNERABILITY_PROMPT = """Analyze the following code for security vulnerabilities related to a detected attack.

## Attack Context
- **Attack Type:** {attack_type}
- **Attack Description:** {description}

## Code to Analyze
**File:** {file_path}
**Lines:** {line_start}-{line_end}

```{language}
{code_snippet}
```

## Analysis Tasks
Please analyze this code and identify:

1. **Vulnerability Identification**
   - What specific vulnerabilities exist in this code?
   - How can they be exploited?
   - What is the severity of each vulnerability?

2. **Attack Surface Analysis**
   - What inputs could be used to trigger the vulnerability?
   - What conditions must be true for exploitation?
   - Are there any existing mitigations?

3. **Root Cause Analysis**
   - Why does this vulnerability exist?
   - Is it a coding error, design flaw, or missing validation?
   - Could this pattern appear elsewhere?

4. **Fix Recommendations**
   - What changes would fix this vulnerability?
   - Provide specific code modifications if possible
   - What security patterns should be applied?

5. **Testing Recommendations**
   - How should the fix be tested?
   - What edge cases should be considered?
   - What regression tests are needed?

Provide structured output with:
- Vulnerability type
- Severity (1-5)
- Confidence level
- Specific line numbers
- Recommended fix
"""


MITRE_CLASSIFICATION_PROMPT = """Classify the following attack according to the MITRE ATT&CK framework.

## Attack Details
- **Attack Type:** {attack_type}
- **Description:** {description}
- **Indicators:** {indicators}
- **Source:** {source}
- **Target:** {target}

## Classification Tasks
Please classify this attack:

1. **Tactics** (What is the attacker's goal?)
   List all applicable tactics from:
   - Initial Access
   - Execution
   - Persistence
   - Privilege Escalation
   - Defense Evasion
   - Credential Access
   - Discovery
   - Lateral Movement
   - Collection
   - Command and Control
   - Exfiltration
   - Impact

2. **Techniques** (How is the goal achieved?)
   For each tactic, list specific techniques (e.g., T1059 - Command and Scripting Interpreter)

3. **Sub-Techniques** (Specific variations)
   List any applicable sub-techniques (e.g., T1059.001 - PowerShell)

4. **Confidence Assessment**
   - How confident are you in this classification?
   - What additional information would increase confidence?

Return structured JSON with tactics, techniques, and confidence levels.
"""


# =============================================================================
# Data Models
# =============================================================================


class AnalysisConfidence(Enum):
    """Confidence level for analysis findings."""

    VERY_HIGH = "very_high"  # 90%+ confidence
    HIGH = "high"  # 75-90% confidence
    MODERATE = "moderate"  # 50-75% confidence
    LOW = "low"  # 25-50% confidence
    SPECULATIVE = "speculative"  # <25% confidence


@dataclass
class AttackIntent:
    """Analyzed intent of an attack."""

    primary_goal: str
    secondary_goals: List[str] = field(default_factory=list)
    target_data: List[str] = field(default_factory=list)
    attack_chain_position: str = "unknown"  # initial, intermediate, final
    confidence: AnalysisConfidence = AnalysisConfidence.MODERATE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_goal": self.primary_goal,
            "secondary_goals": self.secondary_goals,
            "target_data": self.target_data,
            "attack_chain_position": self.attack_chain_position,
            "confidence": self.confidence.value,
        }


@dataclass
class MITRETactic:
    """MITRE ATT&CK tactic classification."""

    tactic_id: str
    tactic_name: str
    techniques: List[str] = field(default_factory=list)
    confidence: AnalysisConfidence = AnalysisConfidence.MODERATE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tactic_id": self.tactic_id,
            "tactic_name": self.tactic_name,
            "techniques": self.techniques,
            "confidence": self.confidence.value,
        }


@dataclass
class ImpactAssessment:
    """Assessment of attack impact."""

    severity_score: int  # 1-10
    data_at_risk: List[str] = field(default_factory=list)
    systems_affected: List[str] = field(default_factory=list)
    business_impact: str = ""
    recovery_difficulty: str = "medium"  # low, medium, high, critical
    confidence: AnalysisConfidence = AnalysisConfidence.MODERATE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity_score": self.severity_score,
            "data_at_risk": self.data_at_risk,
            "systems_affected": self.systems_affected,
            "business_impact": self.business_impact,
            "recovery_difficulty": self.recovery_difficulty,
            "confidence": self.confidence.value,
        }


@dataclass
class CodeVulnerability:
    """Identified code vulnerability."""

    vulnerability_id: str
    file_path: str
    line_start: int
    line_end: int
    vulnerability_type: str
    description: str
    severity: int  # 1-5
    exploitability: str = "medium"  # low, medium, high
    cwe_id: Optional[str] = None
    suggested_fix: str = ""
    fix_code: str = ""
    confidence: AnalysisConfidence = AnalysisConfidence.MODERATE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vulnerability_id": self.vulnerability_id,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "vulnerability_type": self.vulnerability_type,
            "description": self.description,
            "severity": self.severity,
            "exploitability": self.exploitability,
            "cwe_id": self.cwe_id,
            "suggested_fix": self.suggested_fix,
            "fix_code": self.fix_code,
            "confidence": self.confidence.value,
        }


@dataclass
class LLMAnalysisResult:
    """Complete LLM analysis result."""

    analysis_id: str
    attack_id: str
    analyzed_at: datetime

    # Analysis components
    intent: Optional[AttackIntent] = None
    impact: Optional[ImpactAssessment] = None
    mitre_tactics: List[MITRETactic] = field(default_factory=list)
    vulnerabilities: List[CodeVulnerability] = field(default_factory=list)

    # Recommendations
    immediate_actions: List[str] = field(default_factory=list)
    long_term_fixes: List[str] = field(default_factory=list)
    prevention_measures: List[str] = field(default_factory=list)

    # Metadata
    reasoning_chain_id: Optional[str] = None
    processing_time_ms: float = 0.0
    llm_model_used: str = ""
    fallback_used: bool = False
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "attack_id": self.attack_id,
            "analyzed_at": self.analyzed_at.isoformat(),
            "intent": self.intent.to_dict() if self.intent else None,
            "impact": self.impact.to_dict() if self.impact else None,
            "mitre_tactics": [t.to_dict() for t in self.mitre_tactics],
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "immediate_actions": self.immediate_actions,
            "long_term_fixes": self.long_term_fixes,
            "prevention_measures": self.prevention_measures,
            "reasoning_chain_id": self.reasoning_chain_id,
            "processing_time_ms": self.processing_time_ms,
            "llm_model_used": self.llm_model_used,
            "fallback_used": self.fallback_used,
        }


# =============================================================================
# LLM Analyzer
# =============================================================================


class LLMAnalyzer:
    """
    LLM-powered attack analyzer using Sage.

    Provides intelligent analysis of security attacks including:
    - Attack intent and goal analysis
    - MITRE ATT&CK classification
    - Code vulnerability discovery
    - Impact assessment
    - Fix recommendations
    """

    # MITRE ATT&CK tactic mapping
    MITRE_TACTICS = {
        "TA0001": "Initial Access",
        "TA0002": "Execution",
        "TA0003": "Persistence",
        "TA0004": "Privilege Escalation",
        "TA0005": "Defense Evasion",
        "TA0006": "Credential Access",
        "TA0007": "Discovery",
        "TA0008": "Lateral Movement",
        "TA0009": "Collection",
        "TA0010": "Exfiltration",
        "TA0011": "Command and Control",
        "TA0040": "Impact",
    }

    # Attack type to MITRE mapping (fallback)
    ATTACK_TYPE_MITRE_MAP = {
        "PROMPT_INJECTION": ["TA0002", "TA0005"],  # Execution, Defense Evasion
        "JAILBREAK": ["TA0005"],  # Defense Evasion
        "DATA_EXFILTRATION": ["TA0009", "TA0010"],  # Collection, Exfiltration
        "RESOURCE_EXHAUSTION": ["TA0040"],  # Impact
        "SQL_INJECTION": ["TA0001", "TA0006"],  # Initial Access, Credential Access
        "XSS": ["TA0001", "TA0002"],  # Initial Access, Execution
        "COMMAND_INJECTION": ["TA0002"],  # Execution
        "PATH_TRAVERSAL": ["TA0009"],  # Collection
        "PRIVILEGE_ESCALATION": ["TA0004"],  # Privilege Escalation
        "UNAUTHORIZED_ACCESS": ["TA0001", "TA0004"],  # Initial Access, Privilege Escalation
    }

    def __init__(
        self,
        sage_agent: Optional[Any] = None,
        use_mock: bool = False,
        codebase_root: Optional[Path] = None,
    ):
        """
        Initialize LLM analyzer.

        Args:
            sage_agent: Pre-configured Sage agent
            use_mock: Use mock responses for testing
            codebase_root: Root path for code analysis
        """
        self._sage: Optional[Any] = sage_agent
        self._use_mock = use_mock
        self._codebase_root = codebase_root or Path.cwd()
        self._initialized = False

        # Try to initialize Sage if not provided
        if not self._sage and SAGE_AVAILABLE and not use_mock:
            try:
                self._sage = create_sage_agent(use_mock=False)
                self._initialized = True
                logger.info("LLM analyzer initialized with Sage agent")
            except Exception as e:
                logger.warning(f"Could not initialize Sage: {e}")
                self._sage = None
        elif self._sage:
            self._initialized = True

    @property
    def is_available(self) -> bool:
        """Check if LLM analysis is available."""
        return self._sage is not None or self._use_mock

    def analyze_attack(
        self,
        attack_id: str,
        attack_type: str,
        severity: int,
        description: str,
        source: str,
        target: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> LLMAnalysisResult:
        """
        Perform comprehensive LLM analysis of an attack.

        Args:
            attack_id: Unique attack identifier
            attack_type: Type of attack
            severity: Attack severity (1-5)
            description: Attack description
            source: Attack source
            target: Attack target
            context: Additional context

        Returns:
            Complete LLM analysis result
        """
        import time
        start_time = time.time()

        analysis_id = f"LLM-{hashlib.md5(f'{attack_id}{datetime.now().isoformat()}'.encode(), usedforsecurity=False).hexdigest()[:12]}"

        result = LLMAnalysisResult(
            analysis_id=analysis_id,
            attack_id=attack_id,
            analyzed_at=datetime.now(),
        )

        # Build context string
        context_str = self._build_context_string(context or {})

        if self._sage and not self._use_mock:
            # Use Sage for analysis
            result = self._analyze_with_sage(
                result, attack_id, attack_type, severity,
                description, source, target, context_str
            )
        else:
            # Fallback to pattern-based analysis
            result = self._analyze_with_fallback(
                result, attack_id, attack_type, severity,
                description, source, target, context_str
            )
            result.fallback_used = True

        result.processing_time_ms = (time.time() - start_time) * 1000
        return result

    def _analyze_with_sage(
        self,
        result: LLMAnalysisResult,
        attack_id: str,
        attack_type: str,
        severity: int,
        description: str,
        source: str,
        target: Optional[str],
        context: str,
    ) -> LLMAnalysisResult:
        """Analyze attack using Sage reasoning agent."""
        try:
            # Build the analysis prompt
            prompt = ATTACK_ANALYSIS_PROMPT.format(
                attack_id=attack_id,
                attack_type=attack_type,
                severity=self._severity_name(severity),
                source=source,
                description=description,
                detected_at=datetime.now().isoformat(),
                context=context or "No additional context provided.",
            )

            # Use Sage for reasoning
            reasoning_chain = self._sage.reason(prompt)

            if reasoning_chain:
                result.reasoning_chain_id = reasoning_chain.chain_id
                result.raw_response = reasoning_chain.final_conclusion
                result.llm_model_used = getattr(self._sage, "model", "sage")

                # Parse the reasoning chain into structured results
                result = self._parse_sage_response(result, reasoning_chain)
            else:
                # Sage returned nothing, use fallback
                logger.warning("Sage returned empty response, using fallback")
                result = self._analyze_with_fallback(
                    result, attack_id, attack_type, severity,
                    description, source, target, context
                )
                result.fallback_used = True

        except Exception as e:
            logger.error(f"Sage analysis failed: {e}")
            # Fallback to pattern-based
            result = self._analyze_with_fallback(
                result, attack_id, attack_type, severity,
                description, source, target, context
            )
            result.fallback_used = True

        return result

    def _analyze_with_fallback(
        self,
        result: LLMAnalysisResult,
        attack_id: str,
        attack_type: str,
        severity: int,
        description: str,
        source: str,
        target: Optional[str],
        context: str,
    ) -> LLMAnalysisResult:
        """Pattern-based fallback analysis when LLM is unavailable."""
        logger.info(f"Using fallback analysis for attack {attack_id}")

        # Analyze intent based on attack type
        result.intent = self._infer_intent(attack_type, description)

        # Assess impact based on severity
        result.impact = self._assess_impact(severity, attack_type, target)

        # Classify MITRE tactics
        result.mitre_tactics = self._classify_mitre(attack_type, description)

        # Generate recommendations
        result.immediate_actions = self._generate_immediate_actions(attack_type, severity)
        result.long_term_fixes = self._generate_long_term_fixes(attack_type)
        result.prevention_measures = self._generate_prevention_measures(attack_type)

        result.llm_model_used = "fallback-patterns"

        return result

    def analyze_code(
        self,
        attack_type: str,
        description: str,
        file_path: Path,
        code_snippet: str,
        line_start: int,
        line_end: int,
    ) -> List[CodeVulnerability]:
        """
        Analyze code for vulnerabilities related to an attack.

        Args:
            attack_type: Type of attack
            description: Attack description
            file_path: Path to the file
            code_snippet: Code to analyze
            line_start: Starting line number
            line_end: Ending line number

        Returns:
            List of identified vulnerabilities
        """
        vulnerabilities = []

        # Determine language from file extension
        language = self._detect_language(file_path)

        if self._sage and not self._use_mock:
            vulnerabilities = self._analyze_code_with_sage(
                attack_type, description, str(file_path),
                code_snippet, line_start, line_end, language
            )
        else:
            vulnerabilities = self._analyze_code_with_fallback(
                attack_type, description, str(file_path),
                code_snippet, line_start, line_end, language
            )

        return vulnerabilities

    def _analyze_code_with_sage(
        self,
        attack_type: str,
        description: str,
        file_path: str,
        code_snippet: str,
        line_start: int,
        line_end: int,
        language: str,
    ) -> List[CodeVulnerability]:
        """Analyze code using Sage."""
        vulnerabilities = []

        try:
            prompt = CODE_VULNERABILITY_PROMPT.format(
                attack_type=attack_type,
                description=description,
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                language=language,
                code_snippet=code_snippet,
            )

            reasoning_chain = self._sage.reason(prompt)

            if reasoning_chain:
                vulnerabilities = self._parse_vulnerability_response(
                    reasoning_chain, file_path, line_start, line_end
                )
        except Exception as e:
            logger.error(f"Sage code analysis failed: {e}")
            vulnerabilities = self._analyze_code_with_fallback(
                attack_type, description, file_path,
                code_snippet, line_start, line_end, language
            )

        return vulnerabilities

    def _analyze_code_with_fallback(
        self,
        attack_type: str,
        description: str,
        file_path: str,
        code_snippet: str,
        line_start: int,
        line_end: int,
        language: str,
    ) -> List[CodeVulnerability]:
        """Pattern-based code vulnerability detection."""
        vulnerabilities = []

        # Common vulnerability patterns by attack type
        patterns = self._get_vulnerability_patterns(attack_type, language)

        for pattern_name, pattern_data in patterns.items():
            regex = pattern_data.get("regex")
            if regex and re.search(regex, code_snippet, re.IGNORECASE):
                vuln_id = f"VULN-{hashlib.md5(f'{file_path}{pattern_name}'.encode(), usedforsecurity=False).hexdigest()[:8]}"
                vulnerabilities.append(CodeVulnerability(
                    vulnerability_id=vuln_id,
                    file_path=file_path,
                    line_start=line_start,
                    line_end=line_end,
                    vulnerability_type=pattern_name,
                    description=pattern_data.get("description", "Potential vulnerability detected"),
                    severity=pattern_data.get("severity", 3),
                    exploitability=pattern_data.get("exploitability", "medium"),
                    cwe_id=pattern_data.get("cwe_id"),
                    suggested_fix=pattern_data.get("fix", "Review and remediate this code"),
                    confidence=AnalysisConfidence.MODERATE,
                ))

        return vulnerabilities

    def classify_mitre_attack(
        self,
        attack_type: str,
        description: str,
        indicators: List[str],
        source: str,
        target: Optional[str] = None,
    ) -> List[MITRETactic]:
        """
        Classify an attack using MITRE ATT&CK framework.

        Args:
            attack_type: Type of attack
            description: Attack description
            indicators: Attack indicators
            source: Attack source
            target: Attack target

        Returns:
            List of classified MITRE tactics
        """
        if self._sage and not self._use_mock:
            try:
                prompt = MITRE_CLASSIFICATION_PROMPT.format(
                    attack_type=attack_type,
                    description=description,
                    indicators=", ".join(indicators) if indicators else "None",
                    source=source,
                    target=target or "Unknown",
                )

                reasoning_chain = self._sage.reason(prompt)
                if reasoning_chain:
                    return self._parse_mitre_response(reasoning_chain)
            except Exception as e:
                logger.warning(f"MITRE classification with Sage failed: {e}")

        # Fallback to pattern-based classification
        return self._classify_mitre(attack_type, description)

    # =========================================================================
    # Response Parsing
    # =========================================================================

    def _parse_sage_response(
        self,
        result: LLMAnalysisResult,
        reasoning_chain: Any,
    ) -> LLMAnalysisResult:
        """Parse Sage reasoning chain into structured results."""
        # Extract information from reasoning steps
        conclusion = reasoning_chain.final_conclusion or ""

        # Parse intent from conclusion
        result.intent = self._extract_intent(conclusion, reasoning_chain)

        # Parse impact assessment
        result.impact = self._extract_impact(conclusion, reasoning_chain)

        # Parse MITRE tactics
        result.mitre_tactics = self._extract_mitre_tactics(conclusion, reasoning_chain)

        # Parse recommendations
        result.immediate_actions = self._extract_list(conclusion, "immediate", "action")
        result.long_term_fixes = self._extract_list(conclusion, "long-term", "fix")
        result.prevention_measures = self._extract_list(conclusion, "prevent", "measure")

        return result

    def _parse_vulnerability_response(
        self,
        reasoning_chain: Any,
        file_path: str,
        line_start: int,
        line_end: int,
    ) -> List[CodeVulnerability]:
        """Parse Sage response for vulnerabilities."""
        vulnerabilities = []
        conclusion = reasoning_chain.final_conclusion or ""

        # Look for vulnerability patterns in the response
        vuln_patterns = [
            (r"injection", "INJECTION", 4, "CWE-79"),
            (r"cross-site", "XSS", 4, "CWE-79"),
            (r"sql\s*injection", "SQL_INJECTION", 5, "CWE-89"),
            (r"command\s*injection", "COMMAND_INJECTION", 5, "CWE-78"),
            (r"path\s*traversal", "PATH_TRAVERSAL", 4, "CWE-22"),
            (r"authentication", "AUTH_BYPASS", 5, "CWE-287"),
            (r"authorization", "AUTHZ_BYPASS", 4, "CWE-285"),
            (r"buffer\s*overflow", "BUFFER_OVERFLOW", 5, "CWE-120"),
            (r"insecure\s*deserial", "INSECURE_DESERIAL", 5, "CWE-502"),
            (r"hardcoded", "HARDCODED_SECRET", 3, "CWE-798"),
        ]

        for pattern, vuln_type, severity, cwe_id in vuln_patterns:
            if re.search(pattern, conclusion, re.IGNORECASE):
                vuln_id = f"VULN-{hashlib.md5(f'{file_path}{vuln_type}'.encode(), usedforsecurity=False).hexdigest()[:8]}"
                vulnerabilities.append(CodeVulnerability(
                    vulnerability_id=vuln_id,
                    file_path=file_path,
                    line_start=line_start,
                    line_end=line_end,
                    vulnerability_type=vuln_type,
                    description=self._extract_description(conclusion, pattern),
                    severity=severity,
                    cwe_id=cwe_id,
                    suggested_fix=self._extract_fix(conclusion),
                    confidence=AnalysisConfidence.HIGH,
                ))

        return vulnerabilities

    def _parse_mitre_response(self, reasoning_chain: Any) -> List[MITRETactic]:
        """Parse Sage response for MITRE classification."""
        tactics = []
        conclusion = reasoning_chain.final_conclusion or ""

        # Look for tactic names in response
        for tactic_id, tactic_name in self.MITRE_TACTICS.items():
            if tactic_name.lower() in conclusion.lower():
                # Extract techniques mentioned with this tactic
                techniques = self._extract_techniques(conclusion, tactic_name)
                tactics.append(MITRETactic(
                    tactic_id=tactic_id,
                    tactic_name=tactic_name,
                    techniques=techniques,
                    confidence=AnalysisConfidence.HIGH,
                ))

        return tactics

    # =========================================================================
    # Extraction Helpers
    # =========================================================================

    def _extract_intent(
        self,
        conclusion: str,
        reasoning_chain: Any,
    ) -> AttackIntent:
        """Extract attack intent from analysis."""
        # Look for goal-related keywords
        goal_patterns = [
            (r"goal\s*(?:is|:)\s*([^.]+)", "primary"),
            (r"objective\s*(?:is|:)\s*([^.]+)", "primary"),
            (r"attacker\s*(?:is\s+)?trying\s+to\s+([^.]+)", "primary"),
            (r"attempting\s+to\s+([^.]+)", "primary"),
        ]

        primary_goal = "Unknown attack goal"
        for pattern, _ in goal_patterns:
            match = re.search(pattern, conclusion, re.IGNORECASE)
            if match:
                primary_goal = match.group(1).strip()
                break

        # Look for data targets
        data_targets = self._extract_list(conclusion, "data", "target")
        if not data_targets:
            data_targets = self._extract_list(conclusion, "access", "to")

        return AttackIntent(
            primary_goal=primary_goal,
            target_data=data_targets,
            confidence=AnalysisConfidence.HIGH if reasoning_chain else AnalysisConfidence.MODERATE,
        )

    def _extract_impact(
        self,
        conclusion: str,
        reasoning_chain: Any,
    ) -> ImpactAssessment:
        """Extract impact assessment from analysis."""
        # Look for severity indicators
        severity_keywords = {
            "critical": 10,
            "severe": 9,
            "high": 7,
            "significant": 6,
            "moderate": 5,
            "medium": 5,
            "low": 3,
            "minimal": 2,
        }

        severity_score = 5  # Default
        for keyword, score in severity_keywords.items():
            if keyword in conclusion.lower():
                severity_score = score
                break

        # Extract affected systems
        systems = self._extract_list(conclusion, "system", "affect")
        data_risk = self._extract_list(conclusion, "data", "risk")

        return ImpactAssessment(
            severity_score=severity_score,
            data_at_risk=data_risk,
            systems_affected=systems,
            confidence=AnalysisConfidence.HIGH if reasoning_chain else AnalysisConfidence.MODERATE,
        )

    def _extract_mitre_tactics(
        self,
        conclusion: str,
        reasoning_chain: Any,
    ) -> List[MITRETactic]:
        """Extract MITRE tactics from analysis."""
        tactics = []

        for tactic_id, tactic_name in self.MITRE_TACTICS.items():
            if tactic_name.lower() in conclusion.lower():
                tactics.append(MITRETactic(
                    tactic_id=tactic_id,
                    tactic_name=tactic_name,
                    confidence=AnalysisConfidence.HIGH,
                ))

        return tactics

    def _extract_techniques(self, text: str, tactic: str) -> List[str]:
        """Extract MITRE techniques for a given tactic."""
        techniques = []

        # Look for technique IDs (T####)
        technique_pattern = r"T\d{4}(?:\.\d{3})?"
        matches = re.findall(technique_pattern, text)
        techniques.extend(matches)

        return list(set(techniques))

    def _extract_list(self, text: str, keyword1: str, keyword2: str) -> List[str]:
        """Extract a list of items near keywords."""
        items = []

        # Look for bullet points near keywords
        if keyword1.lower() in text.lower():
            lines = text.split("\n")
            capture = False
            for line in lines:
                if keyword1.lower() in line.lower() or keyword2.lower() in line.lower():
                    capture = True
                elif capture:
                    if line.strip().startswith("-") or line.strip().startswith("*"):
                        items.append(line.strip().lstrip("-* "))
                    elif line.strip() and not line.strip().startswith("#"):
                        items.append(line.strip())
                    else:
                        capture = False
                        if len(items) >= 3:
                            break

        return items[:10]  # Limit to 10 items

    def _extract_description(self, text: str, pattern: str) -> str:
        """Extract description near a pattern."""
        match = re.search(rf"{pattern}[^.]*\.?([^.]+\.)?", text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
        return "Vulnerability detected matching pattern"

    def _extract_fix(self, text: str) -> str:
        """Extract fix recommendation from text."""
        fix_patterns = [
            r"fix\s*(?:is|:)\s*([^.]+)",
            r"recommend\s*(?:is|:)\s*([^.]+)",
            r"solution\s*(?:is|:)\s*([^.]+)",
            r"should\s+([^.]+)",
        ]

        for pattern in fix_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return "Review and remediate the identified vulnerability"

    # =========================================================================
    # Fallback Helpers
    # =========================================================================

    def _infer_intent(self, attack_type: str, description: str) -> AttackIntent:
        """Infer attack intent from type and description."""
        intent_map = {
            "PROMPT_INJECTION": AttackIntent(
                primary_goal="Manipulate AI behavior through crafted prompts",
                secondary_goals=["Bypass safety controls", "Extract sensitive data"],
                target_data=["AI model responses", "System prompts"],
            ),
            "JAILBREAK": AttackIntent(
                primary_goal="Bypass AI safety guardrails",
                secondary_goals=["Generate prohibited content", "Reveal system information"],
                target_data=["AI restrictions", "Safety parameters"],
            ),
            "DATA_EXFILTRATION": AttackIntent(
                primary_goal="Extract sensitive data from the system",
                secondary_goals=["Establish persistence", "Cover tracks"],
                target_data=["User data", "System credentials", "Configuration"],
            ),
            "SQL_INJECTION": AttackIntent(
                primary_goal="Execute unauthorized database queries",
                secondary_goals=["Extract data", "Modify data", "Gain access"],
                target_data=["Database contents", "User credentials"],
            ),
            "COMMAND_INJECTION": AttackIntent(
                primary_goal="Execute arbitrary system commands",
                secondary_goals=["Gain shell access", "Install malware"],
                target_data=["System access", "File system"],
            ),
        }

        return intent_map.get(attack_type, AttackIntent(
            primary_goal=f"Execute {attack_type.replace('_', ' ').lower()} attack",
            confidence=AnalysisConfidence.LOW,
        ))

    def _assess_impact(
        self,
        severity: int,
        attack_type: str,
        target: Optional[str],
    ) -> ImpactAssessment:
        """Assess attack impact based on severity and type."""
        severity_score = min(severity * 2, 10)

        recovery_map = {
            1: "low",
            2: "low",
            3: "medium",
            4: "high",
            5: "critical",
        }

        return ImpactAssessment(
            severity_score=severity_score,
            data_at_risk=[target] if target else ["Unknown"],
            recovery_difficulty=recovery_map.get(severity, "medium"),
            confidence=AnalysisConfidence.MODERATE,
        )

    def _classify_mitre(self, attack_type: str, description: str) -> List[MITRETactic]:
        """Classify attack using MITRE ATT&CK (fallback)."""
        tactics = []

        tactic_ids = self.ATTACK_TYPE_MITRE_MAP.get(attack_type, [])
        for tactic_id in tactic_ids:
            if tactic_id in self.MITRE_TACTICS:
                tactics.append(MITRETactic(
                    tactic_id=tactic_id,
                    tactic_name=self.MITRE_TACTICS[tactic_id],
                    confidence=AnalysisConfidence.MODERATE,
                ))

        return tactics

    def _generate_immediate_actions(self, attack_type: str, severity: int) -> List[str]:
        """Generate immediate action recommendations."""
        actions = []

        if severity >= 4:
            actions.append("Immediately isolate affected systems")
            actions.append("Alert security team and incident response")

        if severity >= 3:
            actions.append("Block the attack source if identifiable")
            actions.append("Preserve logs and evidence")

        # Type-specific actions
        type_actions = {
            "PROMPT_INJECTION": [
                "Review and strengthen input validation",
                "Audit recent AI interactions",
            ],
            "DATA_EXFILTRATION": [
                "Monitor network traffic for ongoing exfiltration",
                "Revoke potentially compromised credentials",
            ],
            "SQL_INJECTION": [
                "Review database access logs",
                "Check for data modifications",
            ],
        }

        actions.extend(type_actions.get(attack_type, []))
        return actions[:5]

    def _generate_long_term_fixes(self, attack_type: str) -> List[str]:
        """Generate long-term fix recommendations."""
        fixes = {
            "PROMPT_INJECTION": [
                "Implement robust input sanitization for AI prompts",
                "Add output filtering and safety guardrails",
                "Regular security audits of AI interactions",
            ],
            "DATA_EXFILTRATION": [
                "Implement Data Loss Prevention (DLP) controls",
                "Enhance network monitoring and alerting",
                "Regular access control reviews",
            ],
            "SQL_INJECTION": [
                "Use parameterized queries exclusively",
                "Implement input validation at all entry points",
                "Regular code reviews for injection vulnerabilities",
            ],
            "COMMAND_INJECTION": [
                "Avoid shell execution where possible",
                "Use allowlists for command arguments",
                "Implement principle of least privilege",
            ],
        }

        return fixes.get(attack_type, [
            "Review and strengthen security controls",
            "Implement defense in depth",
            "Regular security assessments",
        ])

    def _generate_prevention_measures(self, attack_type: str) -> List[str]:
        """Generate prevention recommendations."""
        measures = [
            "Implement comprehensive logging and monitoring",
            "Regular security awareness training",
            "Maintain up-to-date security patches",
        ]

        type_measures = {
            "PROMPT_INJECTION": [
                "Implement content filtering for AI inputs",
                "Use structured prompts with strict templates",
            ],
            "DATA_EXFILTRATION": [
                "Implement data classification and handling policies",
                "Deploy endpoint detection and response (EDR)",
            ],
        }

        measures.extend(type_measures.get(attack_type, []))
        return measures[:5]

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _build_context_string(self, context: Dict[str, Any]) -> str:
        """Build context string from dictionary."""
        if not context:
            return "No additional context provided."

        lines = []
        for key, value in context.items():
            if isinstance(value, list):
                lines.append(f"- **{key}:** {', '.join(str(v) for v in value)}")
            else:
                lines.append(f"- **{key}:** {value}")

        return "\n".join(lines)

    def _severity_name(self, severity: int) -> str:
        """Convert severity number to name."""
        names = {
            1: "LOW",
            2: "MEDIUM",
            3: "HIGH",
            4: "CRITICAL",
            5: "CATASTROPHIC",
        }
        return names.get(severity, "UNKNOWN")

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".c": "c",
            ".cpp": "cpp",
            ".rb": "ruby",
            ".php": "php",
            ".sql": "sql",
            ".sh": "bash",
        }
        return ext_map.get(file_path.suffix.lower(), "text")

    def _get_vulnerability_patterns(
        self,
        attack_type: str,
        language: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Get vulnerability patterns for detection."""
        # Common patterns across languages
        common_patterns = {
            "SQL_INJECTION": {
                "regex": r"(execute|query|cursor)\s*\([^)]*['\"].*%.*['\"]",
                "description": "Potential SQL injection - string formatting in query",
                "severity": 5,
                "cwe_id": "CWE-89",
                "fix": "Use parameterized queries instead of string formatting",
            },
            "COMMAND_INJECTION": {
                "regex": r"(os\.system|subprocess\.call|exec|shell_exec|system)\s*\(",
                "description": "Potential command injection - shell execution detected",
                "severity": 5,
                "cwe_id": "CWE-78",
                "fix": "Use subprocess with shell=False and sanitize inputs",
            },
            "PATH_TRAVERSAL": {
                "regex": r"\.\./|\.\.\\",
                "description": "Potential path traversal pattern detected",
                "severity": 4,
                "cwe_id": "CWE-22",
                "fix": "Validate and sanitize file paths, use allowlists",
            },
            "HARDCODED_SECRET": {
                "regex": r"(password|secret|api_key|token)\s*=\s*['\"][^'\"]{8,}['\"]",
                "description": "Hardcoded secret or credential detected",
                "severity": 3,
                "cwe_id": "CWE-798",
                "fix": "Use environment variables or secrets management",
            },
        }

        # Return patterns relevant to attack type
        if attack_type in common_patterns:
            return {attack_type: common_patterns[attack_type]}

        return common_patterns


# =============================================================================
# Factory Functions
# =============================================================================


def create_llm_analyzer(
    sage_agent: Optional[Any] = None,
    use_mock: bool = False,
    codebase_root: Optional[Path] = None,
) -> LLMAnalyzer:
    """
    Create an LLM analyzer instance.

    Args:
        sage_agent: Pre-configured Sage agent
        use_mock: Use mock responses for testing
        codebase_root: Root path for code analysis

    Returns:
        Configured LLMAnalyzer
    """
    return LLMAnalyzer(
        sage_agent=sage_agent,
        use_mock=use_mock,
        codebase_root=codebase_root,
    )
