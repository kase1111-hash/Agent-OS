"""
Outbound Secret Scanner for Agent OS.

Scans inter-agent messages and outbound communications for credential
leakage. Reuses patterns from SensitiveDataRedactor and adds additional
detections for high-entropy strings and Anthropic keys.

V2-3: Outbound secret scanning — prevents agents from transmitting
API keys, tokens, passwords, or other credential material in messages.
"""

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Pattern

from src.core.exceptions import SecurityError

logger = logging.getLogger(__name__)


class SecretLeakageError(SecurityError):
    """Raised when a message contains detected secrets."""

    def __init__(
        self,
        message: str = "Secret detected in outbound message",
        patterns_matched: Optional[List[str]] = None,
        source_agent: Optional[str] = None,
    ):
        self.patterns_matched = patterns_matched or []
        self.source_agent = source_agent
        super().__init__(message)


@dataclass
class ScanResult:
    """Result of scanning content for secrets."""

    found_secrets: bool
    patterns_matched: List[str] = field(default_factory=list)
    redacted_content: str = ""
    scan_time_ms: float = 0.0


@dataclass
class _ScanPattern:
    """Internal pattern for secret detection."""

    pattern: Pattern
    name: str
    replacement: str


class SecretScanner:
    """
    Scans text content for embedded secrets and credentials.

    Can operate in two modes:
    - scan(): Returns a ScanResult with findings (non-blocking)
    - scan_and_block(): Raises SecretLeakageError if secrets found (blocking)
    """

    def __init__(self):
        self._patterns: List[_ScanPattern] = [
            # API keys — generic assignment pattern
            _ScanPattern(
                re.compile(
                    r'(?:api[_-]?key|apikey)[=:]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
                    re.I,
                ),
                "api_key_assignment",
                "[BLOCKED_API_KEY]",
            ),
            # OpenAI keys
            _ScanPattern(
                re.compile(r"sk-[a-zA-Z0-9]{20,}"),
                "openai_key",
                "[BLOCKED_OPENAI_KEY]",
            ),
            # Anthropic keys
            _ScanPattern(
                re.compile(r"sk-ant-api03-[a-zA-Z0-9_-]{20,}"),
                "anthropic_key",
                "[BLOCKED_ANTHROPIC_KEY]",
            ),
            # Hugging Face tokens
            _ScanPattern(
                re.compile(r"hf_[a-zA-Z0-9]{20,}"),
                "huggingface_token",
                "[BLOCKED_HF_TOKEN]",
            ),
            # GitHub tokens
            _ScanPattern(
                re.compile(r"ghp_[a-zA-Z0-9]{20,}"),
                "github_token",
                "[BLOCKED_GITHUB_TOKEN]",
            ),
            # GitHub OAuth / app tokens
            _ScanPattern(
                re.compile(r"gho_[a-zA-Z0-9]{20,}"),
                "github_oauth_token",
                "[BLOCKED_GITHUB_OAUTH]",
            ),
            # AWS access keys
            _ScanPattern(
                re.compile(r"AKIA[0-9A-Z]{16}"),
                "aws_access_key",
                "[BLOCKED_AWS_KEY]",
            ),
            # AWS secret keys (40 char base64)
            _ScanPattern(
                re.compile(r"(?:aws_secret_access_key|secret_key)[=:]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?", re.I),
                "aws_secret_key",
                "[BLOCKED_AWS_SECRET]",
            ),
            # Bearer tokens
            _ScanPattern(
                re.compile(r"Bearer\s+[a-zA-Z0-9_.-]{20,}", re.I),
                "bearer_token",
                "Bearer [BLOCKED]",
            ),
            # Basic auth
            _ScanPattern(
                re.compile(r"Basic\s+[a-zA-Z0-9+/=]{20,}", re.I),
                "basic_auth",
                "Basic [BLOCKED]",
            ),
            # JWT tokens
            _ScanPattern(
                re.compile(r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*"),
                "jwt_token",
                "[BLOCKED_JWT]",
            ),
            # Private keys (PEM blocks)
            _ScanPattern(
                re.compile(
                    r"-----BEGIN\s+(?:RSA\s+|EC\s+|ED25519\s+|OPENSSH\s+)?PRIVATE\s+KEY-----"
                    r"[\s\S]*?"
                    r"-----END\s+(?:RSA\s+|EC\s+|ED25519\s+|OPENSSH\s+)?PRIVATE\s+KEY-----"
                ),
                "private_key",
                "[BLOCKED_PRIVATE_KEY]",
            ),
            # Password assignments
            _ScanPattern(
                re.compile(r'(?:password|passwd|pwd)\s*[=:]\s*["\']?[^"\'\s&]{8,}["\']?', re.I),
                "password_assignment",
                "[BLOCKED_PASSWORD]",
            ),
            # Database connection strings with credentials
            _ScanPattern(
                re.compile(r"(?:mongodb|postgresql|mysql|redis|amqp):\/\/[^:]+:[^@]+@", re.I),
                "db_connection_string",
                "[BLOCKED_CONNECTION_STRING]",
            ),
            # Credit card numbers
            _ScanPattern(
                re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
                "credit_card",
                "[BLOCKED_CARD]",
            ),
            # SSN
            _ScanPattern(
                re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
                "ssn",
                "[BLOCKED_SSN]",
            ),
        ]

        # Minimum entropy threshold for high-entropy string detection
        self._entropy_threshold = 4.5
        self._entropy_min_length = 20

        # Track blocked attempts for monitoring
        self._blocked_count = 0
        self._last_blocked_at: Optional[datetime] = None

    @staticmethod
    def _shannon_entropy(s: str) -> float:
        """Calculate Shannon entropy of a string."""
        if not s:
            return 0.0
        length = len(s)
        freq: Dict[str, int] = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1
        return -sum(
            (count / length) * math.log2(count / length)
            for count in freq.values()
        )

    def _check_high_entropy(self, text: str) -> Optional[str]:
        """
        Check for high-entropy strings that may be secrets.

        Returns the matched substring if found, None otherwise.
        Looks for contiguous alphanumeric strings with entropy > threshold.
        """
        # Match long alphanumeric+special strings that look like tokens/keys
        for match in re.finditer(r'[a-zA-Z0-9_/+=.-]{20,}', text):
            candidate = match.group()
            # Skip common non-secret patterns
            if candidate.startswith("http") or candidate.startswith("file"):
                continue
            # Skip paths
            if "/" in candidate and candidate.count("/") > 2:
                continue
            entropy = self._shannon_entropy(candidate)
            if entropy >= self._entropy_threshold:
                return candidate
        return None

    def scan(self, content: str) -> ScanResult:
        """
        Scan content for secrets. Returns a ScanResult.

        This is the non-blocking version — it reports findings but
        does not raise exceptions.
        """
        import time

        start = time.monotonic()

        if not content or not isinstance(content, str):
            return ScanResult(found_secrets=False, redacted_content=str(content))

        patterns_matched: List[str] = []
        redacted = content

        # Check each pattern
        for sp in self._patterns:
            if sp.pattern.search(content):
                patterns_matched.append(sp.name)
                redacted = sp.pattern.sub(sp.replacement, redacted)

        # Check for high-entropy strings (only if no patterns already matched,
        # to avoid double-flagging)
        if not patterns_matched:
            high_ent = self._check_high_entropy(content)
            if high_ent:
                patterns_matched.append("high_entropy_string")
                redacted = redacted.replace(high_ent, "[BLOCKED_HIGH_ENTROPY]")

        elapsed_ms = (time.monotonic() - start) * 1000

        return ScanResult(
            found_secrets=len(patterns_matched) > 0,
            patterns_matched=patterns_matched,
            redacted_content=redacted,
            scan_time_ms=elapsed_ms,
        )

    def scan_and_block(self, content: str, source_agent: str = "unknown") -> str:
        """
        Scan content and raise SecretLeakageError if secrets are detected.

        Returns the original content unchanged if clean.

        Args:
            content: The text content to scan
            source_agent: Name of the agent that produced this content

        Returns:
            The original content if no secrets found

        Raises:
            SecretLeakageError: If secrets are detected
        """
        result = self.scan(content)

        if result.found_secrets:
            self._blocked_count += 1
            self._last_blocked_at = datetime.utcnow()

            logger.critical(
                "BLOCKED: Agent '%s' attempted to send message containing secrets. "
                "Patterns matched: %s",
                source_agent,
                result.patterns_matched,
            )

            raise SecretLeakageError(
                message=(
                    f"Secret detected in outbound message from agent '{source_agent}'. "
                    f"Patterns matched: {', '.join(result.patterns_matched)}"
                ),
                patterns_matched=result.patterns_matched,
                source_agent=source_agent,
            )

        return content

    def scan_dict(self, data: Dict[str, Any], path: str = "") -> ScanResult:
        """
        Recursively scan a dictionary for secrets.

        Args:
            data: Dictionary to scan
            path: Dot-separated path for reporting (internal)

        Returns:
            Aggregated ScanResult
        """
        all_matched: List[str] = []

        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            if isinstance(value, str):
                result = self.scan(value)
                if result.found_secrets:
                    all_matched.extend(
                        f"{current_path}:{p}" for p in result.patterns_matched
                    )
            elif isinstance(value, dict):
                result = self.scan_dict(value, current_path)
                all_matched.extend(result.patterns_matched)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    item_path = f"{current_path}[{i}]"
                    if isinstance(item, str):
                        result = self.scan(item)
                        if result.found_secrets:
                            all_matched.extend(
                                f"{item_path}:{p}" for p in result.patterns_matched
                            )
                    elif isinstance(item, dict):
                        result = self.scan_dict(item, item_path)
                        all_matched.extend(result.patterns_matched)

        return ScanResult(
            found_secrets=len(all_matched) > 0,
            patterns_matched=all_matched,
            redacted_content="",
        )

    @property
    def blocked_count(self) -> int:
        """Number of messages blocked since initialization."""
        return self._blocked_count

    @property
    def last_blocked_at(self) -> Optional[datetime]:
        """Timestamp of last blocked message."""
        return self._last_blocked_at


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_scanner: Optional[SecretScanner] = None


def get_secret_scanner() -> SecretScanner:
    """Get the global SecretScanner instance."""
    global _scanner
    if _scanner is None:
        _scanner = SecretScanner()
    return _scanner
