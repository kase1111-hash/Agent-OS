"""
Smith Agent Integrity Checker

Provides mutual validation for the Smith security agent to prevent
single-point-of-failure scenarios. If Smith is compromised or unavailable,
the system fails closed (LOCKDOWN) rather than silently bypassing security.

This addresses Finding #6 (Smith agent single point of failure) from the
Agentic Security Audit v3.0.
"""

import hashlib
import inspect
import logging
import threading
import time
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger(__name__)


class SmithStatus(Enum):
    """Health status of the Smith agent."""

    HEALTHY = auto()
    DEGRADED = auto()
    UNAVAILABLE = auto()
    TAMPERED = auto()


class SmithIntegrityChecker:
    """
    Validates the integrity and availability of the Smith security agent.

    Computes a hash of the Smith agent's source code at startup and
    periodically re-validates. If Smith is unavailable or tampered,
    triggers fail-closed LOCKDOWN behavior.
    """

    # Maximum consecutive failures before declaring UNAVAILABLE
    MAX_FAILURES = 3
    # How often to re-validate (seconds)
    CHECK_INTERVAL = 60

    def __init__(self):
        self._known_hash: Optional[str] = None
        self._status = SmithStatus.HEALTHY
        self._consecutive_failures = 0
        self._last_check = 0.0
        self._lock = threading.Lock()

    def initialize(self, smith_agent) -> bool:
        """
        Initialize the checker by computing Smith's known-good code hash.

        Args:
            smith_agent: The Smith agent instance to validate.

        Returns:
            True if initialization succeeded.
        """
        try:
            source = inspect.getsource(smith_agent.__class__)
            self._known_hash = hashlib.sha256(source.encode()).hexdigest()
            self._status = SmithStatus.HEALTHY
            self._consecutive_failures = 0
            self._last_check = time.time()
            logger.info("Smith integrity checker initialized: hash=%s...", self._known_hash[:16])
            return True
        except Exception as e:
            logger.error("Failed to initialize Smith integrity checker: %s", e)
            self._status = SmithStatus.UNAVAILABLE
            return False

    def validate(self, smith_agent) -> SmithStatus:
        """
        Validate Smith agent integrity.

        Checks that:
        1. Smith agent instance is available (not None)
        2. Source code hash matches known-good hash

        Returns:
            Current SmithStatus.
        """
        with self._lock:
            now = time.time()
            if now - self._last_check < self.CHECK_INTERVAL:
                return self._status

            self._last_check = now

            if smith_agent is None:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.MAX_FAILURES:
                    self._status = SmithStatus.UNAVAILABLE
                    logger.error(
                        "Smith agent UNAVAILABLE after %d consecutive failures. "
                        "Triggering fail-closed LOCKDOWN.",
                        self._consecutive_failures,
                    )
                else:
                    self._status = SmithStatus.DEGRADED
                    logger.warning(
                        "Smith agent not responding (failure %d/%d)",
                        self._consecutive_failures,
                        self.MAX_FAILURES,
                    )
                return self._status

            try:
                source = inspect.getsource(smith_agent.__class__)
                current_hash = hashlib.sha256(source.encode()).hexdigest()

                if current_hash != self._known_hash:
                    self._status = SmithStatus.TAMPERED
                    logger.critical(
                        "SECURITY: Smith agent code hash MISMATCH. "
                        "Expected: %s..., Got: %s... "
                        "Triggering fail-closed LOCKDOWN.",
                        self._known_hash[:16],
                        current_hash[:16],
                    )
                    return self._status

                self._status = SmithStatus.HEALTHY
                self._consecutive_failures = 0
                return self._status

            except Exception as e:
                self._consecutive_failures += 1
                self._status = SmithStatus.DEGRADED
                logger.warning("Smith integrity check failed: %s", e)
                return self._status

    def is_safe_to_proceed(self, smith_agent) -> bool:
        """
        Check if it's safe to route security-sensitive requests.

        Returns False (fail-closed) if Smith is UNAVAILABLE or TAMPERED.
        """
        status = self.validate(smith_agent)
        if status in (SmithStatus.UNAVAILABLE, SmithStatus.TAMPERED):
            return False
        return True
