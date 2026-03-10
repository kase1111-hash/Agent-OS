"""
Instruction Integrity Validator

Validates the integrity of instruction files (prompts, constitutional rules,
attack patterns) before they are loaded and used. Prevents tampering by
verifying HMAC-SHA256 signatures against stored hashes.

This addresses Finding #7 (S3 instruction integrity validation not implemented)
from the Agentic Security Audit v3.0.
"""

import hashlib
import hmac
import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class InstructionIntegrityValidator:
    """
    Validates instruction file integrity using HMAC-SHA256 signatures.

    Computes and stores hashes of instruction files on first load.
    On subsequent loads, verifies the hash matches before accepting
    the instruction content.
    """

    def __init__(self, key: Optional[bytes] = None, hash_store_path: Optional[Path] = None):
        """
        Args:
            key: HMAC key for signing. If None, derives from machine identity.
            hash_store_path: Path to persist known-good hashes.
        """
        self._key = key or self._derive_key()
        self._hash_store_path = hash_store_path or (
            Path.home() / ".agent-os" / "data" / ".instruction_hashes.json"
        )
        self._known_hashes: Dict[str, str] = {}
        self._load_hash_store()

    def _derive_key(self) -> bytes:
        """Derive a machine-specific key for HMAC signing."""
        import getpass
        import platform

        machine_id = f"{platform.node()}:{getpass.getuser()}:instruction-integrity"
        return hashlib.sha256(machine_id.encode()).digest()

    def _load_hash_store(self) -> None:
        """Load known-good hashes from disk."""
        try:
            if self._hash_store_path.exists():
                data = json.loads(self._hash_store_path.read_text())
                self._known_hashes = data.get("hashes", {})
                logger.debug("Loaded %d instruction hashes", len(self._known_hashes))
        except Exception as e:
            logger.warning("Failed to load instruction hash store: %s", e)
            self._known_hashes = {}

    def _save_hash_store(self) -> None:
        """Persist known-good hashes to disk."""
        try:
            self._hash_store_path.parent.mkdir(parents=True, exist_ok=True)
            self._hash_store_path.write_text(
                json.dumps({"hashes": self._known_hashes}, indent=2)
            )
            self._hash_store_path.chmod(0o600)
        except Exception as e:
            logger.warning("Failed to save instruction hash store: %s", e)

    def compute_hash(self, content: str) -> str:
        """Compute HMAC-SHA256 of instruction content."""
        return hmac.new(self._key, content.encode(), hashlib.sha256).hexdigest()

    def register(self, instruction_id: str, content: str) -> str:
        """
        Register an instruction's known-good hash.

        Call this at build/deployment time to establish the baseline.

        Returns:
            The computed hash.
        """
        content_hash = self.compute_hash(content)
        self._known_hashes[instruction_id] = content_hash
        self._save_hash_store()
        logger.info("Registered instruction hash: %s → %s...", instruction_id, content_hash[:16])
        return content_hash

    def validate(self, instruction_id: str, content: str) -> bool:
        """
        Validate instruction content against its known-good hash.

        Returns:
            True if content matches the stored hash.
            If no hash is stored (first load), registers it and returns True.
        """
        content_hash = self.compute_hash(content)

        if instruction_id not in self._known_hashes:
            # First load — establish baseline
            logger.info(
                "No known hash for instruction '%s'. Registering baseline.",
                instruction_id,
            )
            self._known_hashes[instruction_id] = content_hash
            self._save_hash_store()
            return True

        expected = self._known_hashes[instruction_id]
        if hmac.compare_digest(content_hash, expected):
            return True

        logger.critical(
            "SECURITY: Instruction '%s' INTEGRITY CHECK FAILED. "
            "Expected hash: %s..., Got: %s... "
            "Instruction may have been tampered with.",
            instruction_id,
            expected[:16],
            content_hash[:16],
        )
        return False

    def validate_file(self, file_path: Path) -> bool:
        """Validate an instruction file's integrity."""
        try:
            content = file_path.read_text()
            instruction_id = str(file_path)
            return self.validate(instruction_id, content)
        except FileNotFoundError:
            logger.error("Instruction file not found: %s", file_path)
            return False

    def remove(self, instruction_id: str) -> None:
        """Remove a registered hash (for legitimate updates)."""
        self._known_hashes.pop(instruction_id, None)
        self._save_hash_store()
