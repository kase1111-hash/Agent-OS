"""
Agent Identity System

Provides cryptographic identity for agents using Ed25519 keypairs.
Each agent gets a unique keypair at initialization. Messages are signed
with the sender's private key and verified by recipients using the
registered public key.

This addresses V4 (Bot-to-Bot Social Engineering) and V8 (Identity Spoofing)
from the Moltbook/OpenClaw vulnerability analysis.
"""

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Try to use cryptography library for Ed25519
_HAS_CRYPTO = False
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )
    from cryptography.exceptions import InvalidSignature

    _HAS_CRYPTO = True
except ImportError:
    logger.warning(
        "cryptography library not available — agent identity uses HMAC fallback"
    )


@dataclass
class AgentIdentity:
    """
    Cryptographic identity for an agent.

    When Ed25519 is available, uses asymmetric keypairs.
    Falls back to HMAC-based signing using a shared secret.
    """

    agent_name: str
    created_at: datetime = field(default_factory=datetime.now)
    _private_key: object = field(default=None, repr=False)
    _public_key_bytes: bytes = field(default=b"", repr=False)
    _hmac_secret: bytes = field(default=b"", repr=False)

    def __post_init__(self):
        if _HAS_CRYPTO:
            self._private_key = Ed25519PrivateKey.generate()
            self._public_key_bytes = self._private_key.public_key().public_bytes(
                Encoding.Raw, PublicFormat.Raw
            )
        else:
            import os

            self._hmac_secret = os.urandom(32)
            self._public_key_bytes = hashlib.sha256(self._hmac_secret).digest()

    def sign(self, message: bytes) -> bytes:
        """Sign a message with this agent's private key."""
        if _HAS_CRYPTO and self._private_key is not None:
            return self._private_key.sign(message)
        else:
            import hmac

            return hmac.new(self._hmac_secret, message, hashlib.sha256).digest()

    @property
    def public_key(self) -> bytes:
        """Get the public key bytes for this agent."""
        return self._public_key_bytes

    def sign_message_payload(self, request_id: str, source: str, content_hash: str) -> bytes:
        """Sign a canonical message representation."""
        canonical = f"{request_id}:{source}:{content_hash}".encode()
        return self.sign(canonical)


class AgentIdentityRegistry:
    """
    Registry of agent identities and public keys.

    Provides:
    - Agent registration with keypair generation
    - Message signature verification
    - Public key lookup
    """

    def __init__(self):
        self._identities: Dict[str, AgentIdentity] = {}
        self._public_keys: Dict[str, bytes] = {}
        self._lock = threading.RLock()

    def register(self, agent_name: str) -> AgentIdentity:
        """
        Register an agent and generate its identity.

        Args:
            agent_name: Unique agent name

        Returns:
            AgentIdentity with keypair

        Raises:
            ValueError: If agent already registered
        """
        with self._lock:
            if agent_name in self._identities:
                raise ValueError(f"Agent '{agent_name}' already registered")

            identity = AgentIdentity(agent_name=agent_name)
            self._identities[agent_name] = identity
            self._public_keys[agent_name] = identity.public_key

            logger.info(
                f"Registered agent identity: {agent_name} "
                f"(key={identity.public_key[:8].hex()}...)"
            )
            return identity

    def get_identity(self, agent_name: str) -> Optional[AgentIdentity]:
        """Get an agent's identity (including private key — use carefully)."""
        with self._lock:
            return self._identities.get(agent_name)

    def get_public_key(self, agent_name: str) -> Optional[bytes]:
        """Get an agent's public key for verification."""
        with self._lock:
            return self._public_keys.get(agent_name)

    def verify(self, agent_name: str, message: bytes, signature: bytes) -> bool:
        """
        Verify a message signature from an agent.

        Args:
            agent_name: Name of the signing agent
            message: The original message bytes
            signature: The signature to verify

        Returns:
            True if signature is valid
        """
        with self._lock:
            identity = self._identities.get(agent_name)
            if not identity:
                logger.warning(f"Cannot verify: agent '{agent_name}' not registered")
                return False

        if _HAS_CRYPTO and identity._private_key is not None:
            try:
                public_key = identity._private_key.public_key()
                public_key.verify(signature, message)
                return True
            except InvalidSignature:
                logger.warning(f"Invalid signature from agent '{agent_name}'")
                return False
            except Exception as e:
                logger.error(f"Signature verification error: {e}")
                return False
        else:
            import hmac

            expected = hmac.new(identity._hmac_secret, message, hashlib.sha256).digest()
            return hmac.compare_digest(signature, expected)

    def verify_message_payload(
        self,
        agent_name: str,
        request_id: str,
        source: str,
        content_hash: str,
        signature: bytes,
    ) -> bool:
        """Verify a signed message payload."""
        canonical = f"{request_id}:{source}:{content_hash}".encode()
        return self.verify(agent_name, canonical, signature)

    def is_registered(self, agent_name: str) -> bool:
        """Check if an agent is registered."""
        with self._lock:
            return agent_name in self._identities

    def list_agents(self) -> list:
        """List all registered agent names."""
        with self._lock:
            return list(self._identities.keys())

    def unregister(self, agent_name: str) -> bool:
        """Remove an agent identity."""
        with self._lock:
            if agent_name in self._identities:
                del self._identities[agent_name]
                del self._public_keys[agent_name]
                logger.info(f"Unregistered agent identity: {agent_name}")
                return True
            return False


# Global registry instance
_global_identity_registry: Optional[AgentIdentityRegistry] = None


def get_identity_registry() -> AgentIdentityRegistry:
    """Get the global agent identity registry."""
    global _global_identity_registry
    if _global_identity_registry is None:
        _global_identity_registry = AgentIdentityRegistry()
    return _global_identity_registry


def set_identity_registry(registry: AgentIdentityRegistry) -> None:
    """Set the global agent identity registry (for testing)."""
    global _global_identity_registry
    _global_identity_registry = registry
