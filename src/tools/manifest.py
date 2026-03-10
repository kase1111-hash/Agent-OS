"""
Tool / Skill Manifest System.

Declares what resources a tool is permitted to access (network endpoints,
file paths, shell commands, external APIs). The manifest is optionally
signed with the agent's Ed25519 identity key so that it can be verified
at registration time.

Phase 6.4 of the Agentic Security Audit remediation.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolPermissions:
    """Declared permissions for a tool."""

    network_endpoints: List[str] = field(default_factory=list)
    file_paths: List[str] = field(default_factory=list)
    shell_commands: List[str] = field(default_factory=list)
    apis_called: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "network_endpoints": self.network_endpoints,
            "file_paths": self.file_paths,
            "shell_commands": self.shell_commands,
            "apis_called": self.apis_called,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolPermissions":
        return cls(
            network_endpoints=data.get("network_endpoints", []),
            file_paths=data.get("file_paths", []),
            shell_commands=data.get("shell_commands", []),
            apis_called=data.get("apis_called", []),
        )


@dataclass
class ToolManifest:
    """
    Signed manifest declaring a tool's identity and permissions.

    If signature is present it should be an Ed25519 signature over the
    canonical JSON of the manifest content (tool_name + version + author +
    permissions).
    """

    tool_name: str
    version: str
    author: str
    permissions: ToolPermissions = field(default_factory=ToolPermissions)
    signature: Optional[str] = None

    def canonical_content(self) -> str:
        """Return the canonical JSON of the manifest for signing/verification."""
        payload = {
            "tool_name": self.tool_name,
            "version": self.version,
            "author": self.author,
            "permissions": self.permissions.to_dict(),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def content_hash(self) -> str:
        """SHA-256 of the canonical content."""
        return hashlib.sha256(self.canonical_content().encode()).hexdigest()

    def verify_signature(self) -> bool:
        """
        Verify the manifest signature using the agent identity system.

        Rejects unsigned manifests by default (fail-closed).
        Set AGENT_OS_ALLOW_UNSIGNED_MANIFESTS=true for dev mode.
        """
        if not self.signature:
            import os

            allow_unsigned = os.getenv(
                "AGENT_OS_ALLOW_UNSIGNED_MANIFESTS", "false"
            ).lower() in ("1", "true", "yes")
            if allow_unsigned:
                logger.warning(
                    "Manifest for %s is unsigned — accepting (dev mode)",
                    self.tool_name,
                )
                return True
            logger.warning(
                "Manifest for %s is unsigned — REJECTED. "
                "Set AGENT_OS_ALLOW_UNSIGNED_MANIFESTS=true for dev mode.",
                self.tool_name,
            )
            return False

        try:
            from src.agents.identity import verify_message

            content = self.canonical_content().encode()
            sig_bytes = bytes.fromhex(self.signature)
            return verify_message(content, sig_bytes, self.author)
        except ImportError:
            logger.warning(
                "Agent identity module not available — rejecting manifest "
                "for %s (fail-closed).",
                self.tool_name,
            )
            return False
        except Exception as e:
            logger.error(
                "Manifest signature verification failed for %s: %s",
                self.tool_name,
                e,
            )
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "version": self.version,
            "author": self.author,
            "permissions": self.permissions.to_dict(),
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolManifest":
        return cls(
            tool_name=data["tool_name"],
            version=data["version"],
            author=data["author"],
            permissions=ToolPermissions.from_dict(data.get("permissions", {})),
            signature=data.get("signature"),
        )
