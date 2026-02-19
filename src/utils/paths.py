"""
Configurable paths for Agent OS.

All file-system locations that were previously hardcoded to ~/.agent-os/
are now routed through get_config_dir(), which respects:

  1. AGENT_OS_CONFIG_DIR  (explicit override)
  2. XDG_CONFIG_HOME      (XDG fallback)
  3. ~/.config/agent-os   (default)

Phase 6.1 of the Agentic Security Audit remediation.
"""

import os
from pathlib import Path


def get_config_dir() -> Path:
    """Return the Agent-OS config directory, configurable via env var."""
    config_dir = os.environ.get("AGENT_OS_CONFIG_DIR")
    if config_dir:
        return Path(config_dir)
    xdg_config = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    return Path(xdg_config) / "agent-os"


def get_encryption_key_path() -> str:
    """Return the path to the encryption key file."""
    return str(get_config_dir() / "encryption.key")


def get_credentials_path() -> str:
    """Return the path to the encrypted credentials file."""
    return str(get_config_dir() / "credentials.enc")


def get_machine_salt_path() -> Path:
    """Return the path to the machine salt file."""
    return get_config_dir() / ".machine_salt"
