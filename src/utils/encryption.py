"""
Encryption Utilities

Provides encryption utilities for securing sensitive data:
- Symmetric encryption (AES-256-GCM)
- Secure key derivation (PBKDF2)
- Credential encryption/decryption
- Sensitive data redaction
"""

import base64
import hashlib
import hmac
import logging
import os
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Pattern, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EncryptionConfig:
    """Encryption configuration."""

    algorithm: str = "AES-256-GCM"
    key_length: int = 32  # 256 bits
    iv_length: int = 12  # 96 bits for GCM
    salt_length: int = 32  # Standardized to 32 bytes
    iterations: int = 600000  # NIST SP 800-132 recommends >= 600,000 for SHA-256
    tag_length: int = 16


# =============================================================================
# Encryption Service
# =============================================================================


class EncryptionService:
    """
    Service for encrypting and decrypting sensitive data.

    Uses AES-256-GCM for authenticated encryption.
    Requires the 'cryptography' library - no insecure fallbacks.
    """

    def __init__(
        self, master_key: Optional[bytes] = None, config: Optional[EncryptionConfig] = None
    ):
        self.config = config or EncryptionConfig()
        self._check_crypto()  # Will raise if not available

        # Master key for encryption
        if master_key:
            self._master_key = master_key
        else:
            # Try to load from environment or generate
            self._master_key = self._get_or_create_master_key()

    def _check_crypto(self) -> None:
        """Verify cryptography library is available. Raises if not."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            raise RuntimeError(
                "The 'cryptography' library is required for encryption. "
                "Install it with: pip install cryptography"
            )

    def _get_or_create_master_key(self) -> bytes:
        """
        Get or create master encryption key.

        Priority order:
        1. Environment variable AGENT_OS_ENCRYPTION_KEY
        2. OS keyring/credential store (if keyring library available)
        3. Encrypted file storage (with machine-specific protection)
        """
        # Try environment variable first (highest priority)
        env_key = os.environ.get("AGENT_OS_ENCRYPTION_KEY")
        if env_key:
            return base64.b64decode(env_key)

        # Try OS keyring if available
        key = self._try_keyring_load()
        if key:
            return key

        # Try to load from encrypted file
        key = self._try_file_load()
        if key:
            return key

        # Generate new key and store it
        key = secrets.token_bytes(self.config.key_length)
        self._store_master_key(key)
        return key

    def _try_keyring_load(self) -> Optional[bytes]:
        """Try to load master key from OS keyring."""
        try:
            import keyring

            key_b64 = keyring.get_password("agent-os", "master-key")
            if key_b64:
                logger.debug("Loaded master key from OS keyring")
                return base64.b64decode(key_b64)
        except ImportError:
            pass  # keyring not installed
        except Exception as e:
            logger.debug(f"Could not access keyring: {e}")
        return None

    def _try_file_load(self) -> Optional[bytes]:
        """Try to load master key from encrypted file."""
        key_file = os.path.expanduser("~/.agent-os/encryption.key")
        if not os.path.exists(key_file):
            return None

        try:
            with open(key_file, "rb") as f:
                stored_data = f.read()

            # Check if file uses new encrypted format (starts with version marker)
            if stored_data.startswith(b"AOSKEY1:"):
                return self._decrypt_stored_key(stored_data)
            else:
                # Legacy unencrypted format - migrate on next save
                logger.warning(
                    "Master key stored in legacy unencrypted format. "
                    "Key will be migrated to encrypted format on next save."
                )
                return stored_data
        except Exception as e:
            logger.error(f"Failed to load master key from file: {e}")
            return None

    def _get_machine_key(self) -> bytes:
        """
        Derive a machine-specific key for protecting the master key at rest.

        Uses machine-specific identifiers to derive a key that is unique to
        this machine, making the stored key file less useful if copied.
        """
        import getpass
        import platform

        # Combine machine-specific identifiers
        machine_id_parts = [
            platform.node(),  # hostname
            getpass.getuser(),  # username
            os.path.expanduser("~"),  # home directory
            platform.machine(),  # architecture
        ]
        machine_id = ":".join(machine_id_parts).encode()

        # Derive key using PBKDF2
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        # Fixed salt for machine key derivation (not a security concern as
        # the machine-specific data provides the entropy)
        salt = b"agent-os-machine-key-v1"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,  # Faster since machine ID has good entropy
        )
        return kdf.derive(machine_id)

    def _encrypt_stored_key(self, key: bytes) -> bytes:
        """Encrypt the master key for file storage."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        machine_key = self._get_machine_key()
        nonce = secrets.token_bytes(12)
        aesgcm = AESGCM(machine_key)
        ciphertext = aesgcm.encrypt(nonce, key, b"agent-os-master-key")

        # Format: VERSION:NONCE:CIPHERTEXT
        return b"AOSKEY1:" + nonce + ciphertext

    def _decrypt_stored_key(self, stored_data: bytes) -> bytes:
        """Decrypt the master key from file storage."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        # Parse format: VERSION:NONCE:CIPHERTEXT
        if not stored_data.startswith(b"AOSKEY1:"):
            raise ValueError("Invalid stored key format")

        data = stored_data[8:]  # Skip "AOSKEY1:"
        nonce = data[:12]
        ciphertext = data[12:]

        machine_key = self._get_machine_key()
        aesgcm = AESGCM(machine_key)
        return aesgcm.decrypt(nonce, ciphertext, b"agent-os-master-key")

    def _store_master_key(self, key: bytes) -> None:
        """Store the master key securely."""
        # Try OS keyring first
        try:
            import keyring

            keyring.set_password("agent-os", "master-key", base64.b64encode(key).decode())
            logger.info("Stored master key in OS keyring")
            return
        except ImportError:
            logger.info(
                "keyring library not installed. Install with 'pip install keyring' "
                "for more secure key storage."
            )
        except Exception as e:
            logger.warning(f"Could not store key in keyring: {e}")

        # Fall back to encrypted file storage
        key_file = os.path.expanduser("~/.agent-os/encryption.key")
        try:
            key_dir = os.path.dirname(key_file)
            os.makedirs(key_dir, mode=0o700, exist_ok=True)

            encrypted_key = self._encrypt_stored_key(key)
            with open(key_file, "wb") as f:
                f.write(encrypted_key)
            os.chmod(key_file, 0o600)

            logger.info(f"Created encrypted master key at {key_file}")
            logger.warning(
                "Master key stored in encrypted file. For better security, "
                "install 'keyring' library or set AGENT_OS_ENCRYPTION_KEY environment variable."
            )
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not persist encryption key: {e}")

    def derive_key(self, password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """
        Derive encryption key from password using PBKDF2.

        Returns:
            Tuple of (derived_key, salt)
        """
        if salt is None:
            salt = secrets.token_bytes(self.config.salt_length)

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.config.key_length,
            salt=salt,
            iterations=self.config.iterations,
        )
        key = kdf.derive(password.encode())

        return key, salt

    def encrypt(
        self,
        plaintext: Union[str, bytes],
        key: Optional[bytes] = None,
        associated_data: Optional[bytes] = None,
    ) -> str:
        """
        Encrypt data and return as base64 string.

        Args:
            plaintext: Data to encrypt
            key: Encryption key (uses master key if not provided)
            associated_data: Additional authenticated data

        Returns:
            Base64-encoded encrypted data with format: "enc:<iv>:<ciphertext>:<tag>"
        """
        if isinstance(plaintext, str):
            plaintext = plaintext.encode("utf-8")

        key = key or self._master_key

        return self._encrypt_aes_gcm(plaintext, key, associated_data)

    def decrypt(
        self,
        ciphertext: str,
        key: Optional[bytes] = None,
        associated_data: Optional[bytes] = None,
    ) -> str:
        """
        Decrypt data.

        Args:
            ciphertext: Base64-encoded encrypted data
            key: Decryption key (uses master key if not provided)
            associated_data: Additional authenticated data

        Returns:
            Decrypted string

        Raises:
            ValueError: If ciphertext format is invalid or uses deprecated insecure format
        """
        # Check if already decrypted (backward compatibility)
        if not ciphertext.startswith("enc:") and not ciphertext.startswith("obs:"):
            return ciphertext

        key = key or self._master_key

        if ciphertext.startswith("enc:"):
            return self._decrypt_aes_gcm(ciphertext, key, associated_data)
        else:
            # Reject legacy insecure format
            raise ValueError(
                "Ciphertext uses deprecated insecure 'obs:' format. "
                "Data must be re-encrypted with secure AES-GCM encryption."
            )

    def _encrypt_aes_gcm(
        self,
        plaintext: bytes,
        key: bytes,
        associated_data: Optional[bytes],
    ) -> str:
        """Encrypt using AES-GCM."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        iv = secrets.token_bytes(self.config.iv_length)
        aesgcm = AESGCM(key[: self.config.key_length])

        ciphertext_with_tag = aesgcm.encrypt(iv, plaintext, associated_data)

        # Split ciphertext and tag
        ciphertext = ciphertext_with_tag[: -self.config.tag_length]
        tag = ciphertext_with_tag[-self.config.tag_length :]

        # Encode as base64
        iv_b64 = base64.b64encode(iv).decode()
        ct_b64 = base64.b64encode(ciphertext).decode()
        tag_b64 = base64.b64encode(tag).decode()

        return f"enc:{iv_b64}:{ct_b64}:{tag_b64}"

    def _decrypt_aes_gcm(
        self,
        encrypted: str,
        key: bytes,
        associated_data: Optional[bytes],
    ) -> str:
        """Decrypt using AES-GCM."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        try:
            parts = encrypted.split(":")
            if len(parts) != 4:
                raise ValueError("Invalid encrypted format")

            iv = base64.b64decode(parts[1])
            ciphertext = base64.b64decode(parts[2])
            tag = base64.b64decode(parts[3])

            aesgcm = AESGCM(key[: self.config.key_length])
            plaintext = aesgcm.decrypt(iv, ciphertext + tag, associated_data)

            return plaintext.decode("utf-8")

        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Decryption failed") from e


# =============================================================================
# Credential Manager
# =============================================================================


class CredentialManager:
    """
    Manages encrypted credentials storage.
    """

    def __init__(self, encryption_service: Optional[EncryptionService] = None):
        self.encryption = encryption_service or EncryptionService()
        self._credentials: Dict[str, str] = {}
        self._storage_path = os.path.expanduser("~/.agent-os/credentials.enc")

    def store(self, key: str, value: str) -> None:
        """Store an encrypted credential."""
        encrypted = self.encryption.encrypt(value)
        self._credentials[key] = encrypted
        self._persist()

    def retrieve(self, key: str) -> Optional[str]:
        """Retrieve and decrypt a credential."""
        encrypted = self._credentials.get(key)
        if encrypted:
            try:
                return self.encryption.decrypt(encrypted)
            except ValueError:
                logger.error(f"Failed to decrypt credential: {key}")
                return None
        return None

    def delete(self, key: str) -> None:
        """Delete a credential."""
        if key in self._credentials:
            del self._credentials[key]
            self._persist()

    def list_keys(self) -> List[str]:
        """List all credential keys."""
        return list(self._credentials.keys())

    def _persist(self) -> None:
        """Persist credentials to disk."""
        try:
            os.makedirs(os.path.dirname(self._storage_path), mode=0o700, exist_ok=True)

            # Serialize and encrypt the whole credentials dict
            import json

            data = json.dumps(self._credentials)
            encrypted = self.encryption.encrypt(data)

            with open(self._storage_path, "w") as f:
                f.write(encrypted)

            os.chmod(self._storage_path, 0o600)

        except Exception as e:
            logger.error(f"Failed to persist credentials: {e}")

    def load(self) -> None:
        """Load credentials from disk."""
        if not os.path.exists(self._storage_path):
            return

        try:
            with open(self._storage_path, "r") as f:
                encrypted = f.read()

            import json

            data = self.encryption.decrypt(encrypted)
            self._credentials = json.loads(data)

        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            self._credentials = {}


# =============================================================================
# Sensitive Data Redactor
# =============================================================================


@dataclass
class RedactionPattern:
    """Pattern for redacting sensitive data."""

    pattern: Pattern
    replacement: str
    description: str = ""


class SensitiveDataRedactor:
    """
    Redacts sensitive information from logs and output.
    """

    def __init__(self):
        self.patterns: List[RedactionPattern] = [
            # API Keys
            RedactionPattern(
                re.compile(
                    r'([a-zA-Z0-9_-]*(?:api[_-]?key|apikey)[a-zA-Z0-9_-]*)[=:]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
                    re.I,
                ),
                r"\1=[REDACTED]",
                "API key assignments",
            ),
            RedactionPattern(
                re.compile(r"sk-[a-zA-Z0-9]{20,}"), "[REDACTED_OPENAI_KEY]", "OpenAI API keys"
            ),
            RedactionPattern(
                re.compile(r"hf_[a-zA-Z0-9]{20,}"), "[REDACTED_HF_TOKEN]", "Hugging Face tokens"
            ),
            RedactionPattern(
                re.compile(r"ghp_[a-zA-Z0-9]{20,}"),
                "[REDACTED_GITHUB_TOKEN]",
                "GitHub personal access tokens",
            ),
            # Auth headers
            RedactionPattern(
                re.compile(r"Bearer\s+[a-zA-Z0-9_.-]{20,}", re.I),
                "Bearer [REDACTED]",
                "Bearer tokens",
            ),
            RedactionPattern(
                re.compile(r"Basic\s+[a-zA-Z0-9+/=]{20,}", re.I), "Basic [REDACTED]", "Basic auth"
            ),
            # Passwords
            RedactionPattern(
                re.compile(r'(password|passwd|pwd)[=:]\s*["\']?[^"\'\s&]+["\']?', re.I),
                r"\1=[REDACTED]",
                "Password fields",
            ),
            # Connection strings
            RedactionPattern(
                re.compile(r"(mongodb|postgresql|mysql|redis):\/\/[^:]+:[^@]+@", re.I),
                r"\1://[REDACTED]:[REDACTED]@",
                "Database connection strings",
            ),
            # JWT tokens
            RedactionPattern(
                re.compile(r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*"),
                "[REDACTED_JWT]",
                "JWT tokens",
            ),
            # Private keys
            RedactionPattern(
                re.compile(
                    r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----"
                ),
                "[REDACTED_PRIVATE_KEY]",
                "Private keys",
            ),
            # AWS keys
            RedactionPattern(
                re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED_AWS_KEY]", "AWS access keys"
            ),
            # Credit cards (basic)
            RedactionPattern(
                re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
                "[REDACTED_CARD]",
                "Credit card numbers",
            ),
            # SSN
            RedactionPattern(
                re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]", "Social Security numbers"
            ),
        ]

        # Sensitive field names
        self.sensitive_fields = {
            "password",
            "passwd",
            "pwd",
            "secret",
            "token",
            "apikey",
            "api_key",
            "apiKey",
            "auth",
            "authorization",
            "credential",
            "credentials",
            "private_key",
            "privateKey",
            "access_token",
            "accessToken",
            "refresh_token",
            "refreshToken",
            "session_id",
            "sessionId",
            "encryption_key",
            "encryptionKey",
            "master_key",
            "masterKey",
        }

    def redact(self, text: str) -> str:
        """Redact sensitive data from a string."""
        if not isinstance(text, str):
            return str(text)

        result = text
        for pattern in self.patterns:
            result = pattern.pattern.sub(pattern.replacement, result)

        return result

    def redact_dict(self, data: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
        """Redact sensitive data from a dictionary."""
        if depth > 10:  # Prevent infinite recursion
            return data

        result = {}

        for key, value in data.items():
            key_lower = key.lower()

            # Check if field name indicates sensitive data
            if key_lower in self.sensitive_fields or key in self.sensitive_fields:
                result[key] = "[REDACTED]"
            elif any(
                s in key_lower for s in ["password", "secret", "token", "key", "credential", "auth"]
            ):
                result[key] = "[REDACTED]"
            elif isinstance(value, str):
                result[key] = self.redact(value)
            elif isinstance(value, dict):
                result[key] = self.redact_dict(value, depth + 1)
            elif isinstance(value, list):
                result[key] = [
                    (
                        self.redact_dict(v, depth + 1)
                        if isinstance(v, dict)
                        else self.redact(v) if isinstance(v, str) else v
                    )
                    for v in value
                ]
            else:
                result[key] = value

        return result

    def redact_url(self, url: str) -> str:
        """Redact sensitive parts of a URL."""
        # Redact password in URL
        url = re.sub(r"(https?://[^:]+:)[^@]+(@)", r"\1[REDACTED]\2", url)

        # Redact sensitive query parameters
        sensitive_params = [
            "token",
            "key",
            "secret",
            "password",
            "auth",
            "credential",
            "api_key",
            "apikey",
        ]

        for param in sensitive_params:
            url = re.sub(rf"([?&]{param}=)[^&]+", rf"\1[REDACTED]", url, flags=re.I)

        return url


# =============================================================================
# Factory Functions
# =============================================================================


# Global instances
_encryption_service: Optional[EncryptionService] = None
_credential_manager: Optional[CredentialManager] = None
_redactor: Optional[SensitiveDataRedactor] = None


def get_encryption_service() -> EncryptionService:
    """Get the global encryption service instance."""
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = EncryptionService()
    return _encryption_service


def get_credential_manager() -> CredentialManager:
    """Get the global credential manager instance."""
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = CredentialManager(get_encryption_service())
        _credential_manager.load()
    return _credential_manager


def get_redactor() -> SensitiveDataRedactor:
    """Get the global redactor instance."""
    global _redactor
    if _redactor is None:
        _redactor = SensitiveDataRedactor()
    return _redactor


def encrypt(value: str) -> str:
    """Encrypt a string value."""
    return get_encryption_service().encrypt(value)


def decrypt(value: str) -> str:
    """Decrypt an encrypted string value."""
    return get_encryption_service().decrypt(value)


def redact(text: str) -> str:
    """Redact sensitive data from text."""
    return get_redactor().redact(text)


def redact_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Redact sensitive data from a dictionary."""
    return get_redactor().redact_dict(data)
