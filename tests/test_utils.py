"""
Tests for src/utils/ modules.

Covers:
- EncryptionService (AES-256-GCM encryption/decryption)
- CredentialManager (secure credential storage)
- SensitiveDataRedactor (data redaction patterns)
"""

import base64
import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch

# Skip all tests if cryptography is not available
pytest.importorskip("cryptography")


class TestEncryptionConfig:
    """Tests for EncryptionConfig dataclass."""

    def test_default_values(self):
        """Test default encryption configuration."""
        from src.utils.encryption import EncryptionConfig

        config = EncryptionConfig()
        assert config.algorithm == "AES-256-GCM"
        assert config.key_length == 32
        assert config.iv_length == 12
        assert config.salt_length == 32
        assert config.iterations == 600000
        assert config.tag_length == 16

    def test_custom_values(self):
        """Test custom encryption configuration."""
        from src.utils.encryption import EncryptionConfig

        config = EncryptionConfig(iterations=1000000, salt_length=64)
        assert config.iterations == 1000000
        assert config.salt_length == 64


class TestEncryptionService:
    """Tests for EncryptionService class."""

    @pytest.fixture
    def encryption_service(self):
        """Create an encryption service with a known key."""
        from src.utils.encryption import EncryptionService

        key = base64.b64decode("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
        return EncryptionService(master_key=key)

    def test_encrypt_decrypt_string(self, encryption_service):
        """Test basic string encryption and decryption."""
        plaintext = "Hello, Agent OS!"
        encrypted = encryption_service.encrypt(plaintext)

        assert encrypted.startswith("enc:")
        assert plaintext not in encrypted

        decrypted = encryption_service.decrypt(encrypted)
        assert decrypted == plaintext

    def test_encrypt_decrypt_bytes(self, encryption_service):
        """Test byte data encryption and decryption."""
        plaintext = b"Binary data \x00\x01\x02"
        encrypted = encryption_service.encrypt(plaintext)

        assert encrypted.startswith("enc:")

        decrypted = encryption_service.decrypt(encrypted)
        assert decrypted == plaintext.decode("utf-8")

    def test_encrypt_different_outputs(self, encryption_service):
        """Test that encryption produces different outputs for same input."""
        plaintext = "Same message"
        encrypted1 = encryption_service.encrypt(plaintext)
        encrypted2 = encryption_service.encrypt(plaintext)

        assert encrypted1 != encrypted2  # Different IVs

    def test_decrypt_invalid_format(self, encryption_service):
        """Test decryption of invalid format."""
        with pytest.raises(ValueError):
            encryption_service.decrypt("enc:invalid:format")

    def test_decrypt_legacy_format_rejected(self, encryption_service):
        """Test that legacy 'obs:' format is rejected."""
        with pytest.raises(ValueError, match="deprecated insecure"):
            encryption_service.decrypt("obs:somedata")

    def test_decrypt_plaintext_passthrough(self, encryption_service):
        """Test that non-encrypted strings pass through."""
        plaintext = "Not encrypted"
        result = encryption_service.decrypt(plaintext)
        assert result == plaintext

    def test_derive_key(self, encryption_service):
        """Test key derivation from password."""
        password = "my-secure-password"
        key1, salt1 = encryption_service.derive_key(password)
        key2, salt2 = encryption_service.derive_key(password)

        assert len(key1) == 32
        assert len(salt1) == 32
        assert key1 != key2  # Different salts
        assert salt1 != salt2

        # Same salt produces same key
        key3, _ = encryption_service.derive_key(password, salt=salt1)
        assert key1 == key3

    def test_encrypt_with_associated_data(self, encryption_service):
        """Test encryption with additional authenticated data."""
        plaintext = "Secret message"
        aad = b"additional context"

        encrypted = encryption_service.encrypt(plaintext, associated_data=aad)
        decrypted = encryption_service.decrypt(encrypted, associated_data=aad)
        assert decrypted == plaintext

        # Wrong AAD should fail
        with pytest.raises(ValueError):
            encryption_service.decrypt(encrypted, associated_data=b"wrong")


class TestCredentialManager:
    """Tests for CredentialManager class."""

    @pytest.fixture
    def credential_manager(self, tmp_path):
        """Create a credential manager with temporary storage."""
        from src.utils.encryption import CredentialManager, EncryptionService

        key = base64.b64decode("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
        encryption = EncryptionService(master_key=key)
        manager = CredentialManager(encryption_service=encryption)
        manager._storage_path = str(tmp_path / "credentials.enc")
        return manager

    def test_store_and_retrieve(self, credential_manager):
        """Test storing and retrieving credentials."""
        credential_manager.store("api_key", "secret123")
        retrieved = credential_manager.retrieve("api_key")
        assert retrieved == "secret123"

    def test_retrieve_nonexistent(self, credential_manager):
        """Test retrieving a non-existent credential."""
        result = credential_manager.retrieve("nonexistent")
        assert result is None

    def test_delete_credential(self, credential_manager):
        """Test deleting a credential."""
        credential_manager.store("temp_key", "value")
        assert credential_manager.retrieve("temp_key") == "value"

        credential_manager.delete("temp_key")
        assert credential_manager.retrieve("temp_key") is None

    def test_list_keys(self, credential_manager):
        """Test listing all credential keys."""
        credential_manager.store("key1", "value1")
        credential_manager.store("key2", "value2")

        keys = credential_manager.list_keys()
        assert "key1" in keys
        assert "key2" in keys

    def test_persistence(self, credential_manager):
        """Test credential persistence to disk."""
        from src.utils.encryption import CredentialManager, EncryptionService

        credential_manager.store("persistent_key", "persistent_value")

        # Create new manager with same storage path
        key = base64.b64decode("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
        new_manager = CredentialManager(
            encryption_service=EncryptionService(master_key=key)
        )
        new_manager._storage_path = credential_manager._storage_path
        new_manager.load()

        assert new_manager.retrieve("persistent_key") == "persistent_value"


class TestSensitiveDataRedactor:
    """Tests for SensitiveDataRedactor class."""

    @pytest.fixture
    def redactor(self):
        """Create a redactor instance."""
        from src.utils.encryption import SensitiveDataRedactor

        return SensitiveDataRedactor()

    def test_redact_api_key(self, redactor):
        """Test API key redaction."""
        text = 'api_key="sk-1234567890abcdefghijklmnop"'
        result = redactor.redact(text)
        assert "sk-1234567890" not in result
        assert "REDACTED" in result

    def test_redact_openai_key(self, redactor):
        """Test OpenAI API key redaction."""
        text = "Using key: sk-proj-abcdefghijklmnopqrstuvwxyz123456"
        result = redactor.redact(text)
        assert "sk-proj" not in result
        assert "REDACTED_OPENAI_KEY" in result

    def test_redact_github_token(self, redactor):
        """Test GitHub token redaction."""
        text = "token = ghp_abcdefghijklmnopqrstuvwxyz12345"
        result = redactor.redact(text)
        assert "ghp_" not in result
        assert "REDACTED_GITHUB_TOKEN" in result

    def test_redact_bearer_token(self, redactor):
        """Test Bearer token redaction."""
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        result = redactor.redact(text)
        assert "eyJh" not in result
        assert "Bearer [REDACTED]" in result

    def test_redact_password_field(self, redactor):
        """Test password field redaction."""
        text = 'password="mysecretpassword123"'
        result = redactor.redact(text)
        assert "mysecretpassword123" not in result
        assert "REDACTED" in result

    def test_redact_connection_string(self, redactor):
        """Test database connection string redaction."""
        text = "postgresql://user:password123@localhost:5432/db"
        result = redactor.redact(text)
        assert "password123" not in result
        assert "REDACTED" in result

    def test_redact_jwt(self, redactor):
        """Test JWT token redaction."""
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        result = redactor.redact(jwt)
        assert "eyJhbG" not in result
        assert "REDACTED_JWT" in result

    def test_redact_aws_key(self, redactor):
        """Test AWS key redaction."""
        text = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
        result = redactor.redact(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert "REDACTED_AWS_KEY" in result

    def test_redact_credit_card(self, redactor):
        """Test credit card number redaction."""
        text = "Card number: 4111-1111-1111-1111"
        result = redactor.redact(text)
        assert "4111-1111" not in result
        assert "REDACTED_CARD" in result

    def test_redact_ssn(self, redactor):
        """Test SSN redaction."""
        text = "SSN: 123-45-6789"
        result = redactor.redact(text)
        assert "123-45-6789" not in result
        assert "REDACTED_SSN" in result

    def test_redact_dict(self, redactor):
        """Test dictionary redaction."""
        data = {
            "username": "john",
            "password": "secret123",
            "api_key": "abc123",
            "nested": {"token": "xyz789"},
        }
        result = redactor.redact_dict(data)

        assert result["username"] == "john"
        assert result["password"] == "[REDACTED]"
        assert result["api_key"] == "[REDACTED]"
        assert result["nested"]["token"] == "[REDACTED]"

    def test_redact_dict_depth_limit(self, redactor):
        """Test dictionary redaction depth limit."""
        # Create deeply nested dict
        data = {"level": 0}
        current = data
        for i in range(15):
            current["nested"] = {"level": i + 1}
            current = current["nested"]

        # Should not raise even with deep nesting
        result = redactor.redact_dict(data)
        assert result is not None

    def test_redact_url(self, redactor):
        """Test URL redaction."""
        url = "https://user:password@example.com/path?token=secret123"
        result = redactor.redact_url(url)

        assert "password" not in result
        assert "secret123" not in result
        assert "REDACTED" in result


class TestFactoryFunctions:
    """Tests for module-level factory functions."""

    def test_get_encryption_service(self):
        """Test global encryption service getter."""
        from src.utils.encryption import get_encryption_service

        service1 = get_encryption_service()
        service2 = get_encryption_service()
        assert service1 is service2  # Same instance

    def test_get_redactor(self):
        """Test global redactor getter."""
        from src.utils.encryption import get_redactor

        redactor1 = get_redactor()
        redactor2 = get_redactor()
        assert redactor1 is redactor2  # Same instance

    def test_encrypt_decrypt_shortcuts(self):
        """Test encrypt/decrypt shortcut functions."""
        from src.utils.encryption import encrypt, decrypt

        plaintext = "Test message"
        encrypted = encrypt(plaintext)
        decrypted = decrypt(encrypted)
        assert decrypted == plaintext

    def test_redact_shortcut(self):
        """Test redact shortcut function."""
        from src.utils.encryption import redact

        text = "api_key=sk-abcdefghijklmnopqrstuvwxyz123"
        result = redact(text)
        assert "REDACTED" in result

    def test_redact_dict_shortcut(self):
        """Test redact_dict shortcut function."""
        from src.utils.encryption import redact_dict

        data = {"password": "secret"}
        result = redact_dict(data)
        assert result["password"] == "[REDACTED]"


class TestMasterKeyManagement:
    """Tests for master key loading and storage."""

    def test_master_key_from_env(self, tmp_path, monkeypatch):
        """Test loading master key from environment variable."""
        from src.utils.encryption import EncryptionService

        key = base64.b64encode(os.urandom(32)).decode()
        monkeypatch.setenv("AGENT_OS_ENCRYPTION_KEY", key)

        # Ensure no keyring or file interference
        with patch.object(EncryptionService, "_try_keyring_load", return_value=None):
            with patch.object(EncryptionService, "_try_file_load", return_value=None):
                service = EncryptionService()
                assert service._master_key == base64.b64decode(key)

    def test_encrypted_key_storage_format(self):
        """Test that stored keys use encrypted format."""
        from src.utils.encryption import EncryptionService

        key = os.urandom(32)
        service = EncryptionService(master_key=key)

        encrypted = service._encrypt_stored_key(key)
        assert encrypted.startswith(b"AOSKEY1:")

        decrypted = service._decrypt_stored_key(encrypted)
        assert decrypted == key


# =============================================================================
# Parameterized Tests for Better Coverage
# =============================================================================


class TestSensitiveDataRedactorParameterized:
    """Parameterized tests for SensitiveDataRedactor."""

    @pytest.fixture
    def redactor(self):
        """Create a redactor instance."""
        from src.utils.encryption import SensitiveDataRedactor

        return SensitiveDataRedactor()

    @pytest.mark.parametrize(
        "input_text,should_not_contain,should_contain",
        [
            # API Keys
            ('api_key="sk-1234567890abcdefghijklmnop"', "sk-1234567890", "REDACTED"),
            ("Using key: sk-proj-abcdefghijklmnopqrstuvwxyz123456", "sk-proj", "REDACTED_OPENAI_KEY"),
            # GitHub tokens
            ("token = ghp_abcdefghijklmnopqrstuvwxyz12345", "ghp_", "REDACTED_GITHUB_TOKEN"),
            ("gho_1234567890abcdefghij", "gho_", "REDACTED_GITHUB_TOKEN"),
            ("ghs_abcdefghij1234567890", "ghs_", "REDACTED_GITHUB_TOKEN"),
            # Bearer tokens
            ("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", "eyJh", "Bearer [REDACTED]"),
            # Passwords
            ('password="mysecretpassword123"', "mysecretpassword123", "REDACTED"),
            ("passwd=supersecret", "supersecret", "REDACTED"),
            ("pwd:hidden_value", "hidden_value", "REDACTED"),
            # Connection strings
            ("postgresql://user:password123@localhost:5432/db", "password123", "REDACTED"),
            ("mysql://root:secret@127.0.0.1/mydb", "secret", "REDACTED"),
            # AWS keys
            ("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE", "REDACTED_AWS_KEY"),
            ("AKIA1234567890ABCDEF", "AKIA1234567890ABCDEF", "REDACTED_AWS_KEY"),
            # Credit cards (various formats)
            ("Card number: 4111-1111-1111-1111", "4111-1111", "REDACTED_CARD"),
            ("cc: 4111111111111111", "4111111111111111", "REDACTED_CARD"),
            ("visa: 4012 8888 8888 1881", "4012", "REDACTED_CARD"),
            # SSN
            ("SSN: 123-45-6789", "123-45-6789", "REDACTED_SSN"),
            ("social: 987-65-4321", "987-65-4321", "REDACTED_SSN"),
            # JWT tokens
            (
                "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
                "eyJhbG",
                "REDACTED_JWT",
            ),
            # Slack tokens (using obviously fake test patterns)
            ("xoxb-FAKE-TEST-TOKEN-abcdefghijk", "xoxb-", "REDACTED_SLACK_TOKEN"),
            ("xoxp-FAKE-TEST-TOKEN-abcdefghijk", "xoxp-", "REDACTED_SLACK_TOKEN"),
            # Private keys (partial match)
            ("-----BEGIN RSA PRIVATE KEY-----", "PRIVATE KEY", "REDACTED_PRIVATE_KEY"),
        ],
        ids=[
            "api_key_sk",
            "openai_key",
            "github_pat",
            "github_oauth",
            "github_server",
            "bearer_token",
            "password_field",
            "passwd_field",
            "pwd_field",
            "postgres_conn",
            "mysql_conn",
            "aws_key_env",
            "aws_key_inline",
            "card_dashes",
            "card_plain",
            "card_spaces",
            "ssn_standard",
            "ssn_alt",
            "jwt_full",
            "slack_bot",
            "slack_user",
            "private_key",
        ],
    )
    def test_redaction_patterns(self, redactor, input_text, should_not_contain, should_contain):
        """Test various sensitive data patterns are properly redacted."""
        result = redactor.redact(input_text)
        assert should_not_contain not in result, f"'{should_not_contain}' should be redacted"
        assert should_contain in result, f"'{should_contain}' should be in result"

    @pytest.mark.parametrize(
        "sensitive_key,sensitive_value",
        [
            ("password", "secret123"),
            ("api_key", "abc123xyz"),
            ("token", "tok_12345"),
            ("secret", "mysecret"),
            ("auth_token", "auth_abc"),
            ("access_key", "ak_xyz"),
            ("private_key", "priv_123"),
            ("credentials", "cred_val"),
        ],
        ids=[
            "password_key",
            "api_key_key",
            "token_key",
            "secret_key",
            "auth_token_key",
            "access_key_key",
            "private_key_key",
            "credentials_key",
        ],
    )
    def test_dict_key_redaction(self, redactor, sensitive_key, sensitive_value):
        """Test that sensitive dictionary keys are properly redacted."""
        data = {"username": "john", sensitive_key: sensitive_value}
        result = redactor.redact_dict(data)
        assert result["username"] == "john"
        assert result[sensitive_key] == "[REDACTED]"


class TestEncryptionParameterized:
    """Parameterized tests for EncryptionService."""

    @pytest.fixture
    def encryption_service(self):
        """Create an encryption service with a known key."""
        from src.utils.encryption import EncryptionService

        key = base64.b64decode("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
        return EncryptionService(master_key=key)

    @pytest.mark.parametrize(
        "plaintext",
        [
            "Hello, World!",
            "",  # Empty string
            "a",  # Single character
            "A" * 10000,  # Large string
            "Unicode: こんにちは 🎉",
            "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?",
            "\n\r\t",  # Whitespace only
            "Line 1\nLine 2\nLine 3",  # Multiline
        ],
        ids=[
            "simple",
            "empty",
            "single_char",
            "large_string",
            "unicode",
            "special_chars",
            "whitespace",
            "multiline",
        ],
    )
    def test_encrypt_decrypt_roundtrip(self, encryption_service, plaintext):
        """Test encryption/decryption roundtrip for various inputs."""
        encrypted = encryption_service.encrypt(plaintext)
        decrypted = encryption_service.decrypt(encrypted)
        assert decrypted == plaintext

    @pytest.mark.parametrize(
        "password,expected_key_length",
        [
            ("short", 32),
            ("a" * 100, 32),
            ("password123", 32),
            ("ün1cödé_p@$$wörd", 32),
        ],
        ids=["short_password", "long_password", "alphanumeric", "unicode_password"],
    )
    def test_key_derivation_length(self, encryption_service, password, expected_key_length):
        """Test that derived keys have correct length regardless of password."""
        key, salt = encryption_service.derive_key(password)
        assert len(key) == expected_key_length
