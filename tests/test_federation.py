"""Tests for federation module - inter-instance communication protocol."""

import pytest
import asyncio
import base64
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


class TestIdentityModule:
    """Tests for identity management."""

    def test_key_type_enum(self):
        """Test KeyType enum values."""
        from src.federation.identity import KeyType

        assert KeyType.ED25519.value == "ed25519"
        assert KeyType.RSA_2048.value == "rsa-2048"
        assert KeyType.RSA_4096.value == "rsa-4096"

    def test_identity_status_enum(self):
        """Test IdentityStatus enum values."""
        from src.federation.identity import IdentityStatus

        assert IdentityStatus.UNVERIFIED.value == "unverified"
        assert IdentityStatus.VERIFIED.value == "verified"
        assert IdentityStatus.TRUSTED.value == "trusted"

    def test_public_key_creation(self):
        """Test PublicKey dataclass creation."""
        from src.federation.identity import PublicKey, KeyType

        key_data = b"test_key_data_32_bytes_padding!!"
        pub_key = PublicKey(key_type=KeyType.ED25519, key_data=key_data)

        assert pub_key.key_type == KeyType.ED25519
        assert pub_key.key_data == key_data
        assert pub_key.key_id is not None

    def test_public_key_to_dict(self):
        """Test PublicKey serialization."""
        from src.federation.identity import PublicKey, KeyType

        key_data = b"test_key_data"
        pub_key = PublicKey(key_type=KeyType.ED25519, key_data=key_data)

        data = pub_key.to_dict()
        assert data["key_type"] == "ed25519"
        assert data["key_data"] == base64.b64encode(key_data).decode()

    def test_public_key_from_dict(self):
        """Test PublicKey deserialization."""
        from src.federation.identity import PublicKey, KeyType

        key_data = b"test_key_data"
        data = {
            "key_type": "ed25519",
            "key_data": base64.b64encode(key_data).decode(),
        }

        pub_key = PublicKey.from_dict(data)
        assert pub_key.key_type == KeyType.ED25519
        assert pub_key.key_data == key_data

    def test_private_key_creation(self):
        """Test PrivateKey dataclass creation."""
        from src.federation.identity import PrivateKey, KeyType

        key_data = b"private_key_data_64_bytes_padding_for_testing!!!!!"
        priv_key = PrivateKey(key_type=KeyType.ED25519, key_data=key_data)

        assert priv_key.key_type == KeyType.ED25519
        assert priv_key.key_data == key_data

    def test_key_pair_creation(self):
        """Test KeyPair dataclass creation."""
        from src.federation.identity import KeyPair, PublicKey, PrivateKey, KeyType

        pub = PublicKey(KeyType.ED25519, b"public_key_data_32bytes_pad!!!!!")
        priv = PrivateKey(KeyType.ED25519, b"private_key_data_32bytes_pad!!!!")
        pair = KeyPair(public_key=pub, private_key=priv)

        assert pair.public_key == pub
        assert pair.private_key == priv

    def test_key_pair_generate(self):
        """Test KeyPair.generate creates valid keys."""
        from src.federation.identity import KeyPair, KeyType

        pair = KeyPair.generate(KeyType.ED25519)

        assert pair.public_key is not None
        assert pair.private_key is not None
        assert pair.key_type == KeyType.ED25519

    def test_identity_creation(self):
        """Test Identity dataclass creation."""
        from src.federation.identity import Identity, PublicKey, KeyType, IdentityStatus

        pub_key = PublicKey(KeyType.ED25519, b"key_data_32bytes_padding!!!!!!!!")
        identity = Identity(
            node_id="test-node",
            display_name="Test Node",
            public_key=pub_key,
        )

        assert identity.node_id == "test-node"
        assert identity.display_name == "Test Node"
        assert identity.public_key == pub_key
        assert identity.status == IdentityStatus.UNVERIFIED
        assert identity.is_verified is False

    def test_identity_to_dict(self):
        """Test Identity serialization."""
        from src.federation.identity import Identity, PublicKey, KeyType

        pub_key = PublicKey(KeyType.ED25519, b"keydata1234567890123456")
        identity = Identity(
            node_id="test-node",
            display_name="Test Node",
            public_key=pub_key,
        )

        data = identity.to_dict()
        assert data["node_id"] == "test-node"
        assert data["display_name"] == "Test Node"
        assert "public_key" in data

    def test_identity_from_dict(self):
        """Test Identity deserialization."""
        from src.federation.identity import Identity, KeyType

        key_data = b"keydata12345678901234"
        data = {
            "node_id": "test-node",
            "display_name": "Test Node",
            "public_key": {
                "key_type": "ed25519",
                "key_data": base64.b64encode(key_data).decode(),
            },
        }

        identity = Identity.from_dict(data)
        assert identity.node_id == "test-node"
        assert identity.display_name == "Test Node"
        assert identity.public_key.key_type == KeyType.ED25519

    def test_certificate_creation(self):
        """Test Certificate dataclass creation."""
        from src.federation.identity import Certificate, PublicKey, KeyType

        pub_key = PublicKey(KeyType.ED25519, b"keydata_32bytes_padding_more!!!!")
        now = datetime.utcnow()
        cert = Certificate(
            issuer_id="issuer-node",
            subject_id="subject-node",
            public_key=pub_key,
            valid_from=now,
            valid_until=now + timedelta(days=365),
            signature=b"signature",
        )

        assert cert.issuer_id == "issuer-node"
        assert cert.subject_id == "subject-node"
        assert cert.public_key == pub_key

    def test_certificate_is_valid(self):
        """Test certificate validity check."""
        from src.federation.identity import Certificate, PublicKey, KeyType

        pub_key = PublicKey(KeyType.ED25519, b"keydata_32bytes_padding_more!!!!")

        # Expired certificate
        expired_cert = Certificate(
            issuer_id="issuer",
            subject_id="subject",
            public_key=pub_key,
            valid_from=datetime.utcnow() - timedelta(days=30),
            valid_until=datetime.utcnow() - timedelta(days=1),
            signature=b"sig",
        )
        assert expired_cert.is_valid is False

        # Valid certificate
        valid_cert = Certificate(
            issuer_id="issuer",
            subject_id="subject",
            public_key=pub_key,
            valid_from=datetime.utcnow() - timedelta(days=1),
            valid_until=datetime.utcnow() + timedelta(days=365),
            signature=b"sig",
        )
        assert valid_cert.is_valid is True

    def test_certificate_is_self_signed(self):
        """Test self-signed certificate check."""
        from src.federation.identity import Certificate, PublicKey, KeyType

        pub_key = PublicKey(KeyType.ED25519, b"keydata_32bytes_padding_more!!!!")
        now = datetime.utcnow()

        self_signed = Certificate(
            issuer_id="node",
            subject_id="node",
            public_key=pub_key,
            valid_from=now,
            valid_until=now + timedelta(days=365),
        )
        assert self_signed.is_self_signed is True

        not_self_signed = Certificate(
            issuer_id="issuer",
            subject_id="subject",
            public_key=pub_key,
            valid_from=now,
            valid_until=now + timedelta(days=365),
        )
        assert not_self_signed.is_self_signed is False

    def test_certificate_to_dict(self):
        """Test Certificate serialization."""
        from src.federation.identity import Certificate, PublicKey, KeyType

        pub_key = PublicKey(KeyType.ED25519, b"keydata_32bytes_padding_more!!!!")
        now = datetime.utcnow()
        cert = Certificate(
            issuer_id="issuer",
            subject_id="subject",
            public_key=pub_key,
            valid_from=now,
            valid_until=now + timedelta(days=365),
            signature=b"signature",
        )

        data = cert.to_dict()
        assert data["issuer_id"] == "issuer"
        assert data["subject_id"] == "subject"
        assert "public_key" in data

    def test_identity_manager_initialization(self):
        """Test IdentityManager initialization."""
        from src.federation.identity import IdentityManager

        manager = IdentityManager(node_id="test-node")

        assert manager.node_id == "test-node"
        assert manager.identity is not None
        assert manager.identity.node_id == "test-node"

    def test_identity_manager_sign_and_verify(self):
        """Test signing and verification."""
        from src.federation.identity import IdentityManager

        manager = IdentityManager(node_id="test-node")
        data = b"test message to sign"

        signature = manager.sign(data)
        assert signature is not None
        assert len(signature) > 0

        # Verify with own public key
        is_valid = manager.verify_signature(
            data, signature, manager.identity.public_key
        )
        assert is_valid is True

        # Invalid signature
        is_valid = manager.verify_signature(
            b"different data", signature, manager.identity.public_key
        )
        assert is_valid is False

    def test_identity_manager_register_and_get_identity(self):
        """Test identity registration and retrieval."""
        from src.federation.identity import IdentityManager, Identity, PublicKey, KeyType

        manager = IdentityManager(node_id="node-a")

        # Create a peer identity
        peer_key = PublicKey(KeyType.ED25519, b"peer_key_32bytes_padding!!!!!!!!")
        peer_identity = Identity(
            node_id="node-b",
            display_name="Peer Node",
            public_key=peer_key,
        )

        # Register the identity
        result = manager.register_identity(peer_identity)
        assert result is True

        # Retrieve it
        retrieved = manager.get_identity("node-b")
        assert retrieved is not None
        assert retrieved.node_id == "node-b"

    def test_identity_manager_trust_identity(self):
        """Test trusting an identity."""
        from src.federation.identity import IdentityManager

        manager = IdentityManager(node_id="test-node")

        # Create another manager for the peer
        peer_manager = IdentityManager(node_id="peer-node")

        # Register the peer's identity first
        manager.register_identity(peer_manager.identity)

        # Trust the node
        result = manager.trust_identity("peer-node")
        assert result is True

        # Check if trusted
        assert manager.is_trusted("peer-node") is True
        assert manager.is_trusted("unknown-node") is False

    def test_create_identity_function(self):
        """Test create_identity helper function."""
        from src.federation.identity import create_identity

        identity = create_identity(node_id="test-node", display_name="Test")

        assert identity.node_id == "test-node"
        assert identity.display_name == "Test"
        assert identity.public_key is not None


class TestProtocolModule:
    """Tests for federation protocol."""

    def test_message_type_enum(self):
        """Test MessageType enum values."""
        from src.federation.protocol import MessageType

        assert MessageType.HELLO.value == "hello"
        assert MessageType.HELLO_ACK.value == "hello_ack"
        assert MessageType.PING.value == "ping"
        assert MessageType.PONG.value == "pong"
        assert MessageType.PERMISSION_REQUEST.value == "permission_request"

    def test_message_priority_enum(self):
        """Test MessagePriority enum values."""
        from src.federation.protocol import MessagePriority

        assert MessagePriority.LOW.value == "low"
        assert MessagePriority.NORMAL.value == "normal"
        assert MessagePriority.HIGH.value == "high"

    def test_federation_message_creation(self):
        """Test FederationMessage.create factory method."""
        from src.federation.protocol import FederationMessage, MessageType

        msg = FederationMessage.create(
            message_type=MessageType.DATA_REQUEST,
            sender_id="source-node",
            recipient_id="target-node",
            payload={"action": "test"},
        )

        assert msg.message_type == MessageType.DATA_REQUEST
        assert msg.sender_id == "source-node"
        assert msg.recipient_id == "target-node"
        assert msg.payload["action"] == "test"
        assert msg.message_id is not None

    def test_federation_message_to_dict(self):
        """Test FederationMessage serialization."""
        from src.federation.protocol import FederationMessage, MessageType

        msg = FederationMessage.create(
            message_type=MessageType.PING,
            sender_id="source",
            recipient_id="target",
            payload={},
        )

        data = msg.to_dict()
        assert data["message_type"] == "ping"
        assert data["sender_id"] == "source"
        assert data["recipient_id"] == "target"
        assert "message_id" in data

    def test_federation_message_from_dict(self):
        """Test FederationMessage deserialization."""
        from src.federation.protocol import FederationMessage, MessageType

        data = {
            "message_type": "data_request",
            "sender_id": "source",
            "recipient_id": "target",
            "message_id": "msg-123",
            "payload": {"test": "value"},
            "timestamp": datetime.now().isoformat(),
        }

        msg = FederationMessage.from_dict(data)
        assert msg.message_type == MessageType.DATA_REQUEST
        assert msg.sender_id == "source"
        assert msg.message_id == "msg-123"
        assert msg.payload["test"] == "value"

    def test_federation_message_create_response(self):
        """Test FederationMessage.create_response."""
        from src.federation.protocol import FederationMessage, MessageType

        request = FederationMessage.create(
            message_type=MessageType.DATA_REQUEST,
            sender_id="requester",
            recipient_id="responder",
            payload={"request": "data"},
        )

        response = FederationMessage.create_response(
            request,
            MessageType.DATA_RESPONSE,
            payload={"data": "result"},
        )

        assert response.message_type == MessageType.DATA_RESPONSE
        assert response.sender_id == "responder"
        assert response.recipient_id == "requester"
        assert response.correlation_id == request.message_id

    def test_federation_protocol_initialization(self):
        """Test FederationProtocol initialization."""
        from src.federation.protocol import FederationProtocol

        protocol = FederationProtocol(node_id="test-node")

        assert protocol.node_id == "test-node"
        assert FederationProtocol.VERSION == "1.0"

    @pytest.mark.asyncio
    async def test_federation_protocol_receive_message(self):
        """Test receiving and processing messages."""
        from src.federation.protocol import (
            FederationProtocol,
            FederationMessage,
            MessageType,
        )

        protocol = FederationProtocol(node_id="test-node")

        # Test PING message gets PONG response
        ping = FederationMessage.create(
            message_type=MessageType.PING,
            sender_id="remote-node",
            recipient_id="test-node",
            payload={},
        )

        response = await protocol.receive_message(ping)

        assert response is not None
        assert response.message_type == MessageType.PONG

    @pytest.mark.asyncio
    async def test_federation_protocol_send_request(self):
        """Test sending request and waiting for response."""
        from src.federation.protocol import (
            FederationProtocol,
            FederationMessage,
            MessageType,
        )

        protocol = FederationProtocol(node_id="test-node")

        msg = FederationMessage.create(
            message_type=MessageType.DATA_REQUEST,
            sender_id="test-node",
            recipient_id="remote-node",
            payload={"action": "test"},
        )

        # Simulate response in background
        async def send_response():
            await asyncio.sleep(0.1)
            response = FederationMessage.create_response(
                msg,
                MessageType.DATA_RESPONSE,
                payload={"result": "success"},
            )
            await protocol.receive_message(response)

        asyncio.create_task(send_response())

        response = await protocol.send_request(msg, timeout=1.0)

        assert response is not None
        assert response.payload["result"] == "success"

    def test_create_protocol_function(self):
        """Test create_protocol helper function."""
        from src.federation.protocol import create_protocol

        protocol = create_protocol(node_id="test-node")

        assert protocol.node_id == "test-node"

    def test_protocol_create_hello(self):
        """Test creating hello message for handshake."""
        from src.federation.protocol import FederationProtocol, MessageType

        protocol = FederationProtocol(node_id="test-node")
        hello = protocol.create_hello("peer-node")

        assert hello.message_type == MessageType.HELLO
        assert hello.sender_id == "test-node"
        assert hello.recipient_id == "peer-node"
        assert hello.payload["protocol_version"] == "1.0"

    def test_protocol_stats(self):
        """Test protocol statistics."""
        from src.federation.protocol import FederationProtocol

        protocol = FederationProtocol(node_id="test-node")

        stats = protocol.get_stats()
        assert stats["messages_sent"] == 0
        assert stats["messages_received"] == 0


class TestCryptoModule:
    """Tests for cryptographic operations."""

    def test_cipher_suite_enum(self):
        """Test CipherSuite enum values."""
        from src.federation.crypto import CipherSuite

        assert CipherSuite.AES_256_GCM.value == "aes-256-gcm"
        assert CipherSuite.CHACHA20_POLY1305.value == "chacha20-poly1305"

    def test_key_exchange_method_enum(self):
        """Test KeyExchangeMethod enum values."""
        from src.federation.crypto import KeyExchangeMethod

        assert KeyExchangeMethod.X25519.value == "x25519"
        assert KeyExchangeMethod.ECDH_P256.value == "ecdh-p256"

    def test_session_key_creation(self):
        """Test SessionKey dataclass creation."""
        from src.federation.crypto import SessionKey, CipherSuite

        key = SessionKey(
            key_id="key-123",
            key_data=b"0" * 32,
            peer_id="peer-node",
            cipher_suite=CipherSuite.AES_256_GCM,
        )

        assert len(key.key_data) == 32
        assert key.cipher_suite == CipherSuite.AES_256_GCM
        assert key.peer_id == "peer-node"

    def test_session_key_is_expired(self):
        """Test session key expiration check."""
        from src.federation.crypto import SessionKey, CipherSuite

        # Expired key
        expired_key = SessionKey(
            key_id="key-1",
            key_data=b"0" * 32,
            peer_id="peer",
            cipher_suite=CipherSuite.AES_256_GCM,
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        assert expired_key.is_expired is True
        assert expired_key.is_valid is False

        # Valid key
        valid_key = SessionKey(
            key_id="key-2",
            key_data=b"0" * 32,
            peer_id="peer",
            cipher_suite=CipherSuite.AES_256_GCM,
            expires_at=datetime.utcnow() + timedelta(hours=24),
        )
        assert valid_key.is_expired is False
        assert valid_key.is_valid is True

    def test_encrypted_message_creation(self):
        """Test EncryptedMessage dataclass creation."""
        from src.federation.crypto import EncryptedMessage, CipherSuite

        msg = EncryptedMessage(
            ciphertext=b"encrypted_data",
            nonce=b"nonce_12_bytes",
            key_id="key-123",
            cipher_suite=CipherSuite.AES_256_GCM,
            sender_id="sender",
            recipient_id="recipient",
        )

        assert msg.ciphertext == b"encrypted_data"
        assert msg.nonce == b"nonce_12_bytes"
        assert msg.sender_id == "sender"

    def test_encrypted_message_to_dict(self):
        """Test EncryptedMessage serialization."""
        from src.federation.crypto import EncryptedMessage, CipherSuite

        msg = EncryptedMessage(
            ciphertext=b"encrypted",
            nonce=b"nonce12bytes",
            key_id="key-123",
            cipher_suite=CipherSuite.AES_256_GCM,
            sender_id="sender",
            recipient_id="recipient",
            auth_tag=b"tag16bytes12345!",
        )

        data = msg.to_dict()
        assert "ciphertext" in data
        assert "nonce" in data
        assert data["cipher_suite"] == "aes-256-gcm"

    def test_encrypted_message_from_dict(self):
        """Test EncryptedMessage deserialization."""
        from src.federation.crypto import EncryptedMessage, CipherSuite

        data = {
            "ciphertext": base64.b64encode(b"encrypted_data").decode(),
            "nonce": base64.b64encode(b"nonce_data12").decode(),
            "key_id": "key-123",
            "cipher_suite": "aes-256-gcm",
            "sender_id": "sender",
            "recipient_id": "recipient",
        }

        msg = EncryptedMessage.from_dict(data)
        assert msg.cipher_suite == CipherSuite.AES_256_GCM
        assert msg.ciphertext == b"encrypted_data"

    def test_default_crypto_provider_initialization(self):
        """Test DefaultCryptoProvider initialization."""
        from src.federation.crypto import DefaultCryptoProvider

        crypto = DefaultCryptoProvider()
        assert crypto is not None

    def test_default_crypto_provider_encrypt_decrypt(self):
        """Test encryption and decryption."""
        from src.federation.crypto import DefaultCryptoProvider, SessionKey, CipherSuite

        crypto = DefaultCryptoProvider()
        key = SessionKey(
            key_id="key-123",
            key_data=b"0123456789abcdef0123456789abcdef",  # 32 bytes
            peer_id="peer",
            cipher_suite=CipherSuite.AES_256_GCM,
        )

        plaintext = b"Hello, Federation!"
        ciphertext, nonce, auth_tag = crypto.encrypt(plaintext, key)

        assert ciphertext != plaintext

        decrypted = crypto.decrypt(ciphertext, nonce, key, auth_tag)
        assert decrypted == plaintext

    def test_default_crypto_provider_key_exchange(self):
        """Test key exchange."""
        from src.federation.crypto import DefaultCryptoProvider, KeyExchangeMethod

        crypto1 = DefaultCryptoProvider()
        crypto2 = DefaultCryptoProvider()

        # Generate key pairs
        pub1, priv1 = crypto1.generate_key_pair(KeyExchangeMethod.X25519)
        pub2, priv2 = crypto2.generate_key_pair(KeyExchangeMethod.X25519)

        assert pub1 is not None
        assert pub2 is not None
        assert len(pub1) == 32  # X25519 public key

        # Derive shared secrets
        secret1 = crypto1.derive_shared_secret(priv1, pub2, KeyExchangeMethod.X25519)
        secret2 = crypto2.derive_shared_secret(priv2, pub1, KeyExchangeMethod.X25519)

        # Secrets should match
        assert secret1 == secret2

    def test_mock_crypto_provider(self):
        """Test MockCryptoProvider."""
        from src.federation.crypto import MockCryptoProvider, SessionKey, CipherSuite

        crypto = MockCryptoProvider()
        key = SessionKey(
            key_id="key-123",
            key_data=b"0123456789abcdef0123456789abcdef",
            peer_id="peer",
            cipher_suite=CipherSuite.AES_256_GCM,
        )

        plaintext = b"Test message"
        ciphertext, nonce, auth_tag = crypto.encrypt(plaintext, key)
        decrypted = crypto.decrypt(ciphertext, nonce, key, auth_tag)

        assert decrypted == plaintext

    def test_session_manager_creation(self):
        """Test SessionManager initialization."""
        from src.federation.crypto import SessionManager

        manager = SessionManager(node_id="test-node")
        assert manager.node_id == "test-node"

    def test_session_manager_create_session(self):
        """Test creating a session with a peer."""
        from src.federation.crypto import (
            SessionManager,
            DefaultCryptoProvider,
            KeyExchangeMethod,
        )

        manager = SessionManager(node_id="test-node")

        # Get peer's public key (simulated)
        peer_crypto = DefaultCryptoProvider()
        peer_pub, _ = peer_crypto.generate_key_pair(KeyExchangeMethod.X25519)

        session = manager.create_session(
            peer_id="peer-node",
            peer_public_key=peer_pub,
            method=KeyExchangeMethod.X25519,
        )

        assert session is not None
        assert session.peer_id == "peer-node"
        assert len(session.key_data) == 32

    def test_session_manager_encrypt_for_peer(self):
        """Test encrypting message for a peer."""
        from src.federation.crypto import SessionManager, DefaultCryptoProvider, KeyExchangeMethod

        manager = SessionManager(node_id="test-node")

        # Set up session
        peer_crypto = DefaultCryptoProvider()
        peer_pub, _ = peer_crypto.generate_key_pair(KeyExchangeMethod.X25519)
        manager.create_session("peer-node", peer_pub, KeyExchangeMethod.X25519)

        # Encrypt
        plaintext = b"Secret message"
        encrypted = manager.encrypt_for_peer("peer-node", plaintext)

        assert encrypted is not None
        assert encrypted.ciphertext != plaintext

    def test_create_crypto_provider_function(self):
        """Test create_crypto_provider helper function."""
        from src.federation.crypto import create_crypto_provider, DefaultCryptoProvider, MockCryptoProvider

        default_crypto = create_crypto_provider()
        assert isinstance(default_crypto, DefaultCryptoProvider)

        mock_crypto = create_crypto_provider(provider_type="mock")
        assert isinstance(mock_crypto, MockCryptoProvider)

    def test_encrypt_decrypt_message_functions(self):
        """Test encrypt_message and decrypt_message helper functions."""
        from src.federation.crypto import (
            encrypt_message,
            decrypt_message,
            SessionKey,
            CipherSuite,
        )

        key = SessionKey(
            key_id="key-123",
            key_data=b"0123456789abcdef0123456789abcdef",
            peer_id="peer",
            cipher_suite=CipherSuite.AES_256_GCM,
        )

        plaintext = b"Test message"
        encrypted = encrypt_message(plaintext, key, "sender", "recipient")

        assert encrypted is not None
        decrypted = decrypt_message(encrypted, key)
        assert decrypted == plaintext


class TestPermissionsModule:
    """Tests for permission management."""

    def test_permission_level_enum(self):
        """Test PermissionLevel enum values."""
        from src.federation.permissions import PermissionLevel

        assert PermissionLevel.NONE.value == "none"
        assert PermissionLevel.READ.value == "read"
        assert PermissionLevel.WRITE.value == "write"
        assert PermissionLevel.ADMIN.value == "admin"

    def test_permission_scope_enum(self):
        """Test PermissionScope enum values."""
        from src.federation.permissions import PermissionScope

        assert PermissionScope.IDENTITY.value == "identity"
        assert PermissionScope.MESSAGES.value == "messages"
        assert PermissionScope.DATA.value == "data"

    def test_permission_creation(self):
        """Test Permission dataclass creation."""
        from src.federation.permissions import Permission, PermissionLevel, PermissionScope

        perm = Permission(
            scope=PermissionScope.DATA,
            level=PermissionLevel.READ,
        )

        assert perm.scope == PermissionScope.DATA
        assert perm.level == PermissionLevel.READ

    def test_permission_with_resource(self):
        """Test Permission with specific resource."""
        from src.federation.permissions import Permission, PermissionLevel, PermissionScope

        perm = Permission(
            scope=PermissionScope.DATA,
            level=PermissionLevel.WRITE,
            resource="dataset-123",
        )

        assert perm.resource == "dataset-123"

    def test_permission_matches(self):
        """Test Permission.matches method."""
        from src.federation.permissions import Permission, PermissionLevel, PermissionScope

        # Write permission should match read requirement
        write_perm = Permission(PermissionScope.DATA, PermissionLevel.WRITE)
        read_perm = Permission(PermissionScope.DATA, PermissionLevel.READ)

        assert write_perm.matches(read_perm) is True
        assert read_perm.matches(write_perm) is False

    def test_permission_to_dict(self):
        """Test Permission serialization."""
        from src.federation.permissions import Permission, PermissionLevel, PermissionScope

        perm = Permission(
            scope=PermissionScope.TOOLS,
            level=PermissionLevel.READ,
            resource="tool-abc",
        )

        data = perm.to_dict()
        assert data["scope"] == "tools"
        assert data["level"] == "read"
        assert data["resource"] == "tool-abc"

    def test_permission_from_dict(self):
        """Test Permission deserialization."""
        from src.federation.permissions import Permission, PermissionLevel, PermissionScope

        data = {"scope": "memory", "level": "write", "resource": "*"}

        perm = Permission.from_dict(data)
        assert perm.scope == PermissionScope.MEMORY
        assert perm.level == PermissionLevel.WRITE

    def test_permission_set_creation(self):
        """Test PermissionSet creation."""
        from src.federation.permissions import PermissionSet, Permission, PermissionLevel, PermissionScope

        perm1 = Permission(PermissionScope.DATA, PermissionLevel.READ)
        perm2 = Permission(PermissionScope.TOOLS, PermissionLevel.WRITE)

        perm_set = PermissionSet(permissions=[perm1, perm2])

        assert len(perm_set.permissions) == 2

    def test_permission_set_has_permission(self):
        """Test PermissionSet.has_permission check."""
        from src.federation.permissions import PermissionSet, Permission, PermissionLevel, PermissionScope

        perm_set = PermissionSet(
            permissions=[
                Permission(PermissionScope.DATA, PermissionLevel.WRITE),
                Permission(PermissionScope.TOOLS, PermissionLevel.READ),
            ]
        )

        # Has write permission on data (covers read)
        assert perm_set.has_permission(Permission(PermissionScope.DATA, PermissionLevel.READ)) is True
        assert perm_set.has_permission(Permission(PermissionScope.DATA, PermissionLevel.WRITE)) is True

        # Only has read on tools
        assert perm_set.has_permission(Permission(PermissionScope.TOOLS, PermissionLevel.READ)) is True
        assert perm_set.has_permission(Permission(PermissionScope.TOOLS, PermissionLevel.WRITE)) is False

    def test_permission_set_add_permission(self):
        """Test adding permissions to a set."""
        from src.federation.permissions import PermissionSet, Permission, PermissionLevel, PermissionScope

        perm_set = PermissionSet()
        assert len(perm_set.permissions) == 0

        perm_set.add(Permission(PermissionScope.DATA, PermissionLevel.READ))
        assert len(perm_set.permissions) == 1

    def test_permission_request_creation(self):
        """Test PermissionRequest creation."""
        from src.federation.permissions import (
            PermissionRequest,
            PermissionSet,
            Permission,
            PermissionLevel,
            PermissionScope,
        )

        request = PermissionRequest(
            request_id="req-123",
            requester_id="node-a",
            target_id="node-b",
            permissions=PermissionSet(permissions=[Permission(PermissionScope.DATA, PermissionLevel.READ)]),
            reason="Need to query data",
        )

        assert request.request_id == "req-123"
        assert request.requester_id == "node-a"
        assert len(request.permissions.permissions) == 1

    def test_permission_request_to_dict(self):
        """Test PermissionRequest serialization."""
        from src.federation.permissions import (
            PermissionRequest,
            PermissionSet,
            Permission,
            PermissionLevel,
            PermissionScope,
        )

        request = PermissionRequest(
            request_id="req-456",
            requester_id="node-x",
            target_id="node-y",
            permissions=PermissionSet(permissions=[Permission(PermissionScope.MEMORY, PermissionLevel.WRITE)]),
        )

        data = request.to_dict()
        assert data["request_id"] == "req-456"
        assert "permissions" in data

    def test_permission_grant_creation(self):
        """Test PermissionGrant creation."""
        from src.federation.permissions import (
            PermissionGrant,
            PermissionSet,
            Permission,
            PermissionLevel,
            PermissionScope,
        )

        grant = PermissionGrant(
            grant_id="grant-789",
            granter_id="node-b",
            grantee_id="node-a",
            permissions=PermissionSet(permissions=[Permission(PermissionScope.DATA, PermissionLevel.READ)]),
            request_id="req-123",
        )

        assert grant.grant_id == "grant-789"
        assert grant.granter_id == "node-b"

    def test_permission_grant_is_valid(self):
        """Test PermissionGrant validity check."""
        from src.federation.permissions import (
            PermissionGrant,
            PermissionSet,
            Permission,
            PermissionLevel,
            PermissionScope,
            PermissionStatus,
        )

        perms = PermissionSet(permissions=[Permission(PermissionScope.DATA, PermissionLevel.READ)])

        # Valid grant
        valid_grant = PermissionGrant(
            grant_id="g1",
            granter_id="granter",
            grantee_id="grantee",
            permissions=perms,
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        assert valid_grant.is_valid is True

        # Expired grant
        expired_grant = PermissionGrant(
            grant_id="g2",
            granter_id="granter",
            grantee_id="grantee",
            permissions=perms,
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        assert expired_grant.is_valid is False

        # Revoked grant
        revoked_grant = PermissionGrant(
            grant_id="g3",
            granter_id="granter",
            grantee_id="grantee",
            permissions=perms,
        )
        revoked_grant.revoke()
        assert revoked_grant.is_valid is False
        assert revoked_grant.status == PermissionStatus.REVOKED

    def test_permission_manager_initialization(self):
        """Test PermissionManager initialization."""
        from src.federation.permissions import PermissionManager

        manager = PermissionManager(node_id="test-node")
        assert manager.node_id == "test-node"

    def test_permission_manager_create_request(self):
        """Test creating a permission request."""
        from src.federation.permissions import (
            PermissionManager,
            Permission,
            PermissionLevel,
            PermissionScope,
        )

        manager = PermissionManager(node_id="node-a")

        request = manager.create_request(
            target_id="node-b",
            permissions=[Permission(PermissionScope.DATA, PermissionLevel.READ)],
            reason="Need access",
        )

        assert request.requester_id == "node-a"
        assert request.target_id == "node-b"
        assert request.request_id is not None

    def test_permission_manager_approve_request(self):
        """Test approving a permission request."""
        from src.federation.permissions import (
            PermissionManager,
            PermissionRequest,
            PermissionSet,
            Permission,
            PermissionLevel,
            PermissionScope,
        )

        manager = PermissionManager(node_id="node-b")

        request = PermissionRequest(
            request_id="req-test",
            requester_id="node-a",
            target_id="node-b",
            permissions=PermissionSet(permissions=[Permission(PermissionScope.DATA, PermissionLevel.READ)]),
        )

        # Store the request
        manager.receive_request(request)

        # Approve it
        grant = manager.approve_request(request.request_id)

        assert grant is not None
        assert grant.grantee_id == "node-a"
        assert grant.granter_id == "node-b"

    def test_permission_manager_deny_request(self):
        """Test denying a permission request."""
        from src.federation.permissions import (
            PermissionManager,
            PermissionRequest,
            PermissionSet,
            Permission,
            PermissionLevel,
            PermissionScope,
        )

        manager = PermissionManager(node_id="node-b")

        request = PermissionRequest(
            request_id="req-deny",
            requester_id="node-a",
            target_id="node-b",
            permissions=PermissionSet(permissions=[Permission(PermissionScope.ADMIN, PermissionLevel.ADMIN)]),
        )

        manager.receive_request(request)
        result = manager.deny_request(request.request_id, reason="Not authorized")

        assert result is True

    def test_permission_manager_check_permission(self):
        """Test checking if a peer has permission."""
        from src.federation.permissions import (
            PermissionManager,
            PermissionGrant,
            PermissionSet,
            Permission,
            PermissionLevel,
            PermissionScope,
        )

        manager = PermissionManager(node_id="node-b")

        # Add a grant
        grant = PermissionGrant(
            grant_id="g1",
            granter_id="node-b",
            grantee_id="node-a",
            permissions=PermissionSet(permissions=[Permission(PermissionScope.DATA, PermissionLevel.READ)]),
        )

        # Simulate receiving grant (store it internally)
        if "node-a" not in manager._grants_given:
            manager._grants_given["node-a"] = []
        manager._grants_given["node-a"].append(grant)

        # Check permission
        has_perm = manager.check_permission(
            "node-a", Permission(PermissionScope.DATA, PermissionLevel.READ)
        )
        assert has_perm is True

        # Check permission that wasn't granted
        no_perm = manager.check_permission(
            "node-a", Permission(PermissionScope.DATA, PermissionLevel.WRITE)
        )
        assert no_perm is False

    def test_permission_manager_revoke_grant(self):
        """Test revoking a permission grant."""
        from src.federation.permissions import (
            PermissionManager,
            PermissionGrant,
            PermissionSet,
            Permission,
            PermissionLevel,
            PermissionScope,
        )

        manager = PermissionManager(node_id="node-b")

        grant = PermissionGrant(
            grant_id="g-revoke",
            granter_id="node-b",
            grantee_id="node-a",
            permissions=PermissionSet(permissions=[Permission(PermissionScope.TOOLS, PermissionLevel.WRITE)]),
        )

        # Store grant
        if "node-a" not in manager._grants_given:
            manager._grants_given["node-a"] = []
        manager._grants_given["node-a"].append(grant)

        # Revoke
        result = manager.revoke_grant(grant.grant_id)
        assert result is True

        # Check permission is no longer valid
        has_perm = manager.check_permission(
            "node-a", Permission(PermissionScope.TOOLS, PermissionLevel.WRITE)
        )
        assert has_perm is False

    def test_create_permission_manager_function(self):
        """Test create_permission_manager helper function."""
        from src.federation.permissions import create_permission_manager

        manager = create_permission_manager(node_id="test-node")
        assert manager.node_id == "test-node"


class TestNodeModule:
    """Tests for federation node."""

    def test_node_state_enum(self):
        """Test NodeState enum values."""
        from src.federation.node import NodeState

        assert NodeState.STOPPED.value == "stopped"
        assert NodeState.STARTING.value == "starting"
        assert NodeState.RUNNING.value == "running"
        assert NodeState.STOPPING.value == "stopping"
        assert NodeState.ERROR.value == "error"

    def test_connection_state_enum(self):
        """Test ConnectionState enum values."""
        from src.federation.node import ConnectionState

        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.AUTHENTICATED.value == "authenticated"

    def test_node_config_creation(self):
        """Test NodeConfig dataclass creation."""
        from src.federation.node import NodeConfig

        config = NodeConfig(
            node_id="test-node",
            display_name="Test Node",
            listen_port=9000,
        )

        assert config.node_id == "test-node"
        assert config.display_name == "Test Node"
        assert config.listen_port == 9000

    def test_node_config_defaults(self):
        """Test NodeConfig default values."""
        from src.federation.node import NodeConfig

        config = NodeConfig(node_id="test")

        assert config.display_name == "test"  # Defaults to node_id
        assert config.listen_address == "0.0.0.0"
        assert config.listen_port == 8765
        assert config.max_peers == 100
        assert config.heartbeat_interval == 30.0
        assert config.require_encryption is True
        assert config.require_authentication is True

    def test_peer_info_creation(self):
        """Test PeerInfo dataclass creation."""
        from src.federation.node import PeerInfo, ConnectionState

        info = PeerInfo(
            node_id="peer-node",
            address="192.168.1.100",
            port=8765,
        )

        assert info.node_id == "peer-node"
        assert info.address == "192.168.1.100"
        assert info.state == ConnectionState.DISCONNECTED

    def test_peer_connection_creation(self):
        """Test PeerConnection dataclass creation."""
        from src.federation.node import PeerConnection, PeerInfo

        peer_info = PeerInfo(node_id="peer")
        connection = PeerConnection(peer_id="peer", peer_info=peer_info)

        assert connection.peer_id == "peer"
        assert connection.reader is None
        assert connection.writer is None

    def test_federation_node_initialization(self):
        """Test FederationNode initialization."""
        from src.federation.node import FederationNode, NodeConfig, NodeState

        config = NodeConfig(node_id="test-node")
        node = FederationNode(config)

        assert node.node_id == "test-node"
        assert node.state == NodeState.STOPPED
        assert node.peer_count == 0

    def test_federation_node_properties(self):
        """Test FederationNode properties."""
        from src.federation.node import FederationNode, NodeConfig

        config = NodeConfig(
            node_id="my-node",
            display_name="My Node",
            listen_port=9999,
        )
        node = FederationNode(config)

        assert node.node_id == "my-node"
        assert node.peers == []
        assert node.peer_count == 0

    @pytest.mark.asyncio
    async def test_federation_node_start_stop(self):
        """Test starting and stopping the node."""
        from src.federation.node import FederationNode, NodeConfig, NodeState

        config = NodeConfig(
            node_id="test-node",
            listen_port=0,  # Use random available port
        )
        node = FederationNode(config)

        # Start node
        started = await node.start()
        assert started is True
        assert node.state == NodeState.RUNNING

        # Stop node
        stopped = await node.stop()
        assert stopped is True
        assert node.state == NodeState.STOPPED

    @pytest.mark.asyncio
    async def test_federation_node_double_start(self):
        """Test that double start is prevented."""
        from src.federation.node import FederationNode, NodeConfig

        config = NodeConfig(node_id="test-node", listen_port=0)
        node = FederationNode(config)

        await node.start()
        result = await node.start()  # Second start should fail

        assert result is False
        await node.stop()

    def test_federation_node_event_handlers(self):
        """Test event handler registration."""
        from src.federation.node import FederationNode, NodeConfig

        config = NodeConfig(node_id="test-node", listen_port=0)
        node = FederationNode(config)

        connected_peers = []
        disconnected_peers = []

        node.on_peer_connected(lambda peer_id: connected_peers.append(peer_id))
        node.on_peer_disconnected(lambda peer_id: disconnected_peers.append(peer_id))

        # Handlers should be registered
        assert len(node._on_peer_connected) == 1
        assert len(node._on_peer_disconnected) == 1

    def test_federation_node_is_connected(self):
        """Test is_connected method."""
        from src.federation.node import FederationNode, NodeConfig

        config = NodeConfig(node_id="test-node", listen_port=0)
        node = FederationNode(config)

        # Not connected to any peer
        assert node.is_connected("unknown-peer") is False

    def test_federation_node_get_peer_info(self):
        """Test get_peer_info method."""
        from src.federation.node import FederationNode, NodeConfig

        config = NodeConfig(node_id="test-node", listen_port=0)
        node = FederationNode(config)

        # No peer info for unknown peer
        info = node.get_peer_info("unknown-peer")
        assert info is None

    def test_create_federation_node_function(self):
        """Test create_federation_node helper function."""
        from src.federation.node import create_federation_node, NodeState

        node = create_federation_node(
            node_id="my-node",
            display_name="My Federation Node",
            listen_port=8888,
        )

        assert node.node_id == "my-node"
        assert node.state == NodeState.STOPPED
        assert node.config.listen_port == 8888


class TestModuleExports:
    """Tests for module exports."""

    def test_identity_exports(self):
        """Test identity module exports."""
        from src.federation import (
            Identity,
            IdentityManager,
            KeyPair,
            PublicKey,
            PrivateKey,
            Certificate,
            create_identity,
            verify_identity,
        )

        assert Identity is not None
        assert IdentityManager is not None
        assert KeyPair is not None

    def test_protocol_exports(self):
        """Test protocol module exports."""
        from src.federation import (
            FederationMessage,
            MessageType,
            FederationProtocol,
            ProtocolHandler,
            create_protocol,
        )

        assert FederationMessage is not None
        assert MessageType is not None
        assert FederationProtocol is not None

    def test_crypto_exports(self):
        """Test crypto module exports."""
        from src.federation import (
            CryptoProvider,
            EncryptedMessage,
            SessionKey,
            create_crypto_provider,
            encrypt_message,
            decrypt_message,
        )

        assert CryptoProvider is not None
        assert EncryptedMessage is not None
        assert SessionKey is not None

    def test_permissions_exports(self):
        """Test permissions module exports."""
        from src.federation import (
            Permission,
            PermissionLevel,
            PermissionSet,
            PermissionRequest,
            PermissionGrant,
            PermissionManager,
            create_permission_manager,
        )

        assert Permission is not None
        assert PermissionLevel is not None
        assert PermissionManager is not None

    def test_node_exports(self):
        """Test node module exports."""
        from src.federation import (
            FederationNode,
            NodeConfig,
            NodeState,
            ConnectionState,
            PeerConnection,
            PeerInfo,
            create_federation_node,
        )

        assert FederationNode is not None
        assert NodeConfig is not None
        assert NodeState is not None
        assert ConnectionState is not None


class TestIntegration:
    """Integration tests for federation module."""

    @pytest.mark.asyncio
    async def test_two_nodes_connect(self):
        """Test two nodes connecting to each other."""
        from src.federation.node import create_federation_node, NodeState

        # Create two nodes
        node1 = create_federation_node(
            node_id="node-1",
            listen_port=0,
            require_authentication=False,
            require_encryption=False,
        )
        node2 = create_federation_node(
            node_id="node-2",
            listen_port=0,
            require_authentication=False,
            require_encryption=False,
        )

        try:
            # Start both nodes
            await node1.start()
            await node2.start()

            assert node1.state == NodeState.RUNNING
            assert node2.state == NodeState.RUNNING

            # Get node1's actual port
            if node1._server and node1._server.sockets:
                port1 = node1._server.sockets[0].getsockname()[1]

                # Node2 connects to node1
                peer_id = await node2.connect_to_peer("127.0.0.1", port1)

                if peer_id:
                    assert node2.is_connected(peer_id)
                    await asyncio.sleep(0.1)  # Let connection establish

        finally:
            await node1.stop()
            await node2.stop()

    def test_identity_and_signing(self):
        """Test identity creation and message signing."""
        from src.federation.identity import IdentityManager

        # Create two identity managers
        alice = IdentityManager(node_id="alice")
        bob = IdentityManager(node_id="bob")

        # Alice signs a message
        message = b"Hello from Alice"
        signature = alice.sign(message)

        # Bob verifies the signature
        is_valid = bob.verify_signature(message, signature, alice.identity.public_key)
        assert is_valid is True

        # Tampered message should fail
        is_valid = bob.verify_signature(
            b"Tampered message", signature, alice.identity.public_key
        )
        assert is_valid is False

    def test_permission_workflow(self):
        """Test complete permission request/grant workflow."""
        from src.federation.permissions import (
            PermissionManager,
            Permission,
            PermissionLevel,
            PermissionScope,
        )

        # Node A wants permission from Node B
        node_a = PermissionManager(node_id="node-a")
        node_b = PermissionManager(node_id="node-b")

        # Node A creates a request
        request = node_a.create_request(
            target_id="node-b",
            permissions=[
                Permission(PermissionScope.DATA, PermissionLevel.READ),
                Permission(PermissionScope.TOOLS, PermissionLevel.WRITE),
            ],
            reason="Need to collaborate on data tasks",
        )

        # Node B receives and stores the request
        node_b.receive_request(request)

        # Node B approves with modified permissions (only read access)
        grant = node_b.approve_request(
            request.request_id,
            approved_permissions=[Permission(PermissionScope.DATA, PermissionLevel.READ)],
        )

        assert grant is not None

        # Node A receives the grant
        node_a.receive_grant(grant)

        # Check permissions at node_a (what it received from node_b)
        assert node_a.check_own_permission(
            "node-b", Permission(PermissionScope.DATA, PermissionLevel.READ)
        )

    def test_encryption_roundtrip(self):
        """Test complete encryption/decryption cycle."""
        from src.federation.crypto import (
            DefaultCryptoProvider,
            SessionKey,
            CipherSuite,
        )
        import secrets

        crypto = DefaultCryptoProvider()

        # Create session key
        key = SessionKey(
            key_id="key-123",
            key_data=secrets.token_bytes(32),
            peer_id="peer",
            cipher_suite=CipherSuite.AES_256_GCM,
        )

        # Encrypt various messages
        messages = [
            b"Hello, World!",
            b"A" * 10000,  # Large message
            bytes(range(256)),  # All byte values
        ]

        for original in messages:
            ciphertext, nonce, auth_tag = crypto.encrypt(original, key)
            assert ciphertext != original

            decrypted = crypto.decrypt(ciphertext, nonce, key, auth_tag)
            assert decrypted == original

    def test_key_exchange_session(self):
        """Test establishing an encrypted session between two nodes."""
        from src.federation.crypto import SessionManager, KeyExchangeMethod

        # Create session managers for two nodes
        alice_mgr = SessionManager(node_id="alice")
        bob_mgr = SessionManager(node_id="bob")

        # Generate ephemeral key pairs
        alice_pub, alice_priv = alice_mgr.crypto.generate_key_pair(KeyExchangeMethod.X25519)
        bob_pub, bob_priv = bob_mgr.crypto.generate_key_pair(KeyExchangeMethod.X25519)

        # Each side creates session with peer's public key
        alice_session = alice_mgr.create_session("bob", bob_pub, KeyExchangeMethod.X25519)
        bob_session = bob_mgr.create_session("alice", alice_pub, KeyExchangeMethod.X25519)

        assert alice_session is not None
        assert bob_session is not None
        assert alice_session.peer_id == "bob"
        assert bob_session.peer_id == "alice"
