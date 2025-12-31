"""
Agent OS Federation Protocol Module

Provides secure inter-instance communication between Agent OS nodes including:
- Identity verification and key management
- Inter-instance messaging protocol
- Permission negotiation
- End-to-end encryption

Usage:
    from src.federation import FederationNode, create_federation_node

    node = create_federation_node(node_id="my-agent")
    node.start()
    node.connect_peer("peer-agent", "https://peer.example.com")
"""

from .crypto import (
    CryptoProvider,
    EncryptedMessage,
    SessionKey,
    create_crypto_provider,
    decrypt_message,
    encrypt_message,
)
from .identity import (
    Certificate,
    Identity,
    IdentityManager,
    KeyPair,
    PrivateKey,
    PublicKey,
    create_identity,
    verify_identity,
)
from .node import (
    ConnectionState,
    FederationNode,
    NodeConfig,
    NodeState,
    PeerConnection,
    PeerInfo,
    create_federation_node,
)
from .permissions import (
    Permission,
    PermissionGrant,
    PermissionLevel,
    PermissionManager,
    PermissionRequest,
    PermissionSet,
    create_permission_manager,
)
from .protocol import (
    FederationMessage,
    FederationProtocol,
    MessageType,
    ProtocolHandler,
    create_protocol,
)

__all__ = [
    # Identity
    "Identity",
    "IdentityManager",
    "KeyPair",
    "PublicKey",
    "PrivateKey",
    "Certificate",
    "create_identity",
    "verify_identity",
    # Protocol
    "FederationMessage",
    "MessageType",
    "FederationProtocol",
    "ProtocolHandler",
    "create_protocol",
    # Permissions
    "Permission",
    "PermissionLevel",
    "PermissionSet",
    "PermissionRequest",
    "PermissionGrant",
    "PermissionManager",
    "create_permission_manager",
    # Crypto
    "CryptoProvider",
    "EncryptedMessage",
    "SessionKey",
    "create_crypto_provider",
    "encrypt_message",
    "decrypt_message",
    # Node
    "FederationNode",
    "NodeConfig",
    "NodeState",
    "ConnectionState",
    "PeerConnection",
    "PeerInfo",
    "create_federation_node",
]
