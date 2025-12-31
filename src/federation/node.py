"""Federation Node - Main integration point for inter-instance communication.

This module provides the FederationNode class that coordinates:
- Identity management and verification
- Secure communication protocols
- Permission negotiation
- E2E encrypted messaging
- Peer discovery and connection management
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from .crypto import (
    CryptoProvider,
    EncryptedMessage,
    KeyExchangeMethod,
    SessionKey,
    create_crypto_provider,
)
from .identity import Identity, IdentityManager, create_identity
from .permissions import (
    Permission,
    PermissionGrant,
    PermissionLevel,
    PermissionManager,
    PermissionRequest,
    create_permission_manager,
)
from .protocol import (
    FederationMessage,
    FederationProtocol,
    MessageType,
    ProtocolHandler,
    create_protocol,
)

logger = logging.getLogger(__name__)


class NodeState(Enum):
    """Federation node state."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class ConnectionState(Enum):
    """Peer connection state."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    HANDSHAKING = "handshaking"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ERROR = "error"


@dataclass
class NodeConfig:
    """Configuration for a federation node."""

    node_id: str
    display_name: str = ""
    listen_address: str = "0.0.0.0"
    listen_port: int = 8765
    max_peers: int = 100
    connection_timeout: float = 30.0
    heartbeat_interval: float = 30.0
    auto_reconnect: bool = True
    require_encryption: bool = True
    require_authentication: bool = True
    storage_path: Optional[str] = None

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.node_id


@dataclass
class PeerInfo:
    """Information about a connected peer."""

    node_id: str
    identity: Optional[Identity] = None
    address: str = ""
    port: int = 0
    state: ConnectionState = ConnectionState.DISCONNECTED
    connected_at: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    session_key: Optional[SessionKey] = None
    permissions: Optional[PermissionGrant] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PeerConnection:
    """Active connection to a peer."""

    peer_id: str
    peer_info: PeerInfo
    reader: Optional[asyncio.StreamReader] = None
    writer: Optional[asyncio.StreamWriter] = None
    _send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _receive_task: Optional[asyncio.Task] = None

    async def send(self, data: bytes) -> bool:
        """Send data to peer."""
        if not self.writer:
            return False
        try:
            async with self._send_lock:
                # Length-prefixed message
                length = len(data).to_bytes(4, "big")
                self.writer.write(length + data)
                await self.writer.drain()
            return True
        except Exception as e:
            logger.error(f"Failed to send to {self.peer_id}: {e}")
            return False

    async def receive(self) -> Optional[bytes]:
        """Receive data from peer."""
        if not self.reader:
            return None
        try:
            # Read length prefix
            length_bytes = await self.reader.readexactly(4)
            length = int.from_bytes(length_bytes, "big")
            if length > 10 * 1024 * 1024:  # 10MB max
                raise ValueError(f"Message too large: {length}")
            data = await self.reader.readexactly(length)
            return data
        except asyncio.IncompleteReadError:
            return None
        except Exception as e:
            logger.error(f"Failed to receive from {self.peer_id}: {e}")
            return None

    async def close(self):
        """Close the connection."""
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self.writer:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except Exception:
                pass


class FederationNode:
    """Main federation node for inter-instance communication.

    Coordinates all federation components:
    - Identity management
    - Protocol handling
    - Encryption
    - Permission management
    """

    def __init__(self, config: NodeConfig):
        """Initialize federation node.

        Args:
            config: Node configuration
        """
        self.config = config
        self._state = NodeState.STOPPED
        self._state_lock = asyncio.Lock()

        # Core components
        self._identity_manager: Optional[IdentityManager] = None
        self._protocol: Optional[FederationProtocol] = None
        self._crypto: Optional[CryptoProvider] = None
        self._permission_manager: Optional[PermissionManager] = None

        # Connection management
        self._peers: Dict[str, PeerConnection] = {}
        self._peer_info: Dict[str, PeerInfo] = {}
        self._server: Optional[asyncio.Server] = None

        # Event handlers
        self._on_peer_connected: List[Callable[[str], None]] = []
        self._on_peer_disconnected: List[Callable[[str], None]] = []
        self._on_message: List[Callable[[str, FederationMessage], None]] = []
        self._on_permission_request: List[Callable[[str, PermissionRequest], None]] = []

        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

    @property
    def state(self) -> NodeState:
        """Get current node state."""
        return self._state

    @property
    def node_id(self) -> str:
        """Get node ID."""
        return self.config.node_id

    @property
    def identity(self) -> Optional[Identity]:
        """Get node identity."""
        if self._identity_manager:
            return self._identity_manager.identity
        return None

    @property
    def peers(self) -> List[str]:
        """Get list of connected peer IDs."""
        return list(self._peers.keys())

    @property
    def peer_count(self) -> int:
        """Get number of connected peers."""
        return len(self._peers)

    async def start(self) -> bool:
        """Start the federation node.

        Returns:
            True if started successfully
        """
        async with self._state_lock:
            if self._state != NodeState.STOPPED:
                logger.warning(f"Cannot start node in state {self._state}")
                return False

            self._state = NodeState.STARTING

        try:
            # Initialize components
            await self._initialize_components()

            # Start server
            self._server = await asyncio.start_server(
                self._handle_connection,
                self.config.listen_address,
                self.config.listen_port,
            )

            # Start background tasks
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            async with self._state_lock:
                self._state = NodeState.RUNNING

            logger.info(
                f"Federation node {self.config.node_id} started on "
                f"{self.config.listen_address}:{self.config.listen_port}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to start federation node: {e}")
            async with self._state_lock:
                self._state = NodeState.ERROR
            return False

    async def stop(self) -> bool:
        """Stop the federation node.

        Returns:
            True if stopped successfully
        """
        async with self._state_lock:
            if self._state not in (NodeState.RUNNING, NodeState.ERROR):
                return False
            self._state = NodeState.STOPPING

        try:
            # Cancel background tasks
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass

            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            # Disconnect all peers
            for peer_id in list(self._peers.keys()):
                await self.disconnect_peer(peer_id)

            # Stop server
            if self._server:
                self._server.close()
                await self._server.wait_closed()

            async with self._state_lock:
                self._state = NodeState.STOPPED

            logger.info(f"Federation node {self.config.node_id} stopped")
            return True

        except Exception as e:
            logger.error(f"Error stopping federation node: {e}")
            async with self._state_lock:
                self._state = NodeState.ERROR
            return False

    async def connect_to_peer(self, address: str, port: int) -> Optional[str]:
        """Connect to a remote peer.

        Args:
            address: Peer address
            port: Peer port

        Returns:
            Peer ID if connected, None otherwise
        """
        if self._state != NodeState.RUNNING:
            logger.warning("Cannot connect: node not running")
            return None

        if len(self._peers) >= self.config.max_peers:
            logger.warning("Cannot connect: max peers reached")
            return None

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(address, port),
                timeout=self.config.connection_timeout,
            )

            # Create temporary connection
            temp_peer_id = f"pending_{address}:{port}"
            peer_info = PeerInfo(
                node_id=temp_peer_id,
                address=address,
                port=port,
                state=ConnectionState.CONNECTING,
            )
            connection = PeerConnection(
                peer_id=temp_peer_id, peer_info=peer_info, reader=reader, writer=writer
            )

            # Perform handshake
            peer_id = await self._perform_handshake(connection, initiator=True)
            if not peer_id:
                await connection.close()
                return None

            return peer_id

        except asyncio.TimeoutError:
            logger.error(f"Connection to {address}:{port} timed out")
            return None
        except Exception as e:
            logger.error(f"Failed to connect to {address}:{port}: {e}")
            return None

    async def disconnect_peer(self, peer_id: str) -> bool:
        """Disconnect from a peer.

        Args:
            peer_id: ID of peer to disconnect

        Returns:
            True if disconnected
        """
        connection = self._peers.pop(peer_id, None)
        if connection:
            await connection.close()

            # Update peer info
            if peer_id in self._peer_info:
                self._peer_info[peer_id].state = ConnectionState.DISCONNECTED

            # Notify handlers
            for handler in self._on_peer_disconnected:
                try:
                    handler(peer_id)
                except Exception as e:
                    logger.error(f"Error in disconnect handler: {e}")

            logger.info(f"Disconnected from peer {peer_id}")
            return True
        return False

    async def send_message(
        self,
        peer_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        encrypted: bool = True,
    ) -> bool:
        """Send a message to a peer.

        Args:
            peer_id: Target peer ID
            message_type: Type of message
            payload: Message payload
            encrypted: Whether to encrypt the message

        Returns:
            True if sent successfully
        """
        connection = self._peers.get(peer_id)
        if not connection:
            logger.warning(f"Cannot send message: peer '{peer_id}' not connected")
            return False

        if connection.peer_info.state not in (
            ConnectionState.CONNECTED,
            ConnectionState.AUTHENTICATED,
        ):
            logger.warning(
                f"Cannot send message to peer '{peer_id}': connection state is "
                f"'{connection.peer_info.state.value}', expected 'connected' or 'authenticated'"
            )
            return False

        try:
            # Create protocol message
            message = FederationMessage(
                message_type=message_type,
                source_id=self.config.node_id,
                target_id=peer_id,
                payload=payload,
            )

            # Serialize
            try:
                message_data = json.dumps(message.to_dict()).encode()
            except (TypeError, ValueError) as e:
                logger.error(f"Failed to serialize message for peer {peer_id}: {e}")
                return False

            # Encrypt if required
            if encrypted and self._crypto and connection.peer_info.session_key:
                encrypted_msg = self._crypto.encrypt(message_data, connection.peer_info.session_key)
                if encrypted_msg:
                    message_data = json.dumps(
                        {
                            "encrypted": True,
                            "data": encrypted_msg.to_dict(),
                        }
                    ).encode()
                else:
                    logger.warning(
                        f"Encryption failed for message to peer {peer_id}, "
                        f"sending unencrypted (message_type={message_type.value})"
                    )

            return await connection.send(message_data)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to send message to peer '{peer_id}' "
                f"(type={message_type.value}): {type(e).__name__}: {e}"
            )
            return False

    async def send_request(
        self,
        peer_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        timeout: float = 30.0,
    ) -> Optional[FederationMessage]:
        """Send a request and wait for response.

        Args:
            peer_id: Target peer ID
            message_type: Type of message
            payload: Message payload
            timeout: Response timeout

        Returns:
            Response message or None
        """
        if not self._protocol:
            return None

        message = FederationMessage(
            message_type=message_type,
            source_id=self.config.node_id,
            target_id=peer_id,
            payload=payload,
        )

        return await self._protocol.send_request(message, timeout)

    async def request_permissions(
        self, peer_id: str, permissions: List[Permission], reason: str = ""
    ) -> Optional[PermissionGrant]:
        """Request permissions from a peer.

        Args:
            peer_id: Target peer ID
            permissions: Permissions to request
            reason: Reason for request

        Returns:
            Permission grant if approved, None otherwise
        """
        if not self._permission_manager:
            return None

        # Create request
        request = self._permission_manager.create_request(peer_id, permissions, reason)

        # Send to peer
        response = await self.send_request(
            peer_id,
            MessageType.PERMISSION_REQUEST,
            {
                "request_id": request.request_id,
                "permissions": [p.to_dict() for p in request.permissions],
                "reason": request.reason,
            },
        )

        if not response:
            return None

        # Parse response
        if response.payload.get("approved"):
            grant_data = response.payload.get("grant")
            if grant_data:
                grant = PermissionGrant.from_dict(grant_data)
                self._permission_manager.receive_grant(grant)
                return grant

        return None

    def get_peer_info(self, peer_id: str) -> Optional[PeerInfo]:
        """Get information about a peer.

        Args:
            peer_id: Peer ID

        Returns:
            Peer info or None
        """
        return self._peer_info.get(peer_id)

    def is_connected(self, peer_id: str) -> bool:
        """Check if connected to a peer.

        Args:
            peer_id: Peer ID

        Returns:
            True if connected
        """
        return peer_id in self._peers

    def has_permission(self, peer_id: str, permission: Permission) -> bool:
        """Check if we have a permission for a peer.

        Args:
            peer_id: Peer ID
            permission: Permission to check

        Returns:
            True if permission granted
        """
        if not self._permission_manager:
            return False
        return self._permission_manager.check_permission(peer_id, permission)

    # Event handler registration

    def on_peer_connected(self, handler: Callable[[str], None]):
        """Register handler for peer connection events."""
        self._on_peer_connected.append(handler)

    def on_peer_disconnected(self, handler: Callable[[str], None]):
        """Register handler for peer disconnection events."""
        self._on_peer_disconnected.append(handler)

    def on_message(self, handler: Callable[[str, FederationMessage], None]):
        """Register handler for incoming messages."""
        self._on_message.append(handler)

    def on_permission_request(self, handler: Callable[[str, PermissionRequest], None]):
        """Register handler for permission requests."""
        self._on_permission_request.append(handler)

    # Internal methods

    async def _initialize_components(self):
        """Initialize federation components."""
        from pathlib import Path

        storage_path = None
        if self.config.storage_path:
            storage_path = Path(self.config.storage_path)

        # Identity manager
        self._identity_manager = IdentityManager(
            node_id=self.config.node_id, storage_path=storage_path
        )

        # Protocol handler
        self._protocol = create_protocol(
            node_id=self.config.node_id,
            identity=self._identity_manager.identity,
        )

        # Crypto provider
        self._crypto = create_crypto_provider()

        # Permission manager
        self._permission_manager = create_permission_manager(node_id=self.config.node_id)

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming connection."""
        addr = writer.get_extra_info("peername")
        logger.info(f"Incoming connection from {addr}")

        if len(self._peers) >= self.config.max_peers:
            logger.warning(f"Rejecting connection from {addr}: max peers reached")
            writer.close()
            return

        # Create temporary connection
        temp_peer_id = f"incoming_{addr[0]}:{addr[1]}"
        peer_info = PeerInfo(
            node_id=temp_peer_id,
            address=addr[0],
            port=addr[1],
            state=ConnectionState.CONNECTING,
        )
        connection = PeerConnection(
            peer_id=temp_peer_id, peer_info=peer_info, reader=reader, writer=writer
        )

        # Perform handshake
        peer_id = await self._perform_handshake(connection, initiator=False)
        if not peer_id:
            await connection.close()
            return

    async def _perform_handshake(
        self, connection: PeerConnection, initiator: bool
    ) -> Optional[str]:
        """Perform connection handshake.

        Args:
            connection: Connection to handshake on
            initiator: Whether we initiated the connection

        Returns:
            Peer ID if successful, None otherwise
        """
        try:
            connection.peer_info.state = ConnectionState.HANDSHAKING

            if initiator:
                # Send our identity
                handshake = {
                    "type": "handshake",
                    "node_id": self.config.node_id,
                    "display_name": self.config.display_name,
                    "version": FederationProtocol.VERSION,
                    "require_encryption": self.config.require_encryption,
                }

                if self._identity_manager and self._identity_manager.identity:
                    handshake["public_key"] = (
                        self._identity_manager.identity.public_key.key_data.hex()
                    )
                    handshake["certificate"] = (
                        self._identity_manager.identity.certificate.to_dict()
                        if self._identity_manager.identity.certificate
                        else None
                    )

                await connection.send(json.dumps(handshake).encode())

                # Receive peer's handshake
                response_data = await asyncio.wait_for(
                    connection.receive(), timeout=self.config.connection_timeout
                )
                if not response_data:
                    return None

                peer_handshake = json.loads(response_data.decode())

            else:
                # Receive initiator's handshake
                handshake_data = await asyncio.wait_for(
                    connection.receive(), timeout=self.config.connection_timeout
                )
                if not handshake_data:
                    return None

                peer_handshake = json.loads(handshake_data.decode())

                # Send our response
                response = {
                    "type": "handshake_response",
                    "node_id": self.config.node_id,
                    "display_name": self.config.display_name,
                    "version": FederationProtocol.VERSION,
                    "accepted": True,
                }

                if self._identity_manager and self._identity_manager.identity:
                    response["public_key"] = (
                        self._identity_manager.identity.public_key.key_data.hex()
                    )
                    response["certificate"] = (
                        self._identity_manager.identity.certificate.to_dict()
                        if self._identity_manager.identity.certificate
                        else None
                    )

                await connection.send(json.dumps(response).encode())

            # Extract peer info
            peer_id = peer_handshake.get("node_id")
            if not peer_id:
                logger.error("Peer did not provide node_id")
                return None

            # Check if already connected
            if peer_id in self._peers:
                logger.warning(f"Already connected to {peer_id}")
                return None

            # Create peer identity
            peer_identity = None
            if "public_key" in peer_handshake:
                from .identity import KeyType, PublicKey

                peer_public_key = PublicKey(
                    key_type=KeyType.ED25519,
                    key_data=bytes.fromhex(peer_handshake["public_key"]),
                )
                peer_identity = Identity(
                    node_id=peer_id,
                    display_name=peer_handshake.get("display_name", peer_id),
                    public_key=peer_public_key,
                )

                # Verify identity if required
                if self.config.require_authentication and self._identity_manager:
                    if not self._identity_manager.verify_identity(peer_identity):
                        logger.warning(f"Failed to verify identity of {peer_id}")
                        # Don't reject - just mark as unverified
                        peer_identity.verified = False

            # Establish session key
            session_key = None
            if self.config.require_encryption and self._crypto:
                # Key exchange
                if initiator:
                    # Send key exchange init
                    key_init = self._crypto.create_key_exchange()
                    await connection.send(
                        json.dumps({"type": "key_exchange", "public_key": key_init.hex()}).encode()
                    )

                    # Receive response
                    key_resp_data = await asyncio.wait_for(
                        connection.receive(), timeout=self.config.connection_timeout
                    )
                    if key_resp_data:
                        key_resp = json.loads(key_resp_data.decode())
                        peer_key = bytes.fromhex(key_resp["public_key"])
                        session_key = self._crypto.complete_key_exchange(peer_key)
                else:
                    # Receive key exchange
                    key_init_data = await asyncio.wait_for(
                        connection.receive(), timeout=self.config.connection_timeout
                    )
                    if key_init_data:
                        key_init = json.loads(key_init_data.decode())
                        peer_key = bytes.fromhex(key_init["public_key"])

                        # Send response
                        our_key = self._crypto.create_key_exchange()
                        await connection.send(
                            json.dumps(
                                {"type": "key_exchange", "public_key": our_key.hex()}
                            ).encode()
                        )
                        session_key = self._crypto.complete_key_exchange(peer_key)

            # Update connection info
            peer_info = PeerInfo(
                node_id=peer_id,
                identity=peer_identity,
                address=connection.peer_info.address,
                port=connection.peer_info.port,
                state=ConnectionState.AUTHENTICATED if peer_identity else ConnectionState.CONNECTED,
                connected_at=datetime.now(),
                last_seen=datetime.now(),
                session_key=session_key,
            )

            # Update connection with real peer ID
            connection.peer_id = peer_id
            connection.peer_info = peer_info

            # Store connection
            self._peers[peer_id] = connection
            self._peer_info[peer_id] = peer_info

            # Start receive loop
            connection._receive_task = asyncio.create_task(self._receive_loop(connection))

            # Notify handlers
            for handler in self._on_peer_connected:
                try:
                    handler(peer_id)
                except Exception as e:
                    logger.error(f"Error in connection handler: {e}")

            logger.info(f"Connected to peer {peer_id}")
            return peer_id

        except asyncio.TimeoutError:
            logger.error(
                f"Handshake timed out after {self.config.connection_timeout}s "
                f"with peer at {connection.peer_info.address}:{connection.peer_info.port}"
            )
            return None
        except json.JSONDecodeError as e:
            logger.error(
                f"Handshake failed: invalid JSON received from peer at "
                f"{connection.peer_info.address}:{connection.peer_info.port}: {e}"
            )
            return None
        except (KeyError, ValueError) as e:
            logger.error(
                f"Handshake failed: malformed handshake data from peer at "
                f"{connection.peer_info.address}:{connection.peer_info.port}: {e}"
            )
            return None
        except ConnectionError as e:
            logger.error(
                f"Handshake failed: connection error with peer at "
                f"{connection.peer_info.address}:{connection.peer_info.port}: {e}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Handshake failed with peer at "
                f"{connection.peer_info.address}:{connection.peer_info.port}: "
                f"{type(e).__name__}: {e}"
            )
            return None

    async def _receive_loop(self, connection: PeerConnection):
        """Receive messages from a peer."""
        peer_id = connection.peer_id
        consecutive_errors = 0
        max_consecutive_errors = 5

        try:
            while True:
                data = await connection.receive()
                if not data:
                    logger.debug(f"Connection closed by peer {peer_id}")
                    break

                # Update last seen
                connection.peer_info.last_seen = datetime.now()

                # Parse message
                try:
                    msg_dict = json.loads(data.decode())
                    consecutive_errors = 0  # Reset on successful parse

                    # Check for encrypted message
                    if msg_dict.get("encrypted") and self._crypto:
                        enc_data = msg_dict.get("data")
                        if not enc_data:
                            logger.warning(
                                f"Received encrypted message from {peer_id} without data field"
                            )
                            continue
                        try:
                            enc_msg = EncryptedMessage.from_dict(enc_data)
                            decrypted = self._crypto.decrypt(
                                enc_msg, connection.peer_info.session_key
                            )
                            if decrypted:
                                msg_dict = json.loads(decrypted.decode())
                            else:
                                logger.warning(f"Failed to decrypt message from {peer_id}")
                                continue
                        except (KeyError, ValueError) as e:
                            logger.warning(f"Invalid encrypted message format from {peer_id}: {e}")
                            continue

                    # Parse as federation message
                    message = FederationMessage.from_dict(msg_dict)

                    # Handle message
                    await self._handle_message(peer_id, message)

                except json.JSONDecodeError as e:
                    consecutive_errors += 1
                    logger.warning(f"Invalid JSON received from {peer_id}: {e}")
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(
                            f"Too many consecutive parse errors from {peer_id}, disconnecting"
                        )
                        break
                except (KeyError, ValueError) as e:
                    consecutive_errors += 1
                    logger.warning(f"Malformed message from {peer_id}: {e}")
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(
                            f"Too many consecutive parse errors from {peer_id}, disconnecting"
                        )
                        break
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(
                        f"Failed to process message from {peer_id}: {type(e).__name__}: {e}"
                    )
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"Too many consecutive errors from {peer_id}, disconnecting")
                        break

        except asyncio.CancelledError:
            logger.debug(f"Receive loop cancelled for peer {peer_id}")
        except Exception as e:
            logger.error(f"Error in receive loop for peer {peer_id}: {type(e).__name__}: {e}")
        finally:
            # Disconnect
            await self.disconnect_peer(peer_id)

    async def _handle_message(self, peer_id: str, message: FederationMessage):
        """Handle incoming message."""
        # Handle protocol messages
        if self._protocol:
            response = await self._protocol.receive_message(message)
            if response:
                await self.send_message(peer_id, response.message_type, response.payload)

        # Handle permission requests
        if message.message_type == MessageType.PERMISSION_REQUEST:
            if self._permission_manager:
                request = PermissionRequest.from_dict(
                    {
                        "request_id": message.payload.get("request_id"),
                        "requester_id": peer_id,
                        "target_id": self.config.node_id,
                        "permissions": message.payload.get("permissions", []),
                        "reason": message.payload.get("reason", ""),
                    }
                )

                # Notify handlers
                for handler in self._on_permission_request:
                    try:
                        handler(peer_id, request)
                    except Exception as e:
                        logger.error(f"Error in permission request handler: {e}")

        # Notify message handlers
        for handler in self._on_message:
            try:
                handler(peer_id, message)
            except Exception as e:
                logger.error(f"Error in message handler: {e}")

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to peers."""
        while True:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)

                for peer_id in list(self._peers.keys()):
                    await self.send_message(
                        peer_id,
                        MessageType.HEARTBEAT,
                        {"timestamp": datetime.now().isoformat()},
                        encrypted=False,
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")

    async def _cleanup_loop(self):
        """Periodically cleanup stale connections."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                now = datetime.now()
                stale_peers = []

                for peer_id, info in self._peer_info.items():
                    if info.last_seen:
                        idle_time = (now - info.last_seen).total_seconds()
                        if idle_time > self.config.heartbeat_interval * 3:
                            stale_peers.append(peer_id)

                for peer_id in stale_peers:
                    logger.info(f"Disconnecting stale peer {peer_id}")
                    await self.disconnect_peer(peer_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")


def create_federation_node(
    node_id: str,
    display_name: str = "",
    listen_port: int = 8765,
    **kwargs,
) -> FederationNode:
    """Create a federation node.

    Args:
        node_id: Unique node identifier
        display_name: Human-readable name
        listen_port: Port to listen on
        **kwargs: Additional config options

    Returns:
        Configured federation node
    """
    config = NodeConfig(
        node_id=node_id,
        display_name=display_name or node_id,
        listen_port=listen_port,
        **kwargs,
    )
    return FederationNode(config)
