"""
End-to-End Simulation for Agent OS

Runs a complete system simulation from setup through multiple conversations.
Tests all major components working together.
"""

import asyncio
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.parser import ConstitutionParser
from src.core.validator import ConstitutionValidator
from src.core.constitution import ConstitutionalKernel, RequestContext, create_kernel
from src.core.models import AuthorityLevel, RuleType

# Mobile components
from src.mobile import (
    MobileClient, ClientConfig, ConnectionState,
    MobileAuth, AuthConfig, AuthToken,
    VPNTunnel, VPNConfig, VPNState,
    PushNotificationService, NotificationConfig, NotificationPayload,
    OfflineStorage, StorageConfig, SyncManager, SyncState,
    MobileAPI, Platform, PlatformType, DeviceInfo,
)

# Kernel components
from src.kernel import (
    ConversationalKernel as CKernel, KernelConfig, KernelState,
    RuleRegistry, Rule, RuleScope, RuleEffect,
    PolicyInterpreter, PolicyCompiler,
)

# Federation components
from src.federation import (
    FederationNode, NodeConfig, PeerInfo, NodeState,
    FederationMessage, MessageType,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class ConversationResult:
    """Result of a single conversation."""
    conversation_id: int
    success: bool
    duration_ms: float
    messages_exchanged: int
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Result of a complete simulation run."""
    run_id: int
    success: bool
    setup_duration_ms: float
    total_duration_ms: float
    conversations: List[ConversationResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def successful_conversations(self) -> int:
        return sum(1 for c in self.conversations if c.success)

    @property
    def failed_conversations(self) -> int:
        return sum(1 for c in self.conversations if not c.success)

    @property
    def avg_conversation_duration_ms(self) -> float:
        if not self.conversations:
            return 0.0
        return sum(c.duration_ms for c in self.conversations) / len(self.conversations)


class E2ESimulation:
    """End-to-end simulation runner."""

    def __init__(self, temp_dir: str):
        self.temp_dir = Path(temp_dir)
        self.project_root = Path(__file__).parent.parent

        # Components
        self.kernel: Optional[ConstitutionalKernel] = None
        self.mobile_client: Optional[MobileClient] = None
        self.mobile_auth: Optional[MobileAuth] = None
        self.vpn_tunnel: Optional[VPNTunnel] = None
        self.storage: Optional[OfflineStorage] = None
        self.notification_service: Optional[PushNotificationService] = None
        self.federation_node: Optional[FederationNode] = None
        self.rule_registry: Optional[RuleRegistry] = None

    async def setup(self) -> Dict[str, Any]:
        """Initialize all components."""
        results = {
            "components_initialized": [],
            "errors": [],
        }

        # 1. Initialize Constitutional Kernel
        try:
            constitution_path = self.project_root / "CONSTITUTION.md"
            agents_dir = self.project_root / "agents"

            if constitution_path.exists():
                self.kernel = create_kernel(
                    self.project_root,
                    enable_hot_reload=False
                )
                results["components_initialized"].append("constitutional_kernel")
        except Exception as e:
            results["errors"].append(f"Kernel init failed: {e}")

        # 2. Initialize Mobile Client
        try:
            config = ClientConfig(
                base_url="https://api.agentos.local",
                timeout=30,
                retry_count=3,
            )
            self.mobile_client = MobileClient(config)
            await self.mobile_client.connect()
            results["components_initialized"].append("mobile_client")
        except Exception as e:
            results["errors"].append(f"Mobile client init failed: {e}")

        # 3. Initialize Mobile Auth
        try:
            auth_config = AuthConfig(
                client_id="e2e-simulation",
                biometric_enabled=True,
            )
            self.mobile_auth = MobileAuth(auth_config)
            await self.mobile_auth.register_device(
                device_id="sim-device-001",
                platform="ios",
            )
            results["components_initialized"].append("mobile_auth")
        except Exception as e:
            results["errors"].append(f"Mobile auth init failed: {e}")

        # 4. Initialize VPN Tunnel
        try:
            vpn_config = VPNConfig(
                server_address="vpn.agentos.local",
                server_public_key="simulation_key",
            )
            self.vpn_tunnel = VPNTunnel(vpn_config)
            await self.vpn_tunnel.connect()
            results["components_initialized"].append("vpn_tunnel")
        except Exception as e:
            results["errors"].append(f"VPN init failed: {e}")

        # 5. Initialize Offline Storage
        try:
            storage_config = StorageConfig(
                database_path=str(self.temp_dir / "simulation.db"),
                cache_size_mb=50,
            )
            self.storage = OfflineStorage(storage_config)
            await self.storage.initialize()
            results["components_initialized"].append("offline_storage")
        except Exception as e:
            results["errors"].append(f"Storage init failed: {e}")

        # 6. Initialize Push Notifications
        try:
            notif_config = NotificationConfig(
                apns_key_id="sim_key",
                apns_team_id="sim_team",
                apns_bundle_id="com.agentos.sim",
                fcm_project_id="sim_project",
            )
            self.notification_service = PushNotificationService(notif_config)
            self.notification_service.register_device("sim-device-001", "ios", "sim_token")
            await self.notification_service.start()
            results["components_initialized"].append("push_notifications")
        except Exception as e:
            results["errors"].append(f"Notifications init failed: {e}")

        # 7. Initialize Rule Registry
        try:
            self.rule_registry = RuleRegistry(
                db_path=str(self.temp_dir / "rules.db")
            )
            # Add some test rules
            from src.kernel.rules import RuleAction
            self.rule_registry.add_rule(Rule(
                rule_id="sim-001",
                scope=RuleScope.SYSTEM,
                target="/public/*",
                effect=RuleEffect.ALLOW,
                actions=[RuleAction.READ],
                reason="Allow read access to public data",
            ), check_conflicts=False)
            self.rule_registry.add_rule(Rule(
                rule_id="sim-002",
                scope=RuleScope.SYSTEM,
                target="/system/*",
                effect=RuleEffect.DENY,
                actions=[RuleAction.WRITE],
                reason="Deny write access to system files",
            ), check_conflicts=False)
            results["components_initialized"].append("rule_registry")
        except Exception as e:
            results["errors"].append(f"Rule registry init failed: {e}")

        # 8. Initialize Federation Node
        try:
            fed_config = NodeConfig(
                node_id="sim-node-001",
                display_name="Simulation Node",
                listen_address="localhost",
                listen_port=8765,
            )
            self.federation_node = FederationNode(fed_config)
            results["components_initialized"].append("federation_node")
        except Exception as e:
            results["errors"].append(f"Federation init failed: {e}")

        return results

    async def teardown(self) -> None:
        """Clean up all components."""
        if self.kernel:
            self.kernel.shutdown()

        if self.mobile_client:
            await self.mobile_client.disconnect()

        if self.mobile_auth:
            await self.mobile_auth.logout()

        if self.vpn_tunnel:
            await self.vpn_tunnel.disconnect()

        if self.storage:
            await self.storage.close()

        if self.notification_service:
            await self.notification_service.stop()

    async def run_conversation(self, conversation_id: int) -> ConversationResult:
        """Run a single simulated conversation."""
        start_time = time.time()
        errors = []
        messages_exchanged = 0
        metadata = {}

        try:
            # Simulate authentication
            if self.mobile_auth:
                token = await self.mobile_auth.authenticate(
                    username=f"user_{conversation_id}",
                    password="sim_password",
                )
                messages_exchanged += 1
                metadata["auth_token_expires_in"] = token.expires_in

            # Simulate API requests
            if self.mobile_client:
                # Health check
                response = await self.mobile_client.get("/health")
                messages_exchanged += 1

                # Data request
                response = await self.mobile_client.post(
                    "/api/v1/query",
                    data={"question": f"Conversation {conversation_id} query"}
                )
                messages_exchanged += 1

                # Another request
                response = await self.mobile_client.get("/api/v1/status")
                messages_exchanged += 1

            # Simulate storage operations
            if self.storage:
                await self.storage.save_entity(
                    "conversation",
                    f"conv_{conversation_id}",
                    {
                        "id": conversation_id,
                        "timestamp": datetime.now().isoformat(),
                        "status": "active",
                    }
                )
                messages_exchanged += 1

                # Read it back
                entity = await self.storage.get_entity("conversation", f"conv_{conversation_id}")
                if entity:
                    messages_exchanged += 1

            # Simulate rule enforcement
            if self.kernel:
                context = RequestContext(
                    request_id=f"conv-{conversation_id}-001",
                    source="user",
                    destination="sage",
                    intent="query.factual",
                    content=f"Query from conversation {conversation_id}",
                )
                result = self.kernel.enforce(context)
                messages_exchanged += 1
                metadata["enforcement_allowed"] = result.allowed

            # Simulate notification
            if self.notification_service:
                payload = NotificationPayload(
                    title="Conversation Update",
                    body=f"Conversation {conversation_id} completed",
                )
                await self.notification_service.send("sim-device-001", payload)
                messages_exchanged += 1

            # Check VPN status
            if self.vpn_tunnel:
                diag = self.vpn_tunnel.get_diagnostics()
                metadata["vpn_state"] = diag["state"]
                messages_exchanged += 1

            success = True

        except Exception as e:
            errors.append(str(e))
            success = False

        duration_ms = (time.time() - start_time) * 1000

        return ConversationResult(
            conversation_id=conversation_id,
            success=success,
            duration_ms=duration_ms,
            messages_exchanged=messages_exchanged,
            errors=errors,
            metadata=metadata,
        )

    async def run_simulation(self, run_id: int, num_conversations: int = 10) -> SimulationResult:
        """Run a complete simulation."""
        start_time = time.time()

        # Setup
        setup_start = time.time()
        setup_result = await self.setup()
        setup_duration_ms = (time.time() - setup_start) * 1000

        errors = setup_result.get("errors", [])
        conversations = []

        # Run conversations
        for i in range(1, num_conversations + 1):
            conv_result = await self.run_conversation(i)
            conversations.append(conv_result)

        # Teardown
        await self.teardown()

        total_duration_ms = (time.time() - start_time) * 1000

        return SimulationResult(
            run_id=run_id,
            success=len(errors) == 0 and all(c.success for c in conversations),
            setup_duration_ms=setup_duration_ms,
            total_duration_ms=total_duration_ms,
            conversations=conversations,
            errors=errors,
        )


async def run_simulations(num_runs: int = 5, conversations_per_run: int = 10):
    """Run multiple simulation runs."""
    print("=" * 70)
    print(f"Agent OS End-to-End Simulation")
    print(f"Runs: {num_runs} | Conversations per run: {conversations_per_run}")
    print("=" * 70)
    print()

    results: List[SimulationResult] = []

    for run_id in range(1, num_runs + 1):
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Run {run_id}/{num_runs}...")

            sim = E2ESimulation(temp_dir)
            result = await sim.run_simulation(run_id, conversations_per_run)
            results.append(result)

            # Print run summary
            status = "✓ PASS" if result.success else "✗ FAIL"
            print(f"  {status} | Setup: {result.setup_duration_ms:.1f}ms | "
                  f"Total: {result.total_duration_ms:.1f}ms | "
                  f"Conversations: {result.successful_conversations}/{len(result.conversations)}")

            if result.errors:
                for err in result.errors[:3]:
                    print(f"    Error: {err}")

    # Print overall summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_runs = len(results)
    successful_runs = sum(1 for r in results if r.success)
    total_conversations = sum(len(r.conversations) for r in results)
    successful_conversations = sum(r.successful_conversations for r in results)

    avg_setup = sum(r.setup_duration_ms for r in results) / len(results)
    avg_total = sum(r.total_duration_ms for r in results) / len(results)
    avg_conv = sum(r.avg_conversation_duration_ms for r in results) / len(results)

    print(f"Runs:          {successful_runs}/{total_runs} passed")
    print(f"Conversations: {successful_conversations}/{total_conversations} passed")
    print(f"Avg Setup:     {avg_setup:.1f}ms")
    print(f"Avg Total:     {avg_total:.1f}ms")
    print(f"Avg Conv:      {avg_conv:.1f}ms")
    print()

    # Detailed breakdown by run
    print("Per-Run Breakdown:")
    print("-" * 70)
    for r in results:
        status = "PASS" if r.success else "FAIL"
        print(f"  Run {r.run_id}: {status} | "
              f"{r.successful_conversations}/{len(r.conversations)} convs | "
              f"{r.total_duration_ms:.1f}ms total | "
              f"{r.avg_conversation_duration_ms:.1f}ms/conv")

    print("=" * 70)

    # Overall status
    all_passed = all(r.success for r in results)
    if all_passed:
        print("✓ ALL SIMULATIONS PASSED")
    else:
        print("✗ SOME SIMULATIONS FAILED")
    print("=" * 70)

    return all_passed, results


if __name__ == "__main__":
    success, _ = asyncio.run(run_simulations(num_runs=5, conversations_per_run=10))
    sys.exit(0 if success else 1)
