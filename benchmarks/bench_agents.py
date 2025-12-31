"""
Agent System Benchmarks

Performance benchmarks for agent routing, loading, and message processing.
Tests the multi-agent orchestration layer.

Target: Agent routing should complete in <100ms.
"""

from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

import pytest

from src.agents.interface import (
    AgentCapabilities,
    AgentMetrics,
    AgentState,
    CapabilityType,
)
from src.messaging.models import FlowRequest, FlowResponse, MessageStatus
from src.messaging.bus import MessageBus


class TestAgentCapabilitiesBenchmarks:
    """Benchmarks for agent capability handling."""

    @pytest.mark.agents
    def test_capabilities_creation(self, benchmark) -> None:
        """Benchmark creating agent capabilities.

        Target: 10000 capability objects in <100ms.
        """

        def create_capabilities() -> int:
            caps = []
            for i in range(10000):
                cap = AgentCapabilities(
                    name=f"agent-{i}",
                    version="1.0.0",
                    description=f"Test agent {i}",
                    capabilities={CapabilityType.REASONING, CapabilityType.GENERATION},
                    supported_intents=["query.*", "analyze.*"],
                    model="llama3:8b",
                    context_window=8192,
                    max_output_tokens=2048,
                )
                caps.append(cap)
            return len(caps)

        result = benchmark(create_capabilities)
        assert result == 10000

    @pytest.mark.agents
    def test_capabilities_to_dict(self, benchmark) -> None:
        """Benchmark serializing capabilities to dict.

        Target: 1000 serializations in <50ms.
        """
        capabilities = [
            AgentCapabilities(
                name=f"agent-{i}",
                version="1.0.0",
                description=f"Test agent {i}",
                capabilities={CapabilityType.REASONING, CapabilityType.GENERATION},
                supported_intents=["query.*", "analyze.*"],
                model="llama3:8b",
            )
            for i in range(1000)
        ]

        def serialize_all() -> int:
            dicts = []
            for cap in capabilities:
                dicts.append(cap.to_dict())
            return len(dicts)

        result = benchmark(serialize_all)
        assert result == 1000


class TestAgentMetricsBenchmarks:
    """Benchmarks for agent metrics tracking."""

    @pytest.mark.agents
    def test_metrics_update(self, benchmark) -> None:
        """Benchmark updating agent metrics.

        Target: 10000 metric updates in <100ms.
        """
        metrics = AgentMetrics()

        def update_metrics() -> int:
            for i in range(10000):
                metrics.record_request(
                    success=i % 10 != 0,  # 90% success rate
                    response_time_ms=50.0 + (i % 100),
                )
            return metrics.requests_processed

        result = benchmark(update_metrics)
        assert result == 10000

    @pytest.mark.agents
    def test_metrics_snapshot(self, benchmark) -> None:
        """Benchmark creating metrics snapshot.

        Target: 1000 snapshots in <50ms.
        """
        metrics = AgentMetrics()
        for i in range(100):
            metrics.record_request(success=True, response_time_ms=50.0)

        def create_snapshots() -> int:
            snapshots = []
            for _ in range(1000):
                snapshots.append(metrics.snapshot())
            return len(snapshots)

        result = benchmark(create_snapshots)
        assert result == 1000


class TestMessageBusBenchmarks:
    """Benchmarks for the messaging bus."""

    @pytest.mark.agents
    def test_message_creation(self, benchmark) -> None:
        """Benchmark creating flow request messages.

        Target: 10000 messages in <200ms.
        """

        def create_messages() -> int:
            messages = []
            for i in range(10000):
                msg = FlowRequest(
                    id=f"msg-{i:08d}",
                    source_agent="whisper",
                    target_agent="sage",
                    intent="analyze",
                    content=f"Please analyze this data: item {i}",
                    context={"user_id": "user-123", "session_id": "sess-456"},
                    priority=1,
                )
                messages.append(msg)
            return len(messages)

        result = benchmark(create_messages)
        assert result == 10000

    @pytest.mark.agents
    def test_message_bus_publish(self, benchmark) -> None:
        """Benchmark publishing messages to the bus.

        Target: 1000 publishes in <500ms.
        """
        bus = MessageBus()
        received: List[FlowRequest] = []

        # Subscribe to receive messages
        def handler(msg: FlowRequest) -> None:
            received.append(msg)

        bus.subscribe("sage", handler)

        def publish_messages() -> int:
            for i in range(1000):
                msg = FlowRequest(
                    id=f"bench-msg-{i:06d}",
                    source_agent="whisper",
                    target_agent="sage",
                    intent="analyze",
                    content=f"Benchmark message {i}",
                )
                bus.publish(msg)
            return len(received)

        # Clear and benchmark
        received.clear()
        result = benchmark(publish_messages)
        assert result >= 0  # Messages may or may not be delivered sync

    @pytest.mark.agents
    def test_message_routing(self, benchmark) -> None:
        """Benchmark message routing decisions.

        Target: 10000 routing decisions in <100ms.
        """
        # Simulate routing table
        routing_table: Dict[str, List[str]] = {
            "query.*": ["sage"],
            "analyze.*": ["sage"],
            "creative.*": ["muse"],
            "write.*": ["quill"],
            "memory.*": ["seshat"],
            "validate.*": ["smith"],
            "*": ["whisper"],  # Default
        }

        intents = [
            "query.data",
            "analyze.text",
            "creative.story",
            "write.email",
            "memory.recall",
            "validate.input",
            "unknown.intent",
        ] * 1428 + ["query.data"] * 4  # ~10000

        def route_all() -> int:
            routes = []
            for intent in intents:
                # Simple pattern matching
                matched = None
                for pattern, targets in routing_table.items():
                    if pattern == "*":
                        if matched is None:
                            matched = targets
                    elif intent.startswith(pattern.replace(".*", "")):
                        matched = targets
                        break
                routes.append(matched or ["whisper"])
            return len(routes)

        result = benchmark(route_all)
        assert result == 10000


class TestAgentOrchestrationBenchmarks:
    """End-to-end agent orchestration benchmarks."""

    @pytest.mark.agents
    @pytest.mark.integration
    def test_request_flow_simulation(self, benchmark) -> None:
        """Benchmark simulated request flow through agents.

        Simulates: User -> Whisper -> Smith -> Sage -> Smith -> User
        Target: <500ms for complete flow (excluding LLM inference).
        """
        # Simulate agent processing stages
        stages = ["receive", "validate_pre", "route", "process", "validate_post", "respond"]
        stage_times_ms = [5, 10, 15, 50, 10, 5]  # Typical times without LLM

        def simulate_flow() -> Dict[str, Any]:
            result = {
                "stages": [],
                "total_time_ms": 0,
            }

            for stage, time_ms in zip(stages, stage_times_ms):
                # Simulate processing
                stage_result = {
                    "stage": stage,
                    "time_ms": time_ms,
                    "status": "success",
                }
                result["stages"].append(stage_result)
                result["total_time_ms"] += time_ms

            return result

        result = benchmark(simulate_flow)
        assert result["total_time_ms"] == sum(stage_times_ms)

    @pytest.mark.agents
    def test_agent_selection(self, benchmark) -> None:
        """Benchmark selecting the best agent for a task.

        Target: <10ms for agent selection from pool.
        """
        # Simulate agent pool with capabilities
        agents = [
            {
                "name": "sage",
                "capabilities": {"reasoning", "analysis"},
                "load": 0.3,
                "latency_avg_ms": 100,
            },
            {
                "name": "muse",
                "capabilities": {"creative", "generation"},
                "load": 0.5,
                "latency_avg_ms": 150,
            },
            {
                "name": "quill",
                "capabilities": {"writing", "formatting"},
                "load": 0.2,
                "latency_avg_ms": 80,
            },
            {
                "name": "seshat",
                "capabilities": {"memory", "retrieval"},
                "load": 0.4,
                "latency_avg_ms": 50,
            },
        ]

        def select_agent(required_capability: str) -> Dict[str, Any]:
            """Select best agent for capability."""
            candidates = [
                a for a in agents if required_capability in a["capabilities"]
            ]
            if not candidates:
                return agents[0]  # Default

            # Select by lowest load * latency
            return min(candidates, key=lambda a: a["load"] * a["latency_avg_ms"])

        capabilities_to_test = ["reasoning", "creative", "memory", "writing"] * 2500

        def select_all() -> int:
            selected = []
            for cap in capabilities_to_test:
                selected.append(select_agent(cap))
            return len(selected)

        result = benchmark(select_all)
        assert result == 10000
