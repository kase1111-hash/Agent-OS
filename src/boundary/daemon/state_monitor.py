"""
Smith Daemon State Monitor

Monitors system state including network, hardware, and process activity.
Detects anomalies and security-relevant changes for Agent Smith's
system-level enforcement layer.

This is part of Agent Smith's internal enforcement mechanism within Agent-OS,
distinct from the external boundary-daemon project.
"""

import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class NetworkState(Enum):
    """Network connectivity states."""

    OFFLINE = auto()  # No network connectivity
    LOCAL_ONLY = auto()  # Only local/loopback connections
    ONLINE = auto()  # External network access detected


class ProcessState(Enum):
    """Process monitoring states."""

    NORMAL = auto()  # Expected processes running
    ANOMALY = auto()  # Unexpected process detected
    CRITICAL = auto()  # Critical process violation


class HardwareState(Enum):
    """Hardware state indicators."""

    NORMAL = auto()  # Hardware in expected state
    MODIFIED = auto()  # Hardware configuration changed
    TAMPERED = auto()  # Possible hardware tampering


@dataclass
class SystemState:
    """Complete system state snapshot."""

    timestamp: datetime
    network_state: NetworkState
    process_state: ProcessState
    hardware_state: HardwareState
    active_connections: int = 0
    suspicious_processes: List[str] = field(default_factory=list)
    hardware_changes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_secure(self) -> bool:
        """Check if system is in secure state."""
        return (
            self.network_state == NetworkState.OFFLINE
            and self.process_state == ProcessState.NORMAL
            and self.hardware_state == HardwareState.NORMAL
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "network_state": self.network_state.name,
            "process_state": self.process_state.name,
            "hardware_state": self.hardware_state.name,
            "active_connections": self.active_connections,
            "suspicious_processes": self.suspicious_processes,
            "hardware_changes": self.hardware_changes,
            "is_secure": self.is_secure(),
            "metadata": self.metadata,
        }


@dataclass
class NetworkConnection:
    """Represents a network connection."""

    local_addr: str
    local_port: int
    remote_addr: str
    remote_port: int
    state: str
    pid: Optional[int] = None
    process_name: Optional[str] = None


class StateMonitor:
    """
    System state monitor for the Smith Daemon.

    Part of Agent Smith's system-level security enforcement, this component
    continuously monitors:
    - Network connectivity and connections
    - Running processes
    - Hardware state changes

    Notifies registered callbacks on state changes.
    """

    # Well-known safe/expected processes
    SAFE_PROCESSES: Set[str] = {
        "python",
        "python3",
        "bash",
        "sh",
        "init",
        "systemd",
        "agent-os",
        "boundary-daemon",
    }

    # Suspicious process patterns
    SUSPICIOUS_PATTERNS: List[str] = [
        "curl",
        "wget",
        "nc",
        "netcat",
        "ncat",
        "ssh",
        "scp",
        "rsync",
        "telnet",
    ]

    # Check intervals (seconds)
    DEFAULT_POLL_INTERVAL = 1.0
    FAST_POLL_INTERVAL = 0.1

    def __init__(
        self,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        allowed_processes: Optional[Set[str]] = None,
        network_allowed: bool = False,
    ):
        """
        Initialize state monitor.

        Args:
            poll_interval: How often to poll system state
            allowed_processes: Additional allowed process names
            network_allowed: Whether network access is allowed
        """
        self.poll_interval = poll_interval
        self.allowed_processes = (allowed_processes or set()) | self.SAFE_PROCESSES
        self.network_allowed = network_allowed

        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._last_state: Optional[SystemState] = None
        self._callbacks: List[Callable[[SystemState], None]] = []
        self._state_history: List[SystemState] = []
        self._max_history = 1000

        # Baseline for change detection
        self._baseline_processes: Set[str] = set()
        self._baseline_connections: int = 0

    def start(self) -> None:
        """Start the state monitor."""
        if self._running:
            return

        self._running = True
        self._capture_baseline()

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="StateMonitor",
        )
        self._monitor_thread.start()
        logger.info("State monitor started")

    def stop(self) -> None:
        """Stop the state monitor."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
        logger.info("State monitor stopped")

    def register_callback(self, callback: Callable[[SystemState], None]) -> None:
        """Register a callback for state changes."""
        self._callbacks.append(callback)

    def get_current_state(self) -> SystemState:
        """Get current system state."""
        return self._capture_state()

    def get_state_history(self) -> List[SystemState]:
        """Get state history."""
        return list(self._state_history)

    def check_network(self) -> tuple[NetworkState, int]:
        """
        Check network state.

        Returns:
            Tuple of (NetworkState, active_connection_count)
        """
        try:
            # Check for active network connections
            connections = self._get_network_connections()
            external_connections = [
                c for c in connections if not self._is_local_address(c.remote_addr)
            ]

            if external_connections:
                return NetworkState.ONLINE, len(external_connections)
            elif connections:
                return NetworkState.LOCAL_ONLY, len(connections)
            else:
                return NetworkState.OFFLINE, 0

        except Exception as e:
            logger.warning(f"Error checking network: {e}")
            return NetworkState.OFFLINE, 0

    def check_processes(self) -> tuple[ProcessState, List[str]]:
        """
        Check running processes for anomalies.

        Returns:
            Tuple of (ProcessState, list_of_suspicious_processes)
        """
        suspicious = []

        try:
            current_processes = self._get_running_processes()

            for proc_name in current_processes:
                # Check against suspicious patterns
                for pattern in self.SUSPICIOUS_PATTERNS:
                    if pattern in proc_name.lower():
                        if proc_name not in self.allowed_processes:
                            suspicious.append(proc_name)
                            break

                # Check for new unexpected processes
                if proc_name not in self._baseline_processes:
                    if proc_name not in self.allowed_processes:
                        if proc_name not in suspicious:
                            suspicious.append(f"new:{proc_name}")

            if suspicious:
                return ProcessState.ANOMALY, suspicious
            return ProcessState.NORMAL, []

        except Exception as e:
            logger.warning(f"Error checking processes: {e}")
            return ProcessState.NORMAL, []

    def check_hardware(self) -> tuple[HardwareState, List[str]]:
        """
        Check hardware state for modifications.

        Returns:
            Tuple of (HardwareState, list_of_changes)
        """
        changes = []

        try:
            # Check CPU count (simple hardware fingerprint)
            cpu_count = os.cpu_count() or 0

            # Check memory info
            try:
                with open("/proc/meminfo", "r") as f:
                    meminfo = f.read()
                    # Just check it's readable
            except (FileNotFoundError, PermissionError):
                # Expected on non-Linux systems or containerized environments
                pass

            # In a real implementation, this would check:
            # - TPM state
            # - Device fingerprints
            # - USB device changes
            # For now, return normal state

            if changes:
                return HardwareState.MODIFIED, changes
            return HardwareState.NORMAL, []

        except Exception as e:
            logger.warning(f"Error checking hardware: {e}")
            return HardwareState.NORMAL, []

    def _capture_state(self) -> SystemState:
        """Capture current system state."""
        network_state, connections = self.check_network()
        process_state, suspicious = self.check_processes()
        hardware_state, hw_changes = self.check_hardware()

        return SystemState(
            timestamp=datetime.now(),
            network_state=network_state,
            process_state=process_state,
            hardware_state=hardware_state,
            active_connections=connections,
            suspicious_processes=suspicious,
            hardware_changes=hw_changes,
            metadata={
                "poll_interval": self.poll_interval,
                "network_allowed": self.network_allowed,
            },
        )

    def _capture_baseline(self) -> None:
        """Capture baseline state for comparison."""
        try:
            self._baseline_processes = self._get_running_processes()
            _, self._baseline_connections = self.check_network()
            logger.info(f"Baseline captured: {len(self._baseline_processes)} processes")
        except Exception as e:
            logger.warning(f"Error capturing baseline: {e}")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                state = self._capture_state()

                # Check for state changes
                if self._state_changed(state):
                    self._notify_callbacks(state)

                # Store in history
                self._state_history.append(state)
                if len(self._state_history) > self._max_history:
                    self._state_history = self._state_history[-self._max_history :]

                self._last_state = state

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

            time.sleep(self.poll_interval)

    def _state_changed(self, new_state: SystemState) -> bool:
        """Check if state has changed significantly."""
        if self._last_state is None:
            return True

        return (
            new_state.network_state != self._last_state.network_state
            or new_state.process_state != self._last_state.process_state
            or new_state.hardware_state != self._last_state.hardware_state
            or new_state.suspicious_processes != self._last_state.suspicious_processes
        )

    def _notify_callbacks(self, state: SystemState) -> None:
        """Notify all registered callbacks of state change."""
        for callback in self._callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _get_network_connections(self) -> List[NetworkConnection]:
        """Get active network connections."""
        connections = []

        try:
            # Try /proc/net/tcp for Linux
            tcp_path = Path("/proc/net/tcp")
            if tcp_path.exists():
                with open(tcp_path, "r") as f:
                    lines = f.readlines()[1:]  # Skip header

                for line in lines:
                    parts = line.split()
                    if len(parts) >= 4:
                        local = parts[1]
                        remote = parts[2]
                        state = parts[3]

                        # Parse addresses
                        local_addr, local_port = self._parse_hex_address(local)
                        remote_addr, remote_port = self._parse_hex_address(remote)

                        if remote_addr != "0.0.0.0":  # Has remote connection
                            connections.append(
                                NetworkConnection(
                                    local_addr=local_addr,
                                    local_port=local_port,
                                    remote_addr=remote_addr,
                                    remote_port=remote_port,
                                    state=state,
                                )
                            )

        except (FileNotFoundError, PermissionError):
            # Fallback: try socket test
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.1)
                result = s.connect_ex(("8.8.8.8", 53))
                s.close()
                if result == 0:
                    connections.append(
                        NetworkConnection(
                            local_addr="0.0.0.0",
                            local_port=0,
                            remote_addr="8.8.8.8",
                            remote_port=53,
                            state="ESTABLISHED",
                        )
                    )
            except Exception as e:
                # Network check fallback failed; no connectivity or blocked
                logger.debug(f"Network connectivity check failed: {e}")

        return connections

    def _get_running_processes(self) -> Set[str]:
        """Get set of running process names."""
        processes = set()

        try:
            # Try /proc for Linux
            proc_path = Path("/proc")
            if proc_path.exists():
                for pid_dir in proc_path.iterdir():
                    if pid_dir.name.isdigit():
                        try:
                            comm_path = pid_dir / "comm"
                            if comm_path.exists():
                                processes.add(comm_path.read_text().strip())
                        except (PermissionError, FileNotFoundError):
                            continue

        except Exception as e:
            logger.warning(f"Error getting processes: {e}")

        return processes

    def _parse_hex_address(self, hex_addr: str) -> tuple[str, int]:
        """Parse hex address from /proc/net/tcp format."""
        try:
            addr_hex, port_hex = hex_addr.split(":")
            # Address is in little-endian format
            addr_int = int(addr_hex, 16)
            addr = socket.inet_ntoa(addr_int.to_bytes(4, "little"))
            port = int(port_hex, 16)
            return addr, port
        except Exception:
            return "0.0.0.0", 0

    def _is_local_address(self, addr: str) -> bool:
        """Check if address is local/loopback."""
        local_prefixes = ["127.", "0.0.0.0", "::1", "localhost"]
        return any(addr.startswith(prefix) for prefix in local_prefixes)


def create_state_monitor(
    poll_interval: float = 1.0,
    network_allowed: bool = False,
) -> StateMonitor:
    """Factory function to create a state monitor."""
    return StateMonitor(
        poll_interval=poll_interval,
        network_allowed=network_allowed,
    )
