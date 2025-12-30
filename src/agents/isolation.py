"""
Agent OS Process Isolation

Provides process isolation for agents using:
- Subprocess-based isolation
- Resource limits (memory, CPU)
- Container support (Docker/Podman)

Ensures agents run in controlled, sandboxed environments.
"""

import os
import re
import sys
import json
import signal
import subprocess
import multiprocessing
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Set
from enum import Enum, auto
import logging


logger = logging.getLogger(__name__)


# =============================================================================
# Security: Module and Class Name Validation
# =============================================================================


class ModuleValidationError(Exception):
    """Raised when a module or class name fails validation."""
    pass


# Allowlist of trusted module prefixes for agent isolation
# Only modules starting with these prefixes are allowed to be loaded
TRUSTED_MODULE_PREFIXES: Set[str] = {
    "src.agents.",           # Agent OS built-in agents
    "agents.",               # Alternative agent path
    "agent_os.agents.",      # Installed package agents
}


def validate_module_name(module: str) -> str:
    """
    Validate a Python module name to prevent code injection.

    This ensures the module name:
    1. Is a valid Python module path (alphanumeric, dots, underscores)
    2. Does not contain dangerous patterns
    3. Is in the trusted modules allowlist

    Args:
        module: Module name to validate (e.g., "src.agents.sage.agent")

    Returns:
        The validated module name

    Raises:
        ModuleValidationError: If the module name is invalid or not trusted
    """
    if not module:
        raise ModuleValidationError("Module name cannot be empty")

    # Check for valid Python module path pattern
    # Valid: alphanumeric, underscores, dots (for submodules)
    # Each component must start with letter or underscore, not digit
    module_pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$"

    if not re.match(module_pattern, module):
        raise ModuleValidationError(
            f"Invalid module name format: {module}. "
            "Must be a valid Python module path (e.g., 'src.agents.sage')"
        )

    # Check for dangerous patterns that could indicate injection attempts
    dangerous_patterns = [
        r"__",           # Dunder attributes
        r"\bos\b",       # os module
        r"\bsys\b",      # sys module
        r"\bsubprocess\b",  # subprocess module
        r"\beval\b",     # eval function
        r"\bexec\b",     # exec function
        r"\bimport\b",   # import statement
        r"\bopen\b",     # open function
        r"\bfile\b",     # file operations
        r"\bsocket\b",   # network operations
        r"\bbuiltins\b", # builtins access
    ]

    module_lower = module.lower()
    for pattern in dangerous_patterns:
        if re.search(pattern, module_lower):
            raise ModuleValidationError(
                f"Module name contains dangerous pattern: {module}"
            )

    # Check against allowlist of trusted module prefixes
    is_trusted = any(module.startswith(prefix) for prefix in TRUSTED_MODULE_PREFIXES)

    if not is_trusted:
        raise ModuleValidationError(
            f"Module '{module}' is not in the trusted modules allowlist. "
            f"Allowed prefixes: {', '.join(sorted(TRUSTED_MODULE_PREFIXES))}"
        )

    return module


def validate_class_name(class_name: str) -> str:
    """
    Validate a Python class name to prevent code injection.

    This ensures the class name:
    1. Is a valid Python identifier
    2. Does not contain dangerous patterns
    3. Follows PEP 8 naming conventions (starts with uppercase)

    Args:
        class_name: Class name to validate (e.g., "SageAgent")

    Returns:
        The validated class name

    Raises:
        ModuleValidationError: If the class name is invalid
    """
    if not class_name:
        raise ModuleValidationError("Class name cannot be empty")

    # Check for valid Python identifier pattern
    # Must start with letter or underscore, followed by alphanumeric/underscores
    identifier_pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"

    if not re.match(identifier_pattern, class_name):
        raise ModuleValidationError(
            f"Invalid class name format: {class_name}. "
            "Must be a valid Python identifier (e.g., 'MyAgent')"
        )

    # Check length limits (reasonable class names shouldn't be too long)
    if len(class_name) > 100:
        raise ModuleValidationError(
            f"Class name too long: {len(class_name)} characters (max 100)"
        )

    # Check for dangerous patterns
    dangerous_patterns = [
        r"^__",          # Starts with dunder
        r"__$",          # Ends with dunder
        r"\bexec\b",     # exec
        r"\beval\b",     # eval
        r"\bimport\b",   # import
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, class_name, re.IGNORECASE):
            raise ModuleValidationError(
                f"Class name contains dangerous pattern: {class_name}"
            )

    # Warn if doesn't follow PEP 8 (should start with uppercase for classes)
    if not class_name[0].isupper():
        logger.warning(
            f"Class name '{class_name}' does not follow PEP 8 naming convention "
            "(should start with uppercase letter)"
        )

    return class_name


def add_trusted_module_prefix(prefix: str) -> None:
    """
    Add a module prefix to the trusted allowlist.

    This should only be called during application initialization
    with trusted module prefixes.

    Args:
        prefix: Module prefix to trust (e.g., "myapp.agents.")
    """
    if not prefix.endswith("."):
        prefix = prefix + "."
    TRUSTED_MODULE_PREFIXES.add(prefix)
    logger.info(f"Added trusted module prefix: {prefix}")


class IsolationLevel(Enum):
    """Process isolation levels."""
    NONE = auto()       # No isolation (in-process)
    THREAD = auto()     # Thread isolation (same process)
    PROCESS = auto()    # Subprocess isolation
    CONTAINER = auto()  # Container isolation (Docker/Podman)


@dataclass
class ResourceLimits:
    """Resource limits for isolated agents."""
    memory_mb: int = 512           # Max memory in MB
    cpu_percent: int = 50          # Max CPU percentage
    timeout_seconds: int = 300     # Max execution time
    max_file_descriptors: int = 256
    max_processes: int = 10
    network_access: bool = False   # Allow network access
    filesystem_read_only: bool = True  # Read-only filesystem


@dataclass
class IsolatedProcessInfo:
    """Information about an isolated process."""
    pid: int
    agent_name: str
    started_at: datetime
    isolation_level: IsolationLevel
    resource_limits: ResourceLimits
    status: str = "running"
    exit_code: Optional[int] = None


class ProcessIsolator:
    """
    Runs agents in isolated subprocess environments.

    Provides:
    - Subprocess spawning with resource limits
    - Communication via pipes
    - Timeout and resource monitoring
    - Graceful shutdown
    """

    def __init__(self, python_path: Optional[str] = None):
        """
        Initialize isolator.

        Args:
            python_path: Path to Python interpreter
        """
        self.python_path = python_path or sys.executable
        self._processes: Dict[str, IsolatedProcessInfo] = {}
        self._lock = threading.Lock()

    def spawn(
        self,
        agent_name: str,
        agent_module: str,
        agent_class: str,
        config: Dict[str, Any],
        limits: Optional[ResourceLimits] = None,
    ) -> IsolatedProcessInfo:
        """
        Spawn an agent in an isolated subprocess.

        Args:
            agent_name: Agent name
            agent_module: Python module containing agent (must be in trusted allowlist)
            agent_class: Agent class name (must be valid Python identifier)
            config: Agent configuration
            limits: Resource limits

        Returns:
            IsolatedProcessInfo for the spawned process

        Raises:
            ModuleValidationError: If module or class name is invalid/untrusted
        """
        limits = limits or ResourceLimits()

        # SECURITY: Validate module and class names to prevent code injection
        # This is critical because they are interpolated into executable Python code
        validated_module = validate_module_name(agent_module)
        validated_class = validate_class_name(agent_class)

        logger.debug(
            f"Spawning agent '{agent_name}' from validated module "
            f"'{validated_module}' class '{validated_class}'"
        )

        # Build the wrapper script with validated inputs
        wrapper_code = self._generate_wrapper(
            validated_module,
            validated_class,
            config,
        )

        # Create subprocess with limited resources
        env = os.environ.copy()
        env["AGENT_OS_ISOLATED"] = "1"
        env["AGENT_NAME"] = agent_name

        # Start subprocess
        process = subprocess.Popen(
            [self.python_path, "-c", wrapper_code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,  # New process group
        )

        info = IsolatedProcessInfo(
            pid=process.pid,
            agent_name=agent_name,
            started_at=datetime.now(),
            isolation_level=IsolationLevel.PROCESS,
            resource_limits=limits,
        )

        with self._lock:
            self._processes[agent_name] = info

        # Start resource monitor
        self._start_monitor(agent_name, process, limits)

        logger.info(f"Spawned isolated process for {agent_name}: PID {process.pid}")
        return info

    def send_request(
        self,
        agent_name: str,
        request: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Send a request to an isolated agent.

        Args:
            agent_name: Agent name
            request: Request data
            timeout: Request timeout

        Returns:
            Response data
        """
        # In a real implementation, this would use IPC
        # For now, return a placeholder
        return {
            "status": "error",
            "message": "IPC not implemented in basic isolation",
        }

    def terminate(self, agent_name: str, graceful: bool = True) -> bool:
        """
        Terminate an isolated agent process.

        Args:
            agent_name: Agent name
            graceful: If True, send SIGTERM first

        Returns:
            True if terminated
        """
        with self._lock:
            info = self._processes.get(agent_name)
            if not info:
                return False

            try:
                if graceful:
                    os.kill(info.pid, signal.SIGTERM)
                    # Wait briefly for graceful shutdown
                    time.sleep(1)

                # Force kill if still running
                try:
                    os.kill(info.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Already dead

                info.status = "terminated"
                logger.info(f"Terminated isolated process: {agent_name}")
                return True

            except Exception as e:
                logger.error(f"Failed to terminate {agent_name}: {e}")
                return False

    def get_process_info(self, agent_name: str) -> Optional[IsolatedProcessInfo]:
        """Get information about an isolated process."""
        with self._lock:
            return self._processes.get(agent_name)

    def get_all_processes(self) -> List[IsolatedProcessInfo]:
        """Get all isolated processes."""
        with self._lock:
            return list(self._processes.values())

    def cleanup(self) -> int:
        """
        Terminate all isolated processes.

        Returns:
            Number of processes terminated
        """
        count = 0
        for name in list(self._processes.keys()):
            if self.terminate(name):
                count += 1
        return count

    def _generate_wrapper(
        self,
        module: str,
        class_name: str,
        config: Dict[str, Any],
    ) -> str:
        """Generate wrapper script for isolated execution."""
        config_json = json.dumps(config)
        return f'''
import sys
import json
import signal
import resource

# Set resource limits
try:
    # Memory limit
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, ({config.get('memory_mb', 512)} * 1024 * 1024, hard))
except Exception:
    pass

# Import and run agent
from {module} import {class_name}

config = json.loads('{config_json}')
agent = {class_name}(config.get('name', 'agent'))

def handle_signal(signum, frame):
    agent.shutdown()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)

if agent.initialize(config):
    print("AGENT_READY", flush=True)
    # Wait for input
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            request = json.loads(line)
            # Process request...
        except Exception as e:
            print(json.dumps({{"error": str(e)}}), flush=True)
else:
    print("AGENT_INIT_FAILED", flush=True)
    sys.exit(1)
'''

    def _start_monitor(
        self,
        agent_name: str,
        process: subprocess.Popen,
        limits: ResourceLimits,
    ) -> None:
        """Start monitoring thread for resource enforcement."""
        def monitor():
            start_time = time.time()
            while process.poll() is None:
                elapsed = time.time() - start_time
                if elapsed > limits.timeout_seconds:
                    logger.warning(f"Agent {agent_name} exceeded timeout, terminating")
                    self.terminate(agent_name, graceful=False)
                    break
                time.sleep(1)

            # Update status
            with self._lock:
                info = self._processes.get(agent_name)
                if info:
                    info.status = "exited"
                    info.exit_code = process.returncode

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()


class ThreadIsolator:
    """
    Runs agents in isolated threads with resource monitoring.

    Lighter weight than process isolation but less secure.
    """

    def __init__(self):
        self._threads: Dict[str, threading.Thread] = {}
        self._stop_events: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def spawn(
        self,
        agent_name: str,
        agent_instance: Any,
        config: Dict[str, Any],
    ) -> bool:
        """
        Run an agent in an isolated thread.

        Args:
            agent_name: Agent name
            agent_instance: Agent instance
            config: Agent configuration

        Returns:
            True if started successfully
        """
        stop_event = threading.Event()

        def run_agent():
            try:
                if agent_instance.initialize(config):
                    logger.info(f"Thread agent {agent_name} initialized")
                    # Wait for stop signal
                    stop_event.wait()
                    agent_instance.shutdown()
            except Exception as e:
                logger.error(f"Thread agent {agent_name} error: {e}")

        thread = threading.Thread(target=run_agent, name=f"agent-{agent_name}")
        thread.daemon = True

        with self._lock:
            self._threads[agent_name] = thread
            self._stop_events[agent_name] = stop_event

        thread.start()
        return True

    def stop(self, agent_name: str, timeout: float = 5.0) -> bool:
        """Stop a thread-isolated agent."""
        with self._lock:
            stop_event = self._stop_events.get(agent_name)
            thread = self._threads.get(agent_name)

        if not stop_event or not thread:
            return False

        stop_event.set()
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def stop_all(self, timeout: float = 5.0) -> int:
        """Stop all thread-isolated agents."""
        count = 0
        for name in list(self._threads.keys()):
            if self.stop(name, timeout):
                count += 1
        return count


class ContainerIsolator:
    """
    Runs agents in Docker/Podman containers for maximum isolation.

    Provides:
    - Full filesystem isolation
    - Network isolation
    - Resource limits via cgroups
    - Security via container runtime
    """

    def __init__(
        self,
        runtime: str = "docker",
        base_image: str = "python:3.11-slim",
    ):
        """
        Initialize container isolator.

        Args:
            runtime: Container runtime ("docker" or "podman")
            base_image: Base container image
        """
        self.runtime = runtime
        self.base_image = base_image
        self._containers: Dict[str, str] = {}  # agent_name -> container_id
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        """Check if container runtime is available."""
        try:
            result = subprocess.run(
                [self.runtime, "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def spawn(
        self,
        agent_name: str,
        agent_code_path: Path,
        config: Dict[str, Any],
        limits: Optional[ResourceLimits] = None,
    ) -> Optional[str]:
        """
        Spawn an agent in a container.

        Args:
            agent_name: Agent name
            agent_code_path: Path to agent code
            config: Agent configuration
            limits: Resource limits

        Returns:
            Container ID if successful
        """
        limits = limits or ResourceLimits()

        # Build container command
        cmd = [
            self.runtime, "run",
            "-d",  # Detached
            "--name", f"agent-{agent_name}",
            "--memory", f"{limits.memory_mb}m",
            "--cpus", str(limits.cpu_percent / 100),
        ]

        if not limits.network_access:
            cmd.extend(["--network", "none"])

        if limits.filesystem_read_only:
            cmd.append("--read-only")

        # Mount agent code
        cmd.extend(["-v", f"{agent_code_path}:/agent:ro"])

        # Add base image and entrypoint
        cmd.extend([
            self.base_image,
            "python", "/agent/main.py",
        ])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                container_id = result.stdout.strip()
                with self._lock:
                    self._containers[agent_name] = container_id
                logger.info(f"Started container for {agent_name}: {container_id[:12]}")
                return container_id
            else:
                logger.error(f"Container start failed: {result.stderr}")
                return None

        except Exception as e:
            logger.error(f"Failed to start container: {e}")
            return None

    def stop(self, agent_name: str, timeout: int = 10) -> bool:
        """Stop a container."""
        with self._lock:
            container_id = self._containers.get(agent_name)

        if not container_id:
            return False

        try:
            subprocess.run(
                [self.runtime, "stop", "-t", str(timeout), container_id],
                capture_output=True,
                timeout=timeout + 5,
            )
            subprocess.run(
                [self.runtime, "rm", container_id],
                capture_output=True,
            )

            with self._lock:
                del self._containers[agent_name]

            return True

        except Exception as e:
            logger.error(f"Failed to stop container: {e}")
            return False

    def stop_all(self) -> int:
        """Stop all containers."""
        count = 0
        for name in list(self._containers.keys()):
            if self.stop(name):
                count += 1
        return count


def create_isolator(
    level: IsolationLevel,
    **kwargs,
) -> Any:
    """
    Create an isolator for the specified level.

    Args:
        level: Isolation level
        **kwargs: Additional arguments for the isolator

    Returns:
        Isolator instance
    """
    if level == IsolationLevel.NONE:
        return None
    elif level == IsolationLevel.THREAD:
        return ThreadIsolator()
    elif level == IsolationLevel.PROCESS:
        return ProcessIsolator(**kwargs)
    elif level == IsolationLevel.CONTAINER:
        return ContainerIsolator(**kwargs)
    else:
        raise ValueError(f"Unknown isolation level: {level}")
