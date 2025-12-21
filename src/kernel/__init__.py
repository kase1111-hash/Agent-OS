"""Conversational Kernel Module for Agent OS.

Provides a constitutional layer for rule-based system governance through
natural language policies translated to system enforcement mechanisms.

Components:
- Rule Registry: Store and manage user-declared policies
- Policy Interpreter: Translate NL rules to syscall policies
- FUSE Wrapper: Filesystem-level policy enforcement
- eBPF/Seccomp: System call filtering
- inotify Hooks: File system monitoring
- Context Memory: Preferences and rule history
"""

from .rules import (
    Rule,
    RuleScope,
    RuleEffect,
    RuleAction,
    RuleRegistry,
    RuleConflict,
    RuleValidationError,
)
from .policy import (
    Policy,
    PolicyType,
    PolicyCompiler,
    SyscallPolicy,
    FilePolicy,
    AccessPolicy,
)
from .interpreter import (
    PolicyInterpreter,
    IntentParser,
    ParsedIntent,
    IntentAction,
)
from .fuse import (
    FuseWrapper,
    FuseConfig,
    FuseMount,
    FuseOperation,
)
from .ebpf import (
    EbpfFilter,
    EbpfProgram,
    EbpfProgType,
    SeccompAction,
    SeccompFilter,
    SyscallFilter,
)
from .monitor import (
    FileMonitor,
    MonitorEvent,
    EventType,
    AuditLog,
    AuditEntry,
)
from .context import (
    ContextMemory,
    UserContext,
    FolderContext,
    AgentContext,
)
from .engine import (
    ConversationalKernel,
    KernelConfig,
    KernelState,
)

__all__ = [
    # Rules
    "Rule",
    "RuleScope",
    "RuleEffect",
    "RuleAction",
    "RuleRegistry",
    "RuleConflict",
    "RuleValidationError",
    # Policy
    "Policy",
    "PolicyType",
    "PolicyCompiler",
    "SyscallPolicy",
    "FilePolicy",
    "AccessPolicy",
    # Interpreter
    "PolicyInterpreter",
    "IntentParser",
    "ParsedIntent",
    "IntentAction",
    # FUSE
    "FuseWrapper",
    "FuseConfig",
    "FuseMount",
    "FuseOperation",
    # eBPF
    "EbpfFilter",
    "EbpfProgram",
    "EbpfProgType",
    "SeccompAction",
    "SeccompFilter",
    "SyscallFilter",
    # Monitor
    "FileMonitor",
    "MonitorEvent",
    "EventType",
    "AuditLog",
    "AuditEntry",
    # Context
    "ContextMemory",
    "UserContext",
    "FolderContext",
    "AgentContext",
    # Engine
    "ConversationalKernel",
    "KernelConfig",
    "KernelState",
]
