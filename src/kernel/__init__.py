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

from .context import (
    AgentContext,
    ContextMemory,
    FolderContext,
    UserContext,
)
from .ebpf import (
    EbpfFilter,
    EbpfProgram,
    EbpfProgType,
    SeccompAction,
    SeccompFilter,
    SyscallFilter,
)
from .engine import (
    ConversationalKernel,
    KernelConfig,
    KernelState,
)
from .fuse import (
    FuseConfig,
    FuseMount,
    FuseOperation,
    FuseWrapper,
)
from .interpreter import (
    IntentAction,
    IntentParser,
    ParsedIntent,
    PolicyInterpreter,
)
from .monitor import (
    AuditEntry,
    AuditLog,
    EventType,
    FileMonitor,
    MonitorEvent,
)
from .policy import (
    AccessPolicy,
    FilePolicy,
    Policy,
    PolicyCompiler,
    PolicyType,
    SyscallPolicy,
)
from .rules import (
    Rule,
    RuleAction,
    RuleConflict,
    RuleEffect,
    RuleRegistry,
    RuleScope,
    RuleValidationError,
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
