"""Policy Compiler for Conversational Kernel.

Translates validated rules into machine-level enforcement policies:
- System call policies (seccomp, eBPF)
- File system policies (FUSE, permissions)
- Access control policies (capabilities)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from .rules import Rule, RuleAction, RuleEffect, RuleScope

logger = logging.getLogger(__name__)


class PolicyType(str, Enum):
    """Types of enforcement policies."""

    SYSCALL = "syscall"  # System call filtering
    FILE = "file"  # File system operations
    ACCESS = "access"  # Access control
    NETWORK = "network"  # Network operations
    CAPABILITY = "capability"  # Linux capabilities
    RESOURCE = "resource"  # Resource limits


class SyscallAction(str, Enum):
    """Actions for syscall policies."""

    ALLOW = "allow"
    DENY = "deny"
    LOG = "log"
    TRAP = "trap"  # Signal handler
    ERRNO = "errno"  # Return specific error


@dataclass
class SyscallPolicy:
    """Policy for system call filtering."""

    policy_id: str
    syscalls: List[str]  # Syscall names or numbers
    action: SyscallAction
    errno_value: int = 0  # For ERRNO action
    conditions: Dict[str, Any] = field(default_factory=dict)
    source_rule_id: Optional[str] = None

    # Common syscall groups
    SYSCALLS_READ = ["read", "pread64", "readv", "preadv", "readlink", "readlinkat"]
    SYSCALLS_WRITE = [
        "write",
        "pwrite64",
        "writev",
        "pwritev",
        "truncate",
        "ftruncate",
    ]
    SYSCALLS_OPEN = ["open", "openat", "openat2", "creat"]
    SYSCALLS_EXEC = ["execve", "execveat"]
    SYSCALLS_NETWORK = ["socket", "connect", "bind", "listen", "accept", "sendto", "recvfrom"]

    def to_seccomp_rule(self) -> Dict[str, Any]:
        """Convert to seccomp-bpf rule format."""
        return {
            "syscalls": self.syscalls,
            "action": self.action.value.upper(),
            "args": self.conditions.get("args", []),
            "errno": self.errno_value,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "policy_id": self.policy_id,
            "syscalls": self.syscalls,
            "action": self.action.value,
            "errno_value": self.errno_value,
            "conditions": self.conditions,
            "source_rule_id": self.source_rule_id,
        }


@dataclass
class FilePolicy:
    """Policy for file system operations."""

    policy_id: str
    path: str
    operations: List[str]  # read, write, execute, etc.
    effect: str  # allow, deny, audit
    recursive: bool = True
    mode: Optional[int] = None  # File mode to set
    owner: Optional[str] = None
    group: Optional[str] = None
    acl: List[Dict[str, Any]] = field(default_factory=list)
    xattrs: Dict[str, str] = field(default_factory=dict)
    source_rule_id: Optional[str] = None

    def to_fuse_config(self) -> Dict[str, Any]:
        """Convert to FUSE configuration."""
        return {
            "path": self.path,
            "recursive": self.recursive,
            "operations": {op: self.effect for op in self.operations},
            "mode": self.mode,
            "owner": self.owner,
            "group": self.group,
        }

    def to_chmod_command(self) -> Optional[str]:
        """Generate chmod command if applicable."""
        if self.mode is None:
            return None

        flags = "-R" if self.recursive else ""
        return f"chmod {flags} {oct(self.mode)[2:]} {self.path}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "policy_id": self.policy_id,
            "path": self.path,
            "operations": self.operations,
            "effect": self.effect,
            "recursive": self.recursive,
            "mode": self.mode,
            "owner": self.owner,
            "group": self.group,
            "acl": self.acl,
            "xattrs": self.xattrs,
            "source_rule_id": self.source_rule_id,
        }


@dataclass
class AccessPolicy:
    """Policy for access control."""

    policy_id: str
    subject: str  # User, group, or agent
    subject_type: str  # user, group, agent, process
    resources: List[str]  # Paths or resource identifiers
    permissions: List[str]  # read, write, execute, admin
    effect: str  # allow, deny
    conditions: Dict[str, Any] = field(default_factory=dict)
    time_restrictions: Optional[Dict[str, str]] = None
    source_rule_id: Optional[str] = None

    def to_sudoers_entry(self) -> Optional[str]:
        """Generate sudoers-style entry if applicable."""
        if self.subject_type != "user":
            return None

        if "execute" not in self.permissions:
            return None

        commands = ", ".join(self.resources)
        if self.effect == "allow":
            return f"{self.subject} ALL = ({self.subject}) {commands}"
        return f"{self.subject} ALL = (ALL) !{commands}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "policy_id": self.policy_id,
            "subject": self.subject,
            "subject_type": self.subject_type,
            "resources": self.resources,
            "permissions": self.permissions,
            "effect": self.effect,
            "conditions": self.conditions,
            "time_restrictions": self.time_restrictions,
            "source_rule_id": self.source_rule_id,
        }


# Union type for all policies
Policy = Union[SyscallPolicy, FilePolicy, AccessPolicy]


class PolicyCompiler:
    """Compiles rules into enforcement policies.

    Handles translation from high-level rules to:
    - seccomp-bpf filters
    - FUSE interceptor configs
    - File permission changes
    - Access control entries
    """

    def __init__(self):
        """Initialize policy compiler."""
        self._policy_counter = 0
        self._action_to_syscalls: Dict[RuleAction, List[str]] = {
            RuleAction.READ: SyscallPolicy.SYSCALLS_READ + ["open", "openat"],
            RuleAction.WRITE: SyscallPolicy.SYSCALLS_WRITE + ["open", "openat"],
            RuleAction.DELETE: ["unlink", "unlinkat", "rmdir"],
            RuleAction.EXECUTE: SyscallPolicy.SYSCALLS_EXEC,
            RuleAction.CREATE: ["creat", "open", "openat", "mkdir", "mkdirat"],
            RuleAction.MODIFY: SyscallPolicy.SYSCALLS_WRITE,
            RuleAction.OVERWRITE: ["truncate", "ftruncate"],
            RuleAction.RENAME: ["rename", "renameat", "renameat2"],
            RuleAction.COPY: ["copy_file_range", "sendfile"],
            RuleAction.MOVE: ["rename", "renameat", "renameat2"],
            RuleAction.LINK: ["link", "linkat", "symlink", "symlinkat"],
            RuleAction.CHMOD: ["chmod", "fchmod", "fchmodat"],
            RuleAction.CHOWN: ["chown", "fchown", "fchownat", "lchown"],
            RuleAction.LIST: ["getdents", "getdents64"],
            RuleAction.STAT: ["stat", "fstat", "lstat", "fstatat"],
            RuleAction.NETWORK: SyscallPolicy.SYSCALLS_NETWORK,
            RuleAction.SYSCALL: [],  # Custom syscalls specified in conditions
        }

        self._action_to_file_ops: Dict[RuleAction, List[str]] = {
            RuleAction.READ: ["read"],
            RuleAction.WRITE: ["write"],
            RuleAction.DELETE: ["unlink", "rmdir"],
            RuleAction.EXECUTE: ["execute"],
            RuleAction.CREATE: ["create"],
            RuleAction.MODIFY: ["write", "truncate"],
            RuleAction.OVERWRITE: ["write", "truncate"],
            RuleAction.RENAME: ["rename"],
            RuleAction.COPY: ["read"],  # Source needs read
            RuleAction.MOVE: ["rename", "unlink"],
            RuleAction.LINK: ["link"],
            RuleAction.CHMOD: ["setattr"],
            RuleAction.CHOWN: ["setattr"],
            RuleAction.LIST: ["readdir"],
            RuleAction.STAT: ["getattr"],
        }

    def _next_policy_id(self, prefix: str = "pol") -> str:
        """Generate next policy ID."""
        self._policy_counter += 1
        return f"{prefix}_{self._policy_counter}"

    def compile_rule(self, rule: Rule) -> List[Policy]:
        """Compile a rule into enforcement policies.

        Args:
            rule: The rule to compile

        Returns:
            List of policies to enforce
        """
        policies: List[Policy] = []

        # Determine policy types based on rule scope and actions
        if rule.scope in (RuleScope.FILE, RuleScope.FOLDER):
            policies.extend(self._compile_file_policies(rule))

        if rule.scope == RuleScope.PROCESS or any(
            a in rule.actions for a in [RuleAction.SYSCALL, RuleAction.NETWORK]
        ):
            policies.extend(self._compile_syscall_policies(rule))

        if rule.scope in (RuleScope.USER, RuleScope.AGENT):
            policies.extend(self._compile_access_policies(rule))

        # System-wide rules may generate multiple policy types
        if rule.scope == RuleScope.SYSTEM:
            policies.extend(self._compile_system_policies(rule))

        return policies

    def _compile_file_policies(self, rule: Rule) -> List[FilePolicy]:
        """Compile file/folder policies."""
        policies = []

        # Map rule actions to file operations
        operations = set()
        for action in rule.actions:
            if action in self._action_to_file_ops:
                operations.update(self._action_to_file_ops[action])

        if operations:
            policy = FilePolicy(
                policy_id=self._next_policy_id("file"),
                path=rule.target,
                operations=list(operations),
                effect=rule.effect.value,
                recursive=rule.scope == RuleScope.FOLDER,
                mode=self._derive_mode(rule),
                source_rule_id=rule.rule_id,
            )
            policies.append(policy)

        return policies

    def _compile_syscall_policies(self, rule: Rule) -> List[SyscallPolicy]:
        """Compile system call policies."""
        policies = []

        # Collect relevant syscalls
        syscalls = set()
        for action in rule.actions:
            if action in self._action_to_syscalls:
                syscalls.update(self._action_to_syscalls[action])

        # Handle custom syscalls from conditions
        if "syscalls" in rule.conditions:
            syscalls.update(rule.conditions["syscalls"])

        if syscalls:
            syscall_action = self._effect_to_syscall_action(rule.effect)

            policy = SyscallPolicy(
                policy_id=self._next_policy_id("syscall"),
                syscalls=list(syscalls),
                action=syscall_action,
                errno_value=13 if rule.effect == RuleEffect.DENY else 0,  # EACCES
                conditions=self._compile_syscall_conditions(rule),
                source_rule_id=rule.rule_id,
            )
            policies.append(policy)

        return policies

    def _compile_access_policies(self, rule: Rule) -> List[AccessPolicy]:
        """Compile access control policies."""
        policies = []

        # Map actions to permissions
        permissions = set()
        for action in rule.actions:
            if action in (RuleAction.READ, RuleAction.AI_READ):
                permissions.add("read")
            elif action in (RuleAction.WRITE, RuleAction.CREATE, RuleAction.MODIFY):
                permissions.add("write")
            elif action == RuleAction.EXECUTE:
                permissions.add("execute")
            elif action == RuleAction.DELETE:
                permissions.add("delete")

        if permissions:
            subject_type = "agent" if rule.scope == RuleScope.AGENT else "user"

            policy = AccessPolicy(
                policy_id=self._next_policy_id("access"),
                subject=rule.target,
                subject_type=subject_type,
                resources=rule.conditions.get("resources", ["*"]),
                permissions=list(permissions),
                effect=rule.effect.value,
                conditions=rule.conditions,
                source_rule_id=rule.rule_id,
            )
            policies.append(policy)

        return policies

    def _compile_system_policies(self, rule: Rule) -> List[Policy]:
        """Compile system-wide policies."""
        policies: List[Policy] = []

        # System rules may need both file and syscall policies
        policies.extend(self._compile_syscall_policies(rule))

        # For AI-related actions, create access policies
        ai_actions = {RuleAction.AI_READ, RuleAction.AI_INDEX, RuleAction.AI_EMBED}
        if any(a in rule.actions for a in ai_actions):
            policy = AccessPolicy(
                policy_id=self._next_policy_id("ai_access"),
                subject="*",  # All agents
                subject_type="agent",
                resources=["*"],
                permissions=["ai_access"],
                effect=rule.effect.value,
                conditions=rule.conditions,
                source_rule_id=rule.rule_id,
            )
            policies.append(policy)

        return policies

    def _effect_to_syscall_action(self, effect: RuleEffect) -> SyscallAction:
        """Convert rule effect to syscall action."""
        mapping = {
            RuleEffect.ALLOW: SyscallAction.ALLOW,
            RuleEffect.DENY: SyscallAction.DENY,
            RuleEffect.AUDIT: SyscallAction.LOG,
            RuleEffect.PROMPT: SyscallAction.TRAP,
            RuleEffect.TRANSFORM: SyscallAction.TRAP,
        }
        return mapping.get(effect, SyscallAction.DENY)

    def _derive_mode(self, rule: Rule) -> Optional[int]:
        """Derive file mode from rule."""
        if rule.effect != RuleEffect.DENY:
            return None

        # Calculate restrictive mode based on denied actions
        mode = 0o777  # Start permissive

        if RuleAction.READ in rule.actions:
            mode &= ~0o444  # Remove read
        if RuleAction.WRITE in rule.actions:
            mode &= ~0o222  # Remove write
        if RuleAction.EXECUTE in rule.actions:
            mode &= ~0o111  # Remove execute

        return mode if mode != 0o777 else None

    def _compile_syscall_conditions(self, rule: Rule) -> Dict[str, Any]:
        """Compile syscall argument conditions."""
        conditions = {}

        # Path-based conditions
        if rule.scope in (RuleScope.FILE, RuleScope.FOLDER):
            conditions["path_prefix"] = rule.target

        # User-based conditions
        if "user" in rule.conditions:
            conditions["uid"] = rule.conditions["user"]

        # Time-based conditions
        if "time" in rule.conditions:
            conditions["time_range"] = rule.conditions["time"]

        return conditions

    def compile_rules(self, rules: List[Rule]) -> Dict[PolicyType, List[Policy]]:
        """Compile multiple rules into organized policies.

        Args:
            rules: Rules to compile

        Returns:
            Dictionary of policies by type
        """
        result: Dict[PolicyType, List[Policy]] = {
            PolicyType.SYSCALL: [],
            PolicyType.FILE: [],
            PolicyType.ACCESS: [],
        }

        for rule in rules:
            policies = self.compile_rule(rule)
            for policy in policies:
                if isinstance(policy, SyscallPolicy):
                    result[PolicyType.SYSCALL].append(policy)
                elif isinstance(policy, FilePolicy):
                    result[PolicyType.FILE].append(policy)
                elif isinstance(policy, AccessPolicy):
                    result[PolicyType.ACCESS].append(policy)

        return result

    def generate_seccomp_filter(
        self, syscall_policies: List[SyscallPolicy]
    ) -> Dict[str, Any]:
        """Generate seccomp-bpf filter configuration.

        Args:
            syscall_policies: Syscall policies to include

        Returns:
            Seccomp filter configuration
        """
        filter_config = {
            "defaultAction": "SCMP_ACT_ALLOW",
            "architectures": ["SCMP_ARCH_X86_64", "SCMP_ARCH_AARCH64"],
            "syscalls": [],
        }

        for policy in syscall_policies:
            rule = {
                "names": policy.syscalls,
                "action": f"SCMP_ACT_{policy.action.value.upper()}",
            }

            if policy.action == SyscallAction.ERRNO:
                rule["action"] = f"SCMP_ACT_ERRNO({policy.errno_value})"

            if policy.conditions.get("args"):
                rule["args"] = policy.conditions["args"]

            filter_config["syscalls"].append(rule)

        return filter_config

    def generate_fuse_config(self, file_policies: List[FilePolicy]) -> Dict[str, Any]:
        """Generate FUSE interceptor configuration.

        Args:
            file_policies: File policies to include

        Returns:
            FUSE configuration
        """
        config = {"mounts": [], "default_policy": "allow"}

        for policy in file_policies:
            mount_config = policy.to_fuse_config()
            config["mounts"].append(mount_config)

        return config

    def to_json(self, policies: Dict[PolicyType, List[Policy]]) -> str:
        """Export policies as JSON."""
        output = {}

        for policy_type, policy_list in policies.items():
            output[policy_type.value] = [p.to_dict() for p in policy_list]

        return json.dumps(output, indent=2)
