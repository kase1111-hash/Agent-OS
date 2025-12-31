"""eBPF and Seccomp Filters for Conversational Kernel.

Provides system call filtering using:
- Seccomp-BPF for process sandboxing
- eBPF for dynamic tracing and filtering
"""

import json
import logging
import os
import struct
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class SeccompAction(str, Enum):
    """Seccomp filter actions."""

    ALLOW = "allow"
    DENY = "deny"  # Alias for ERRNO with EACCES
    KILL = "kill"
    KILL_PROCESS = "kill_process"
    TRAP = "trap"
    ERRNO = "errno"
    TRACE = "trace"
    LOG = "log"
    USER_NOTIF = "user_notif"


class EbpfProgType(str, Enum):
    """eBPF program types."""

    SOCKET_FILTER = "socket_filter"
    KPROBE = "kprobe"
    TRACEPOINT = "tracepoint"
    XDP = "xdp"
    CGROUP_SKB = "cgroup_skb"
    CGROUP_SOCK = "cgroup_sock"
    LSM = "lsm"
    RAW_TRACEPOINT = "raw_tracepoint"


# Syscall numbers for x86_64 (subset of common syscalls)
SYSCALL_NUMBERS: Dict[str, int] = {
    "read": 0,
    "write": 1,
    "open": 2,
    "close": 3,
    "stat": 4,
    "fstat": 5,
    "lstat": 6,
    "poll": 7,
    "lseek": 8,
    "mmap": 9,
    "mprotect": 10,
    "munmap": 11,
    "brk": 12,
    "ioctl": 16,
    "access": 21,
    "pipe": 22,
    "dup": 32,
    "dup2": 33,
    "nanosleep": 35,
    "getpid": 39,
    "socket": 41,
    "connect": 42,
    "accept": 43,
    "sendto": 44,
    "recvfrom": 45,
    "bind": 49,
    "listen": 50,
    "clone": 56,
    "fork": 57,
    "vfork": 58,
    "execve": 59,
    "exit": 60,
    "wait4": 61,
    "kill": 62,
    "uname": 63,
    "fcntl": 72,
    "flock": 73,
    "fsync": 74,
    "ftruncate": 77,
    "getdents": 78,
    "getcwd": 79,
    "chdir": 80,
    "fchdir": 81,
    "rename": 82,
    "mkdir": 83,
    "rmdir": 84,
    "creat": 85,
    "link": 86,
    "unlink": 87,
    "symlink": 88,
    "readlink": 89,
    "chmod": 90,
    "fchmod": 91,
    "chown": 92,
    "fchown": 93,
    "lchown": 94,
    "umask": 95,
    "getuid": 102,
    "getgid": 104,
    "setuid": 105,
    "setgid": 106,
    "geteuid": 107,
    "getegid": 108,
    "setpgid": 109,
    "getppid": 110,
    "getpgrp": 111,
    "setsid": 112,
    "setreuid": 113,
    "setregid": 114,
    "getgroups": 115,
    "setgroups": 116,
    "prctl": 157,
    "openat": 257,
    "mkdirat": 258,
    "fchownat": 260,
    "unlinkat": 263,
    "renameat": 264,
    "linkat": 265,
    "symlinkat": 266,
    "readlinkat": 267,
    "fchmodat": 268,
    "faccessat": 269,
    "getdents64": 217,
    "seccomp": 317,
    "getrandom": 318,
    "memfd_create": 319,
    "execveat": 322,
    "copy_file_range": 326,
    # Dangerous syscalls (commonly blocked)
    "ptrace": 101,
    "mount": 165,
    "umount": 166,
    "umount2": 166,
    "pivot_root": 155,
    "chroot": 161,
    "reboot": 169,
    "sethostname": 170,
    "setdomainname": 171,
    "init_module": 175,
    "delete_module": 176,
    "kexec_load": 246,
}


@dataclass
class SyscallArg:
    """Argument constraint for syscall filtering."""

    index: int  # Argument index (0-5)
    value: int  # Expected value
    op: str = "eq"  # Comparison: eq, ne, lt, le, gt, ge, masked_eq
    mask: int = 0xFFFFFFFFFFFFFFFF  # For masked comparisons

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "value": self.value,
            "op": self.op,
            "mask": self.mask,
        }


@dataclass
class SyscallFilter:
    """Filter for a specific syscall."""

    syscall: str
    action: SeccompAction = SeccompAction.ALLOW
    args: List[SyscallArg] = field(default_factory=list)
    errno_value: int = 0  # For ERRNO action
    priority: int = 0

    @property
    def syscall_nr(self) -> int:
        """Get syscall number."""
        return SYSCALL_NUMBERS.get(self.syscall, -1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "syscall": self.syscall,
            "syscall_nr": self.syscall_nr,
            "action": self.action.value,
            "args": [a.to_dict() for a in self.args],
            "errno_value": self.errno_value,
            "priority": self.priority,
        }


@dataclass
class SeccompFilter:
    """Seccomp-BPF filter configuration."""

    filter_id: str
    default_action: SeccompAction = SeccompAction.ALLOW
    syscalls: List[SyscallFilter] = field(default_factory=list)
    architectures: List[str] = field(default_factory=lambda: ["x86_64"])
    log_violations: bool = True
    created_at: datetime = field(default_factory=datetime.now)

    def add_syscall(
        self,
        syscall: str,
        action: SeccompAction = SeccompAction.ERRNO,
        args: Optional[List[SyscallArg]] = None,
        errno_value: int = 13,  # EACCES
    ) -> None:
        """Add a syscall filter rule."""
        self.syscalls.append(
            SyscallFilter(
                syscall=syscall,
                action=action,
                args=args or [],
                errno_value=errno_value,
            )
        )

    def remove_syscall(self, syscall: str) -> bool:
        """Remove a syscall filter rule."""
        for i, sf in enumerate(self.syscalls):
            if sf.syscall == syscall:
                del self.syscalls[i]
                return True
        return False

    def to_json(self) -> str:
        """Export filter as JSON."""
        return json.dumps(
            {
                "filter_id": self.filter_id,
                "default_action": self.default_action.value,
                "syscalls": [s.to_dict() for s in self.syscalls],
                "architectures": self.architectures,
                "log_violations": self.log_violations,
            },
            indent=2,
        )

    def to_libseccomp_format(self) -> Dict[str, Any]:
        """Export in libseccomp-compatible format."""
        action_map = {
            SeccompAction.ALLOW: "SCMP_ACT_ALLOW",
            SeccompAction.DENY: "SCMP_ACT_ERRNO",  # DENY maps to ERRNO with EACCES
            SeccompAction.KILL: "SCMP_ACT_KILL",
            SeccompAction.KILL_PROCESS: "SCMP_ACT_KILL_PROCESS",
            SeccompAction.TRAP: "SCMP_ACT_TRAP",
            SeccompAction.ERRNO: "SCMP_ACT_ERRNO",
            SeccompAction.TRACE: "SCMP_ACT_TRACE",
            SeccompAction.LOG: "SCMP_ACT_LOG",
        }

        return {
            "defaultAction": action_map.get(self.default_action, "SCMP_ACT_ALLOW"),
            "architectures": [f"SCMP_ARCH_{a.upper()}" for a in self.architectures],
            "syscalls": [
                {
                    "names": [sf.syscall],
                    "action": action_map.get(sf.action, "SCMP_ACT_ERRNO")
                    + (
                        f"({sf.errno_value})"
                        if sf.action in (SeccompAction.ERRNO, SeccompAction.DENY)
                        else ""
                    ),
                    "args": [
                        {
                            "index": arg.index,
                            "value": arg.value,
                            "op": f"SCMP_CMP_{arg.op.upper()}",
                        }
                        for arg in sf.args
                    ],
                }
                for sf in self.syscalls
            ],
        }


@dataclass
class EbpfProgram:
    """eBPF program definition."""

    program_id: str
    prog_type: EbpfProgType
    attach_point: str  # e.g., "syscalls/sys_enter_openat"
    bytecode: Optional[bytes] = None  # Compiled eBPF bytecode
    source: Optional[str] = None  # eBPF C source
    maps: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    license: str = "GPL"
    loaded: bool = False
    fd: int = -1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "program_id": self.program_id,
            "prog_type": self.prog_type.value,
            "attach_point": self.attach_point,
            "has_bytecode": self.bytecode is not None,
            "has_source": self.source is not None,
            "maps": list(self.maps.keys()),
            "loaded": self.loaded,
        }


class EbpfFilter:
    """eBPF-based syscall filter and tracer.

    Provides more flexible filtering than seccomp-BPF,
    with the ability to:
    - Make per-process decisions
    - Track file access patterns
    - Implement complex policies
    """

    def __init__(self):
        """Initialize eBPF filter."""
        self._programs: Dict[str, EbpfProgram] = {}
        self._seccomp_filters: Dict[str, SeccompFilter] = {}
        self._ebpf_available = self._check_ebpf_available()
        self._seccomp_available = self._check_seccomp_available()

    def _check_ebpf_available(self) -> bool:
        """Check if eBPF is available."""
        try:
            # Check for eBPF support
            return os.path.exists("/sys/fs/bpf")
        except Exception:
            return False

    def _check_seccomp_available(self) -> bool:
        """Check if seccomp is available."""
        try:
            # Check for seccomp support in kernel
            with open("/proc/sys/kernel/seccomp/actions_avail", "r") as f:
                return len(f.read().strip()) > 0
        except Exception:
            # Try alternative check
            try:
                return os.path.exists("/proc/self/seccomp")
            except Exception:
                return False

    @property
    def ebpf_available(self) -> bool:
        """Check eBPF availability."""
        return self._ebpf_available

    @property
    def seccomp_available(self) -> bool:
        """Check seccomp availability."""
        return self._seccomp_available

    def create_seccomp_filter(
        self,
        filter_id: str,
        default_action: SeccompAction = SeccompAction.ALLOW,
    ) -> SeccompFilter:
        """Create a new seccomp filter.

        Args:
            filter_id: Unique filter identifier
            default_action: Default action for unmatched syscalls

        Returns:
            New SeccompFilter instance
        """
        filter_obj = SeccompFilter(
            filter_id=filter_id,
            default_action=default_action,
        )
        self._seccomp_filters[filter_id] = filter_obj
        return filter_obj

    def get_seccomp_filter(self, filter_id: str) -> Optional[SeccompFilter]:
        """Get a seccomp filter by ID."""
        return self._seccomp_filters.get(filter_id)

    def delete_seccomp_filter(self, filter_id: str) -> bool:
        """Delete a seccomp filter."""
        if filter_id in self._seccomp_filters:
            del self._seccomp_filters[filter_id]
            return True
        return False

    def create_ebpf_program(
        self,
        program_id: str,
        prog_type: EbpfProgType,
        attach_point: str,
        source: Optional[str] = None,
    ) -> EbpfProgram:
        """Create an eBPF program definition.

        Args:
            program_id: Unique program identifier
            prog_type: Type of eBPF program
            attach_point: Where to attach (e.g., tracepoint name)
            source: Optional eBPF C source code

        Returns:
            New EbpfProgram instance
        """
        program = EbpfProgram(
            program_id=program_id,
            prog_type=prog_type,
            attach_point=attach_point,
            source=source,
        )
        self._programs[program_id] = program
        return program

    def generate_file_access_tracer(self, policy_path: str) -> str:
        """Generate eBPF C source for file access tracing.

        Args:
            policy_path: Path to restrict access to

        Returns:
            eBPF C source code
        """
        # This would generate actual eBPF C code
        # For now, return a template
        return f"""
#include <linux/bpf.h>
#include <linux/ptrace.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

// Map for policy path
struct {{
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, char[256]);
}} policy_path SEC(".maps");

// Map for violation counts
struct {{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u32);  // pid
    __type(value, __u64);
}} violations SEC(".maps");

SEC("tracepoint/syscalls/sys_enter_openat")
int trace_openat(struct trace_event_raw_sys_enter* ctx)
{{
    char filename[256];
    const char *pathname = (const char *)ctx->args[1];

    bpf_probe_read_user_str(filename, sizeof(filename), pathname);

    // Check if path starts with policy path
    // Simplified check - full implementation would do proper prefix matching
    __u32 key = 0;
    char *policy = bpf_map_lookup_elem(&policy_path, &key);
    if (!policy)
        return 0;

    // Log access
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    bpf_printk("openat: pid=%d file=%s", pid, filename);

    return 0;
}}

char LICENSE[] SEC("license") = "GPL";
"""

    def generate_network_filter(self, allowed_ports: List[int]) -> str:
        """Generate eBPF source for network filtering.

        Args:
            allowed_ports: List of allowed destination ports

        Returns:
            eBPF C source code
        """
        ports_array = ", ".join(str(p) for p in allowed_ports)

        return f"""
#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <bpf/bpf_helpers.h>

// Allowed ports
static const __u16 allowed_ports[] = {{ {ports_array} }};
static const int num_ports = {len(allowed_ports)};

SEC("socket")
int network_filter(struct __sk_buff *skb)
{{
    void *data = (void *)(long)skb->data;
    void *data_end = (void *)(long)skb->data_end;

    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return 0;

    if (eth->h_proto != __constant_htons(ETH_P_IP))
        return 0;

    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end)
        return 0;

    if (ip->protocol != IPPROTO_TCP)
        return 0;

    struct tcphdr *tcp = (void *)ip + (ip->ihl * 4);
    if ((void *)(tcp + 1) > data_end)
        return 0;

    __u16 dport = __constant_ntohs(tcp->dest);

    // Check if port is allowed
    for (int i = 0; i < num_ports; i++) {{
        if (dport == allowed_ports[i])
            return 0;  // Allow
    }}

    return -1;  // Drop
}}

char LICENSE[] SEC("license") = "GPL";
"""

    def create_deny_filter(
        self,
        filter_id: str,
        syscalls: List[str],
        errno_value: int = 13,
    ) -> SeccompFilter:
        """Create a filter that denies specific syscalls.

        Args:
            filter_id: Filter identifier
            syscalls: Syscalls to deny
            errno_value: Error code to return

        Returns:
            Configured SeccompFilter
        """
        filter_obj = self.create_seccomp_filter(filter_id, SeccompAction.ALLOW)

        for syscall in syscalls:
            if syscall in SYSCALL_NUMBERS:
                filter_obj.add_syscall(
                    syscall,
                    SeccompAction.ERRNO,
                    errno_value=errno_value,
                )

        return filter_obj

    def create_allowlist_filter(
        self,
        filter_id: str,
        allowed_syscalls: List[str],
    ) -> SeccompFilter:
        """Create a filter that only allows specific syscalls.

        Args:
            filter_id: Filter identifier
            allowed_syscalls: Syscalls to allow

        Returns:
            Configured SeccompFilter
        """
        filter_obj = self.create_seccomp_filter(filter_id, SeccompAction.ERRNO)

        for syscall in allowed_syscalls:
            if syscall in SYSCALL_NUMBERS:
                filter_obj.add_syscall(syscall, SeccompAction.ALLOW)

        return filter_obj

    def create_sandbox_filter(self, filter_id: str) -> SeccompFilter:
        """Create a typical sandbox filter.

        Blocks dangerous syscalls while allowing common operations.

        Args:
            filter_id: Filter identifier

        Returns:
            Configured SeccompFilter
        """
        filter_obj = self.create_seccomp_filter(filter_id, SeccompAction.ALLOW)

        # Block dangerous syscalls
        dangerous = [
            "ptrace",
            "personality",
            "mount",
            "umount",
            "umount2",
            "pivot_root",
            "chroot",
            "acct",
            "sethostname",
            "setdomainname",
            "iopl",
            "ioperm",
            "create_module",
            "init_module",
            "delete_module",
            "finit_module",
            "kexec_load",
            "kexec_file_load",
            "reboot",
            "settimeofday",
            "clock_settime",
            "clock_adjtime",
            "adjtimex",
        ]

        for syscall in dangerous:
            if syscall in SYSCALL_NUMBERS:
                filter_obj.add_syscall(syscall, SeccompAction.ERRNO, errno_value=1)  # EPERM

        return filter_obj

    def list_filters(self) -> List[Dict[str, Any]]:
        """List all configured filters."""
        result = []

        for filter_obj in self._seccomp_filters.values():
            result.append(
                {
                    "filter_id": filter_obj.filter_id,
                    "type": "seccomp",
                    "default_action": filter_obj.default_action.value,
                    "syscall_count": len(filter_obj.syscalls),
                }
            )

        for program in self._programs.values():
            result.append(
                {
                    "program_id": program.program_id,
                    "type": "ebpf",
                    "prog_type": program.prog_type.value,
                    "loaded": program.loaded,
                }
            )

        return result

    def export_filters(self) -> Dict[str, Any]:
        """Export all filters as JSON-compatible dict."""
        return {
            "seccomp_filters": {
                fid: json.loads(f.to_json()) for fid, f in self._seccomp_filters.items()
            },
            "ebpf_programs": {pid: p.to_dict() for pid, p in self._programs.items()},
        }
