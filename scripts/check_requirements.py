#!/usr/bin/env python3
"""
Agent OS Requirements Checker

Verifies that the system meets the minimum requirements for running Agent OS.
Run this script to diagnose installation and runtime issues.

Usage:
    python scripts/check_requirements.py
    python scripts/check_requirements.py --verbose
    python scripts/check_requirements.py --json
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Status(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    name: str
    status: Status
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    fix_hint: Optional[str] = None


class RequirementsChecker:
    """Check system requirements for Agent OS."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[CheckResult] = []

    def add_result(self, result: CheckResult) -> None:
        self.results.append(result)
        if not self.verbose:
            return
        status_symbols = {
            Status.PASS: "\033[92m✓\033[0m",
            Status.WARN: "\033[93m!\033[0m",
            Status.FAIL: "\033[91m✗\033[0m",
            Status.SKIP: "\033[90m-\033[0m",
        }
        print(f"  {status_symbols[result.status]} {result.name}: {result.message}")

    def check_python_version(self) -> None:
        """Check Python version is 3.10+."""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"

        if version.major == 3 and version.minor >= 10:
            self.add_result(CheckResult(
                name="Python Version",
                status=Status.PASS,
                message=f"Python {version_str}",
                details={"version": version_str, "path": sys.executable},
            ))
        else:
            self.add_result(CheckResult(
                name="Python Version",
                status=Status.FAIL,
                message=f"Python {version_str} (requires 3.10+)",
                details={"version": version_str},
                fix_hint="Install Python 3.10 or newer from https://python.org",
            ))

    def check_memory(self) -> None:
        """Check available system memory."""
        try:
            if os.path.exists("/proc/meminfo"):
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            total_kb = int(line.split()[1])
                            total_gb = total_kb / (1024 * 1024)
                            break
                    else:
                        raise ValueError("MemTotal not found")
            else:
                # macOS/Windows fallback
                import ctypes
                if platform.system() == "Darwin":
                    result = subprocess.run(
                        ["sysctl", "-n", "hw.memsize"],
                        capture_output=True, text=True
                    )
                    total_gb = int(result.stdout.strip()) / (1024**3)
                elif platform.system() == "Windows":
                    kernel32 = ctypes.windll.kernel32
                    c_ulonglong = ctypes.c_ulonglong
                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ('dwLength', ctypes.c_ulong),
                            ('dwMemoryLoad', ctypes.c_ulong),
                            ('ullTotalPhys', c_ulonglong),
                            ('ullAvailPhys', c_ulonglong),
                            ('ullTotalPageFile', c_ulonglong),
                            ('ullAvailPageFile', c_ulonglong),
                            ('ullTotalVirtual', c_ulonglong),
                            ('ullAvailVirtual', c_ulonglong),
                            ('ullAvailExtendedVirtual', c_ulonglong),
                        ]
                    stat = MEMORYSTATUSEX()
                    stat.dwLength = ctypes.sizeof(stat)
                    kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                    total_gb = stat.ullTotalPhys / (1024**3)
                else:
                    total_gb = 0

            if total_gb >= 16:
                status = Status.PASS
                message = f"{total_gb:.1f} GB (recommended)"
            elif total_gb >= 8:
                status = Status.WARN
                message = f"{total_gb:.1f} GB (minimum met, 16GB recommended)"
            elif total_gb >= 4:
                status = Status.WARN
                message = f"{total_gb:.1f} GB (below minimum, may have issues)"
            else:
                status = Status.FAIL
                message = f"{total_gb:.1f} GB (insufficient, 8GB minimum)"

            self.add_result(CheckResult(
                name="System Memory",
                status=status,
                message=message,
                details={"total_gb": round(total_gb, 2)},
                fix_hint="Agent OS requires at least 8GB RAM, 16GB recommended" if status != Status.PASS else None,
            ))
        except Exception as e:
            self.add_result(CheckResult(
                name="System Memory",
                status=Status.SKIP,
                message=f"Could not determine: {e}",
            ))

    def check_disk_space(self) -> None:
        """Check available disk space."""
        try:
            total, used, free = shutil.disk_usage("/")
            free_gb = free / (1024**3)

            if free_gb >= 50:
                status = Status.PASS
                message = f"{free_gb:.1f} GB free (recommended)"
            elif free_gb >= 20:
                status = Status.WARN
                message = f"{free_gb:.1f} GB free (minimum met)"
            else:
                status = Status.FAIL
                message = f"{free_gb:.1f} GB free (insufficient)"

            self.add_result(CheckResult(
                name="Disk Space",
                status=status,
                message=message,
                details={"free_gb": round(free_gb, 2), "total_gb": round(total / (1024**3), 2)},
                fix_hint="Agent OS requires at least 20GB free disk space" if status == Status.FAIL else None,
            ))
        except Exception as e:
            self.add_result(CheckResult(
                name="Disk Space",
                status=Status.SKIP,
                message=f"Could not determine: {e}",
            ))

    def check_gpu(self) -> None:
        """Check for GPU availability."""
        try:
            # Check for NVIDIA GPU
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_info = result.stdout.strip().split("\n")[0]
                self.add_result(CheckResult(
                    name="GPU",
                    status=Status.PASS,
                    message=f"NVIDIA: {gpu_info}",
                    details={"type": "nvidia", "info": gpu_info},
                ))
                return
        except FileNotFoundError:
            pass

        # Check for CUDA via PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                self.add_result(CheckResult(
                    name="GPU",
                    status=Status.PASS,
                    message=f"CUDA available: {gpu_name}",
                    details={"type": "cuda", "name": gpu_name},
                ))
                return
        except ImportError:
            pass

        # Check for Apple Silicon
        if platform.system() == "Darwin" and platform.processor() == "arm":
            self.add_result(CheckResult(
                name="GPU",
                status=Status.PASS,
                message="Apple Silicon (Metal acceleration available)",
                details={"type": "apple_silicon"},
            ))
            return

        self.add_result(CheckResult(
            name="GPU",
            status=Status.WARN,
            message="No GPU detected (CPU inference will be slower)",
            fix_hint="GPU is optional but recommended for better performance",
        ))

    def check_ollama(self) -> None:
        """Check if Ollama is installed and running."""
        # Check if installed
        ollama_path = shutil.which("ollama")
        if not ollama_path:
            self.add_result(CheckResult(
                name="Ollama",
                status=Status.FAIL,
                message="Not installed",
                fix_hint="Install Ollama from https://ollama.com",
            ))
            return

        # Check if running
        try:
            import urllib.request
            req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                models = [m["name"] for m in data.get("models", [])]

            if models:
                self.add_result(CheckResult(
                    name="Ollama",
                    status=Status.PASS,
                    message=f"Running with {len(models)} model(s): {', '.join(models[:3])}",
                    details={"models": models},
                ))
            else:
                self.add_result(CheckResult(
                    name="Ollama",
                    status=Status.WARN,
                    message="Running but no models installed",
                    fix_hint="Run: ollama pull mistral",
                ))
        except Exception:
            self.add_result(CheckResult(
                name="Ollama",
                status=Status.WARN,
                message="Installed but not running",
                details={"path": ollama_path},
                fix_hint="Start Ollama with: ollama serve",
            ))

    def check_docker(self) -> None:
        """Check if Docker is available."""
        docker_path = shutil.which("docker")
        if not docker_path:
            self.add_result(CheckResult(
                name="Docker",
                status=Status.SKIP,
                message="Not installed (optional)",
            ))
            return

        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                self.add_result(CheckResult(
                    name="Docker",
                    status=Status.PASS,
                    message="Installed and running",
                ))
            else:
                self.add_result(CheckResult(
                    name="Docker",
                    status=Status.WARN,
                    message="Installed but not running",
                    fix_hint="Start Docker daemon",
                ))
        except subprocess.TimeoutExpired:
            self.add_result(CheckResult(
                name="Docker",
                status=Status.WARN,
                message="Installed but not responding",
            ))

    def check_redis(self) -> None:
        """Check if Redis is available."""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(("localhost", 6379))
            sock.close()

            if result == 0:
                self.add_result(CheckResult(
                    name="Redis",
                    status=Status.PASS,
                    message="Running on localhost:6379",
                ))
            else:
                self.add_result(CheckResult(
                    name="Redis",
                    status=Status.SKIP,
                    message="Not running (optional, will use in-memory rate limiting)",
                ))
        except Exception:
            self.add_result(CheckResult(
                name="Redis",
                status=Status.SKIP,
                message="Not detected (optional)",
            ))

    def check_python_packages(self) -> None:
        """Check critical Python packages."""
        packages = {
            "fastapi": "Web framework",
            "uvicorn": "ASGI server",
            "pydantic": "Data validation",
            "cryptography": "Encryption",
        }

        missing = []
        installed = []

        for package, description in packages.items():
            try:
                __import__(package)
                installed.append(package)
            except ImportError:
                missing.append(package)

        if not missing:
            self.add_result(CheckResult(
                name="Python Packages",
                status=Status.PASS,
                message=f"All {len(installed)} critical packages installed",
                details={"installed": installed},
            ))
        else:
            self.add_result(CheckResult(
                name="Python Packages",
                status=Status.FAIL,
                message=f"Missing: {', '.join(missing)}",
                details={"missing": missing, "installed": installed},
                fix_hint="Run: pip install -r requirements.txt",
            ))

    def check_network(self) -> None:
        """Check network connectivity."""
        try:
            import urllib.request
            req = urllib.request.Request("https://www.google.com", method="HEAD")
            with urllib.request.urlopen(req, timeout=5):
                self.add_result(CheckResult(
                    name="Network",
                    status=Status.PASS,
                    message="Internet connectivity available",
                ))
        except Exception:
            self.add_result(CheckResult(
                name="Network",
                status=Status.WARN,
                message="No internet (may affect model downloads)",
            ))

    def run_all_checks(self) -> None:
        """Run all requirement checks."""
        if self.verbose:
            print("\n🔍 Checking Agent OS Requirements...\n")

        self.check_python_version()
        self.check_memory()
        self.check_disk_space()
        self.check_gpu()
        self.check_python_packages()
        self.check_ollama()
        self.check_docker()
        self.check_redis()
        self.check_network()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all checks."""
        counts = {status: 0 for status in Status}
        for result in self.results:
            counts[result.status] += 1

        all_passed = counts[Status.FAIL] == 0
        has_warnings = counts[Status.WARN] > 0

        return {
            "passed": all_passed,
            "has_warnings": has_warnings,
            "counts": {s.value: c for s, c in counts.items()},
            "results": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "message": r.message,
                    "details": r.details,
                    "fix_hint": r.fix_hint,
                }
                for r in self.results
            ],
            "system": {
                "platform": platform.system(),
                "platform_release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
        }

    def print_summary(self) -> None:
        """Print human-readable summary."""
        summary = self.get_summary()

        print("\n" + "=" * 50)
        print("AGENT OS REQUIREMENTS CHECK SUMMARY")
        print("=" * 50)

        print(f"\nSystem: {summary['system']['platform']} {summary['system']['platform_release']}")
        print(f"Machine: {summary['system']['machine']}")
        print(f"Python: {summary['system']['python']}")

        print(f"\nResults:")
        print(f"  ✓ Passed:  {summary['counts']['PASS']}")
        print(f"  ! Warning: {summary['counts']['WARN']}")
        print(f"  ✗ Failed:  {summary['counts']['FAIL']}")
        print(f"  - Skipped: {summary['counts']['SKIP']}")

        # Show failures and warnings with fix hints
        issues = [r for r in self.results if r.status in (Status.FAIL, Status.WARN) and r.fix_hint]
        if issues:
            print("\n📋 Recommended Actions:")
            for i, result in enumerate(issues, 1):
                status_emoji = "❌" if result.status == Status.FAIL else "⚠️"
                print(f"  {i}. {status_emoji} {result.name}: {result.fix_hint}")

        print("\n" + "=" * 50)
        if summary["passed"]:
            if summary["has_warnings"]:
                print("✅ System meets minimum requirements (with warnings)")
            else:
                print("✅ System meets all requirements!")
        else:
            print("❌ System does NOT meet minimum requirements")
            print("   Please address the failed checks above.")
        print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Check system requirements for Agent OS"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output during checks"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    args = parser.parse_args()

    checker = RequirementsChecker(verbose=args.verbose or not args.json)
    checker.run_all_checks()

    if args.json:
        print(json.dumps(checker.get_summary(), indent=2))
    else:
        checker.print_summary()

    # Exit with error code if any checks failed
    summary = checker.get_summary()
    sys.exit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
