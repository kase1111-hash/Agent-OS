"""
Agent OS Constitutional Kernel

The core kernel that manages constitutional documents, enforces rules,
and provides runtime governance for the Agent OS system.
"""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from watchdog.events import FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .exceptions import (
    ConstitutionLoadError,
    KernelNotInitializedError,
    ReloadCallbackError,
    SupremeConstitutionError,
)
from .models import (
    AuthorityLevel,
    Constitution,
    ConstitutionRegistry,
    Rule,
    RuleType,
    ValidationResult,
)
from .parser import ConstitutionParser
from .validator import ConstitutionValidator

logger = logging.getLogger(__name__)


@dataclass
class RequestContext:
    """Context for a request being validated against the constitution."""

    request_id: str
    source: str  # Agent or "user"
    destination: str  # Target agent
    intent: str  # Classified intent
    content: str  # Request content
    requires_memory: bool = False
    metadata: Dict = field(default_factory=dict)


@dataclass
class EnforcementResult:
    """Result of enforcing constitutional rules on a request."""

    allowed: bool
    violated_rules: List[Rule] = field(default_factory=list)
    applicable_rules: List[Rule] = field(default_factory=list)
    reason: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    escalate_to_human: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now())


class ConstitutionFileHandler(FileSystemEventHandler):
    """File system event handler for constitution file changes."""

    def __init__(self, kernel: "ConstitutionalKernel", paths: Set[Path]):
        self.kernel = kernel
        self.monitored_paths = paths
        self._debounce_time = 1.0  # seconds
        self._last_reload: Dict[Path, float] = {}
        self._lock = threading.Lock()  # Thread-safe debounce tracking

    def on_modified(self, event: FileModifiedEvent) -> None:
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path in self.monitored_paths:
            # Thread-safe debounce with lock
            with self._lock:
                now = time.time()
                last = self._last_reload.get(path, 0)
                if now - last > self._debounce_time:
                    self._last_reload[path] = now
                    should_reload = True
                else:
                    should_reload = False

            if should_reload:
                logger.info(f"Constitution file changed: {path}")
                self.kernel._reload_constitution(path)


class ConstitutionalKernel:
    """
    The Constitutional Kernel is the core of Agent OS governance.

    It:
    - Loads and parses constitutional documents
    - Maintains the constitution registry
    - Validates requests against constitutional rules
    - Enforces rule precedence hierarchy
    - Supports hot-reload of constitution changes
    - Provides rule lookup for agents
    """

    def __init__(
        self,
        constitution_dir: Optional[Path] = None,
        supreme_constitution_path: Optional[Path] = None,
        enable_hot_reload: bool = True,
    ):
        """
        Initialize the Constitutional Kernel.

        Args:
            constitution_dir: Directory containing constitution files
            supreme_constitution_path: Path to supreme constitution (CONSTITUTION.md)
            enable_hot_reload: Enable watching for file changes
        """
        self.constitution_dir = constitution_dir
        self.supreme_constitution_path = supreme_constitution_path
        self.enable_hot_reload = enable_hot_reload

        self._parser = ConstitutionParser()
        self._validator = ConstitutionValidator()
        self._registry = ConstitutionRegistry()

        self._file_hashes: Dict[Path, str] = {}
        self._reload_callbacks: List[Callable[[Path], None]] = []
        self._lock = threading.RLock()

        self._observer: Optional[Observer] = None
        self._initialized = False

    def initialize(self) -> ValidationResult:
        """
        Initialize the kernel by loading all constitutions.

        Returns:
            ValidationResult from loading all constitutions

        Raises:
            SupremeConstitutionError: If the supreme constitution cannot be loaded
                (this is a fatal error that prevents safe operation)
        """
        result = ValidationResult(is_valid=True)

        with self._lock:
            # Load supreme constitution first - this is CRITICAL and must succeed
            if self.supreme_constitution_path:
                try:
                    supreme = self._load_constitution(self.supreme_constitution_path)
                    self._registry.register(supreme)
                    logger.info(f"Loaded supreme constitution: {self.supreme_constitution_path}")
                except FileNotFoundError as e:
                    raise SupremeConstitutionError(
                        f"Supreme constitution file not found: {self.supreme_constitution_path}",
                        path=self.supreme_constitution_path,
                        original_error=e,
                    )
                except PermissionError as e:
                    raise SupremeConstitutionError(
                        f"Permission denied reading supreme constitution: {self.supreme_constitution_path}",
                        path=self.supreme_constitution_path,
                        original_error=e,
                    )
                except Exception as e:
                    raise SupremeConstitutionError(
                        f"Failed to load supreme constitution: {e}",
                        path=self.supreme_constitution_path,
                        original_error=e,
                    )

            # Load other constitutions from directory - these are non-fatal
            if self.constitution_dir and self.constitution_dir.exists():
                for path in self._find_constitution_files(self.constitution_dir):
                    if path != self.supreme_constitution_path:
                        try:
                            constitution = self._load_constitution(path)
                            self._registry.register(constitution)
                            logger.info(f"Loaded constitution: {path}")
                        except FileNotFoundError:
                            result.add_warning(f"Constitution file not found: {path}")
                        except PermissionError:
                            result.add_warning(f"Permission denied reading: {path}")
                        except Exception as e:
                            result.add_warning(f"Failed to load {path}: {type(e).__name__}: {e}")

            # Validate the entire registry
            registry_validation = self._validator.validate_registry(self._registry)
            result.errors.extend(registry_validation.errors)
            result.warnings.extend(registry_validation.warnings)
            result.conflicts.extend(registry_validation.conflicts)
            if not registry_validation.is_valid:
                result.is_valid = False

            # Start file watcher if enabled
            if self.enable_hot_reload:
                self._start_file_watcher()

            self._initialized = True

        return result

    def shutdown(self) -> None:
        """Shutdown the kernel and stop file watching."""
        with self._lock:
            if self._observer:
                self._observer.stop()
                self._observer.join(timeout=5)
                self._observer = None
            self._initialized = False

    def enforce(self, context: RequestContext) -> EnforcementResult:
        """
        Enforce constitutional rules on a request.

        Args:
            context: Request context to validate

        Returns:
            EnforcementResult indicating if request is allowed

        Raises:
            KernelNotInitializedError: If kernel has not been initialized
        """
        if not self._initialized:
            raise KernelNotInitializedError("enforce")

        with self._lock:
            # Get applicable rules for destination agent
            rules = self._registry.get_rules_for_agent(context.destination)

            violated = []
            applicable = []

            for rule in rules:
                if self._rule_applies(rule, context):
                    applicable.append(rule)

                    if self._rule_violated(rule, context):
                        violated.append(rule)

            # Determine result
            if violated:
                # Check if any violation requires human escalation
                escalate = any(r.is_immutable for r in violated)

                # Generate suggestions
                suggestions = self._generate_suggestions(violated)

                return EnforcementResult(
                    allowed=False,
                    violated_rules=violated,
                    applicable_rules=applicable,
                    reason=self._format_violation_reason(violated),
                    suggestions=suggestions,
                    escalate_to_human=escalate,
                )

            return EnforcementResult(
                allowed=True,
                applicable_rules=applicable,
            )

    def get_rules_for_agent(self, agent_scope: str) -> List[Rule]:
        """
        Get all applicable rules for an agent.

        Args:
            agent_scope: Agent scope identifier

        Returns:
            List of applicable rules, sorted by authority
        """
        with self._lock:
            return self._registry.get_rules_for_agent(agent_scope)

    def get_supreme_constitution(self) -> Optional[Constitution]:
        """Get the supreme constitution."""
        with self._lock:
            return self._registry.get_supreme()

    def get_constitution(self, scope: str) -> Optional[Constitution]:
        """Get constitution for a specific scope."""
        with self._lock:
            return self._registry.get(scope)

    def validate_constitution(self, constitution: Constitution) -> ValidationResult:
        """Validate a constitution document."""
        return self._validator.validate(constitution)

    def register_reload_callback(self, callback: Callable[[Path], None]) -> None:
        """
        Register a callback to be called when a constitution is reloaded.

        Args:
            callback: Function to call with path of reloaded constitution
        """
        self._reload_callbacks.append(callback)

    def reload_all(self) -> ValidationResult:
        """Force reload of all constitutions."""
        return self.initialize()

    def _load_constitution(self, path: Path) -> Constitution:
        """Load and parse a constitution file."""
        constitution = self._parser.parse_file(path)
        self._file_hashes[path] = constitution.file_hash
        return constitution

    def _find_constitution_files(self, directory: Path) -> List[Path]:
        """Find all constitution files in a directory."""
        files = []

        # Look for constitution.md files in agents directories
        for path in directory.rglob("constitution.md"):
            files.append(path)

        # Also check for CONSTITUTION.md at root
        root_constitution = directory / "CONSTITUTION.md"
        if root_constitution.exists() and root_constitution not in files:
            files.append(root_constitution)

        return files

    def _start_file_watcher(self) -> None:
        """Start watching constitution files for changes."""
        if self._observer:
            self._observer.stop()

        monitored_paths = set(self._file_hashes.keys())
        if not monitored_paths:
            return

        handler = ConstitutionFileHandler(self, monitored_paths)
        self._observer = Observer()

        # Watch directories containing constitution files
        watched_dirs = set()
        for path in monitored_paths:
            parent = path.parent
            if parent not in watched_dirs:
                self._observer.schedule(handler, str(parent), recursive=False)
                watched_dirs.add(parent)

        self._observer.start()
        logger.info(f"Started watching {len(monitored_paths)} constitution files")

    def _reload_constitution(self, path: Path) -> bool:
        """
        Reload a single constitution file.

        Returns:
            True if reload was successful, False otherwise
        """
        with self._lock:
            try:
                # Check if file actually changed
                content = path.read_text()
                new_hash = hashlib.sha256(content.encode()).hexdigest()

                if new_hash == self._file_hashes.get(path):
                    return True  # No actual change needed

                # Reload the constitution
                constitution = self._parser.parse_content(content, path)

                # Validate before accepting
                validation = self._validator.validate(constitution)
                if not validation.is_valid:
                    logger.error(f"Reloaded constitution is invalid: {validation.errors}")
                    return False

                # Check against supreme constitution if not supreme itself
                if constitution.metadata.authority_level < AuthorityLevel.SUPREME:
                    supreme = self._registry.get_supreme()
                    if supreme:
                        supreme_validation = self._validator.validate_against_supreme(
                            constitution, supreme
                        )
                        if not supreme_validation.is_valid:
                            logger.error(
                                f"Constitution violates supreme: {supreme_validation.errors}"
                            )
                            return False

                # Update registry
                self._registry.register(constitution)
                self._file_hashes[path] = new_hash

                logger.info(f"Successfully reloaded constitution: {path}")

                # Notify callbacks - collect errors but don't let them prevent other callbacks
                callback_errors = []
                for callback in self._reload_callbacks:
                    try:
                        callback(path)
                    except Exception as e:
                        callback_name = getattr(callback, "__name__", repr(callback))
                        logger.error(f"Reload callback '{callback_name}' error: {e}")
                        callback_errors.append(
                            ReloadCallbackError(
                                message=f"Callback failed: {e}",
                                callback_name=callback_name,
                                original_error=e,
                            )
                        )

                # Log summary if there were callback errors
                if callback_errors:
                    logger.warning(
                        f"Constitution reloaded but {len(callback_errors)} callback(s) failed"
                    )

                return True

            except FileNotFoundError:
                logger.error(f"Constitution file no longer exists: {path}")
                return False
            except PermissionError:
                logger.error(f"Permission denied reloading constitution: {path}")
                return False
            except Exception as e:
                logger.error(
                    f"Failed to reload constitution {path}: {type(e).__name__}: {e}"
                )
                return False

    def _rule_applies(self, rule: Rule, context: RequestContext) -> bool:
        """Check if a rule applies to the given context."""
        # Rule applies if:
        # 1. Scope matches (all_agents or specific agent)
        # 2. Intent/content is relevant to rule keywords

        if rule.scope != "all_agents" and rule.scope != context.destination:
            return False

        # Check if context content relates to rule keywords
        content_lower = context.content.lower()
        intent_lower = context.intent.lower()

        for keyword in rule.keywords:
            if keyword in content_lower or keyword in intent_lower:
                return True

        # Memory-related rules apply if request requires memory
        if context.requires_memory:
            memory_keywords = {"memory", "store", "persist", "remember", "save"}
            if rule.keywords & memory_keywords:
                return True

        return False

    def _rule_violated(self, rule: Rule, context: RequestContext) -> bool:
        """Check if a rule is violated by the context."""
        content_lower = context.content.lower()
        intent_lower = context.intent.lower()

        if rule.rule_type == RuleType.PROHIBITION:
            # Check if request attempts prohibited action
            for keyword in rule.keywords:
                if keyword in content_lower:
                    # The prohibition keyword is in the request
                    return True

        elif rule.rule_type == RuleType.MANDATE:
            # Mandate rules require certain actions/elements to be present
            # Check if the mandate keywords are present in the request
            mandate_keywords = rule.keywords
            if not mandate_keywords:
                return False

            # For mandate rules, check if required elements are referenced
            # A mandate is violated if the context relates to the mandate topic
            # but doesn't include the required elements
            topic_match = any(
                kw in content_lower or kw in intent_lower for kw in mandate_keywords
            )

            if topic_match:
                # Check for mandate indicator words that suggest compliance
                compliance_indicators = {
                    "review", "validate", "verify", "check", "confirm",
                    "ensure", "approved", "authorization", "consent"
                }
                has_compliance = any(
                    indicator in content_lower for indicator in compliance_indicators
                )

                # If the topic matches but no compliance indicators, it may be a violation
                # Check metadata for explicit mandate compliance
                mandate_compliance = context.metadata.get("mandate_compliance", {})
                rule_id = rule.id if hasattr(rule, "id") else None

                if rule_id and rule_id in mandate_compliance:
                    # Explicit compliance recorded
                    return not mandate_compliance[rule_id]

                # If no explicit compliance and no indicators, flag as potential violation
                if not has_compliance:
                    return True

        return False

    def _format_violation_reason(self, violated_rules: List[Rule]) -> str:
        """Format a human-readable violation reason."""
        if not violated_rules:
            return "Unknown violation"

        reasons = []
        for rule in violated_rules[:3]:  # Limit to top 3
            section = " > ".join(rule.section_path) if rule.section_path else rule.section
            reasons.append(f"- [{section}]: {rule.content[:100]}...")

        return "Constitutional violation(s):\n" + "\n".join(reasons)

    def _generate_suggestions(self, violated_rules: List[Rule]) -> List[str]:
        """Generate suggestions for resolving violations."""
        suggestions = []

        for rule in violated_rules:
            if rule.rule_type == RuleType.PROHIBITION:
                suggestions.append(
                    f"Avoid actions related to: {', '.join(list(rule.keywords)[:5])}"
                )
            elif rule.rule_type == RuleType.ESCALATION:
                suggestions.append("Request human steward approval for this action")

        if any(r.is_immutable for r in violated_rules):
            suggestions.append("This rule is immutable and cannot be overridden")

        return suggestions


def create_kernel(
    project_root: Path,
    enable_hot_reload: bool = True,
) -> ConstitutionalKernel:
    """
    Convenience function to create and initialize a kernel.

    Args:
        project_root: Root directory of Agent OS project
        enable_hot_reload: Enable file watching

    Returns:
        Initialized ConstitutionalKernel
    """
    supreme_path = project_root / "CONSTITUTION.md"
    agents_dir = project_root / "agents"

    kernel = ConstitutionalKernel(
        constitution_dir=agents_dir,
        supreme_constitution_path=supreme_path if supreme_path.exists() else None,
        enable_hot_reload=enable_hot_reload,
    )

    result = kernel.initialize()
    if not result.is_valid:
        logger.warning(f"Kernel initialized with errors: {result.errors}")

    return kernel
