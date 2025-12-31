"""Conversational Kernel Engine.

The main orchestrator for the conversational kernel, coordinating:
- Natural language interpretation
- Rule management
- Policy compilation and enforcement
- System monitoring
- Context awareness
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .context import AgentContext, ContextMemory, FolderContext, UserContext
from .ebpf import EbpfFilter, SeccompFilter
from .fuse import FuseConfig, FuseWrapper
from .interpreter import IntentAction, IntentParser, ParsedIntent, PolicyInterpreter
from .monitor import AuditEntry, AuditLog, EventType, FileMonitor, MonitorEvent
from .policy import FilePolicy, Policy, PolicyCompiler, PolicyType, SyscallPolicy
from .rules import Rule, RuleAction, RuleEffect, RuleRegistry, RuleScope

logger = logging.getLogger(__name__)


class KernelState(str, Enum):
    """States of the conversational kernel."""

    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ENFORCING = "enforcing"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class KernelConfig:
    """Configuration for the conversational kernel."""

    data_dir: Path = field(default_factory=lambda: Path.home() / ".agentos" / "kernel")
    enable_fuse: bool = True
    enable_ebpf: bool = True
    enable_monitoring: bool = True
    enable_audit: bool = True
    default_effect: RuleEffect = RuleEffect.ALLOW
    clarification_enabled: bool = True
    auto_suggest: bool = True
    max_history: int = 1000

    def __post_init__(self):
        """Ensure data directory exists."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)


class ConversationalKernel:
    """The main conversational kernel engine.

    Provides a unified interface for:
    - Processing natural language policy requests
    - Managing rules and policies
    - Enforcing access control
    - Monitoring and auditing
    """

    def __init__(self, config: Optional[KernelConfig] = None):
        """Initialize the conversational kernel.

        Args:
            config: Kernel configuration
        """
        self.config = config or KernelConfig()
        self._state = KernelState.INITIALIZING

        # Initialize components
        self._init_components()

        self._state = KernelState.READY
        logger.info("Conversational kernel initialized")

    def _init_components(self) -> None:
        """Initialize kernel components."""
        # Rule registry
        self.rule_registry = RuleRegistry(db_path=self.config.data_dir / "rules.db")

        # Context memory
        self.context = ContextMemory(db_path=self.config.data_dir / "context.db")

        # Policy interpreter
        self.interpreter = PolicyInterpreter(
            parser=IntentParser(),
            clarification_handler=(
                self._handle_clarification if self.config.clarification_enabled else None
            ),
        )

        # Policy compiler
        self.compiler = PolicyCompiler()

        # Audit log
        if self.config.enable_audit:
            self.audit_log = AuditLog(db_path=self.config.data_dir / "audit.db")
        else:
            self.audit_log = None

        # File monitor
        if self.config.enable_monitoring:
            self.monitor = FileMonitor(audit_log=self.audit_log)
            self.monitor.on_event(self._on_monitor_event)
        else:
            self.monitor = None

        # FUSE wrapper
        if self.config.enable_fuse:
            self.fuse = FuseWrapper(
                rule_registry=self.rule_registry,
                audit_handler=self._audit_event if self.audit_log else None,
            )
        else:
            self.fuse = None

        # eBPF filter
        if self.config.enable_ebpf:
            self.ebpf = EbpfFilter()
        else:
            self.ebpf = None

        # Callbacks
        self._event_handlers: List[Callable[[Dict[str, Any]], None]] = []
        self._clarification_handler: Optional[Callable[[str, List[str]], str]] = None

    @property
    def state(self) -> KernelState:
        """Get current kernel state."""
        return self._state

    def process_request(
        self,
        text: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process a natural language request.

        Args:
            text: User's natural language input
            user_id: ID of requesting user
            context: Additional context (cwd, etc.)

        Returns:
            Processing result with rule/policy information
        """
        self._state = KernelState.PROCESSING
        context = context or {}

        try:
            # Get user context
            if user_id:
                user_ctx = self.context.get_user(user_id)
                if user_ctx:
                    context["user"] = user_ctx.username
                    context["home"] = user_ctx.home_directory

            # Parse intent
            intent, needs_clarification = self.interpreter.interpret(text, context)

            result: Dict[str, Any] = {
                "success": True,
                "intent": intent.to_dict(),
                "needs_clarification": needs_clarification,
            }

            if needs_clarification:
                result["clarification_questions"] = intent.ambiguities
                self._state = KernelState.READY
                return result

            # Process based on intent action
            if intent.action == IntentAction.SET_RULE:
                result.update(self._create_rule(intent, user_id))
            elif intent.action == IntentAction.MODIFY_RULE:
                result.update(self._modify_rule(intent))
            elif intent.action == IntentAction.DELETE_RULE:
                result.update(self._delete_rule(intent))
            elif intent.action == IntentAction.QUERY_RULES:
                result.update(self._query_rules(intent))
            elif intent.action == IntentAction.CHECK_ACCESS:
                result.update(self._check_access(intent, context))
            elif intent.action == IntentAction.EXPLAIN:
                result.update(self._explain_rules(intent))
            elif intent.action == IntentAction.SUGGEST:
                result.update(self._suggest_rules(intent, context))
            elif intent.action == IntentAction.UNDO:
                result.update(self._undo_last(user_id))

            # Record interaction
            if user_id:
                self.context.record_interaction(
                    user_id=user_id,
                    interaction_type=intent.action.value,
                    input_text=text,
                    output_rule_id=result.get("rule_id"),
                    context_data=context,
                )

            self._state = KernelState.READY
            return result

        except Exception as e:
            logger.exception("Error processing request")
            self._state = KernelState.ERROR
            return {
                "success": False,
                "error": str(e),
            }

    def _create_rule(self, intent: ParsedIntent, user_id: Optional[str]) -> Dict[str, Any]:
        """Create a new rule from intent."""
        rule = intent.to_rule()
        rule.created_by = user_id or "system"

        try:
            rule_id = self.rule_registry.add_rule(rule)

            # Update folder context
            if intent.target:
                folder_ctx = self.context.get_folder(intent.target)
                if not folder_ctx:
                    folder_ctx = FolderContext(path=intent.target)
                folder_ctx.rule_ids.append(rule_id)
                self.context.set_folder(folder_ctx)

            # Compile and apply policies
            policies = self.compiler.compile_rule(rule)
            self._apply_policies(policies)

            return {
                "rule_id": rule_id,
                "rule": rule.to_dict(),
                "policies_created": len(policies),
                "message": f"Created rule: {rule.reason}",
            }

        except Exception as e:
            return {
                "error": str(e),
                "success": False,
            }

    def _modify_rule(self, intent: ParsedIntent) -> Dict[str, Any]:
        """Modify an existing rule."""
        # Find rule by target or conditions
        rule_id = intent.conditions.get("rule_id")
        if not rule_id and intent.target:
            rules = self.rule_registry.get_rules_for_target(intent.target)
            if rules:
                rule_id = rules[0].rule_id

        if not rule_id:
            return {
                "success": False,
                "error": "Could not identify which rule to modify",
            }

        updates = {}
        if intent.effect:
            updates["effect"] = intent.effect
        if intent.rule_actions:
            updates["actions"] = intent.rule_actions

        try:
            rule = self.rule_registry.update_rule(rule_id, updates)
            return {
                "success": True,
                "rule_id": rule_id,
                "rule": rule.to_dict(),
                "message": f"Updated rule {rule_id}",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def _delete_rule(self, intent: ParsedIntent) -> Dict[str, Any]:
        """Delete a rule."""
        rule_id = intent.conditions.get("rule_id")

        if not rule_id:
            return {
                "success": False,
                "error": "Please specify which rule to delete",
            }

        if self.rule_registry.delete_rule(rule_id):
            return {
                "success": True,
                "message": f"Deleted rule {rule_id}",
            }
        else:
            return {
                "success": False,
                "error": f"Rule not found: {rule_id}",
            }

    def _query_rules(self, intent: ParsedIntent) -> Dict[str, Any]:
        """Query existing rules."""
        if intent.target:
            rules = self.rule_registry.get_rules_for_target(intent.target)
        elif intent.scope:
            rules = self.rule_registry.list_rules(scope=intent.scope)
        else:
            rules = self.rule_registry.list_rules()

        return {
            "success": True,
            "rules": [r.to_dict() for r in rules],
            "count": len(rules),
        }

    def _check_access(self, intent: ParsedIntent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check access for a specific operation."""
        if not intent.target:
            return {
                "success": False,
                "error": "Please specify what to check access for",
            }

        action = intent.rule_actions[0] if intent.rule_actions else RuleAction.READ
        effect, rule = self.rule_registry.evaluate(intent.target, action, context)

        return {
            "success": True,
            "allowed": effect in (RuleEffect.ALLOW, RuleEffect.AUDIT),
            "effect": effect.value,
            "matching_rule": rule.to_dict() if rule else None,
            "message": f"Access {'allowed' if effect == RuleEffect.ALLOW else 'denied'} for {action.value} on {intent.target}",
        }

    def _explain_rules(self, intent: ParsedIntent) -> Dict[str, Any]:
        """Explain rules for a target."""
        if not intent.target:
            return {
                "success": False,
                "error": "Please specify what to explain",
            }

        rules = self.rule_registry.get_rules_for_target(intent.target)

        explanations = []
        for rule in rules:
            explanation = {
                "rule_id": rule.rule_id,
                "summary": rule.reason,
                "effect": rule.effect.value,
                "actions": [a.value for a in rule.actions],
                "scope": rule.scope.value,
            }
            if rule.parent_rule_id:
                explanation["inherited_from"] = rule.parent_rule_id

            explanations.append(explanation)

        return {
            "success": True,
            "target": intent.target,
            "explanations": explanations,
            "message": f"Found {len(explanations)} rules for {intent.target}",
        }

    def _suggest_rules(self, intent: ParsedIntent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest rules based on context."""
        path = intent.target or context.get("cwd", "")

        if not path:
            return {
                "success": False,
                "error": "Please specify a path for suggestions",
            }

        existing_rules = self.rule_registry.list_rules()
        suggestions = self.context.suggest_rules_for_folder(path, existing_rules)

        return {
            "success": True,
            "suggestions": [
                {
                    "rule": s["rule"].to_dict(),
                    "source": s["source_folder"],
                    "similarity": s["similarity_score"],
                    "reason": s["reason"],
                }
                for s in suggestions
            ],
            "message": f"Found {len(suggestions)} suggestions for {path}",
        }

    def _undo_last(self, user_id: Optional[str]) -> Dict[str, Any]:
        """Undo the last rule action."""
        if not user_id:
            return {
                "success": False,
                "error": "User ID required for undo",
            }

        history = self.context.get_user_history(user_id, limit=1)
        if not history:
            return {
                "success": False,
                "error": "No actions to undo",
            }

        last_action = history[0]
        rule_id = last_action.get("output_rule_id")

        if not rule_id:
            return {
                "success": False,
                "error": "Last action did not create a rule",
            }

        if self.rule_registry.delete_rule(rule_id):
            return {
                "success": True,
                "message": f"Undid rule {rule_id}",
            }
        else:
            return {
                "success": False,
                "error": f"Could not undo rule {rule_id}",
            }

    def _apply_policies(self, policies: List[Policy]) -> None:
        """Apply compiled policies to enforcement mechanisms."""
        for policy in policies:
            if isinstance(policy, FilePolicy) and self.fuse:
                # Apply FUSE policy
                self.fuse.apply_file_policies([policy])

            elif isinstance(policy, SyscallPolicy) and self.ebpf:
                # Add to seccomp filter
                filter_id = f"policy_{policy.policy_id}"
                seccomp = self.ebpf.create_seccomp_filter(filter_id)
                for syscall in policy.syscalls:
                    from .ebpf import SeccompAction

                    action = (
                        SeccompAction.ALLOW
                        if policy.action.value == "allow"
                        else SeccompAction.ERRNO
                    )
                    seccomp.add_syscall(syscall, action)

    def _handle_clarification(self, question: str, options: List[str]) -> str:
        """Handle clarification request."""
        if self._clarification_handler:
            return self._clarification_handler(question, options)
        return ""

    def _on_monitor_event(self, event: MonitorEvent) -> None:
        """Handle file monitor events."""
        # Check if event violates any rules
        action = self._event_to_action(event.event_type)
        if action:
            effect, rule = self.rule_registry.evaluate(event.full_path, action)

            if effect == RuleEffect.DENY:
                logger.warning(
                    f"Policy violation detected: {event.event_type.name} on {event.full_path}"
                )

                if self.audit_log:
                    entry = AuditEntry(
                        entry_id=f"violation_{event.event_id}",
                        timestamp=event.timestamp,
                        event_type=event.event_type.name,
                        path=event.full_path,
                        action="denied",
                        rule_id=rule.rule_id if rule else None,
                        process_id=event.process_id,
                        user_id=event.user_id,
                        success=False,
                    )
                    self.audit_log.log(entry)

        # Notify event handlers
        for handler in self._event_handlers:
            try:
                handler(
                    {
                        "type": "monitor_event",
                        "event": event.to_dict(),
                    }
                )
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    def _event_to_action(self, event_type: EventType) -> Optional[RuleAction]:
        """Convert event type to rule action."""
        mapping = {
            EventType.ACCESS: RuleAction.READ,
            EventType.OPEN: RuleAction.READ,
            EventType.MODIFY: RuleAction.WRITE,
            EventType.CLOSE_WRITE: RuleAction.WRITE,
            EventType.CREATE: RuleAction.CREATE,
            EventType.DELETE: RuleAction.DELETE,
            EventType.ATTRIB: RuleAction.CHMOD,
        }

        for et, action in mapping.items():
            if event_type & et:
                return action

        return None

    def _audit_event(self, event_data: Dict[str, Any]) -> None:
        """Handle audit events from FUSE."""
        if not self.audit_log:
            return

        entry = AuditEntry(
            entry_id=f"fuse_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.fromisoformat(
                event_data.get("timestamp", datetime.now().isoformat())
            ),
            event_type=event_data.get("operation", "unknown"),
            path=event_data.get("path", ""),
            action="audited" if event_data.get("allowed") else "denied",
            rule_id=event_data.get("rule_id"),
            process_id=event_data.get("pid"),
            user_id=event_data.get("uid"),
        )
        self.audit_log.log(entry)

    def on_event(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register an event handler."""
        self._event_handlers.append(handler)

    def set_clarification_handler(self, handler: Callable[[str, List[str]], str]) -> None:
        """Set the clarification handler."""
        self._clarification_handler = handler

    def start_monitoring(self, paths: Optional[List[str]] = None) -> None:
        """Start file system monitoring.

        Args:
            paths: Paths to monitor (uses rule targets if None)
        """
        if not self.monitor:
            logger.warning("Monitoring not enabled")
            return

        if paths is None:
            # Get paths from existing rules
            paths = list(
                set(
                    rule.target
                    for rule in self.rule_registry.list_rules()
                    if rule.scope in (RuleScope.FILE, RuleScope.FOLDER)
                )
            )

        for path in paths:
            self.monitor.add_watch(path, recursive=True)

        self.monitor.start()
        self._state = KernelState.ENFORCING
        logger.info(f"Started monitoring {len(paths)} paths")

    def stop_monitoring(self) -> None:
        """Stop file system monitoring."""
        if self.monitor:
            self.monitor.stop()
        self._state = KernelState.READY
        logger.info("Stopped monitoring")

    def get_status(self) -> Dict[str, Any]:
        """Get kernel status."""
        return {
            "state": self._state.value,
            "rules_count": len(self.rule_registry.list_rules()),
            "fuse_available": self.fuse.is_available if self.fuse else False,
            "ebpf_available": self.ebpf.ebpf_available if self.ebpf else False,
            "seccomp_available": self.ebpf.seccomp_available if self.ebpf else False,
            "monitoring_available": self.monitor.is_available if self.monitor else False,
            "fuse_mounts": len(self.fuse.list_mounts()) if self.fuse else 0,
            "active_filters": len(self.ebpf.list_filters()) if self.ebpf else 0,
        }

    def export_configuration(self) -> Dict[str, Any]:
        """Export full kernel configuration."""
        return {
            "rules": self.rule_registry.export_rules(),
            "policies": self.compiler.compile_rules(self.rule_registry.list_rules()),
            "context": {
                "analysis": self.context.analyze_patterns(),
            },
            "audit_stats": self.audit_log.get_statistics() if self.audit_log else {},
        }

    def shutdown(self) -> None:
        """Shutdown the kernel."""
        self._state = KernelState.SHUTDOWN

        if self.monitor:
            self.monitor.stop()

        if self.rule_registry:
            self.rule_registry.close()

        if self.context:
            self.context.close()

        if self.audit_log:
            self.audit_log.close()

        logger.info("Conversational kernel shut down")
