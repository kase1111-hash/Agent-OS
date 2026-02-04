"""Rule Registry for Conversational Kernel.

Provides storage and management of user-declared policies with:
- Rule definitions with scope, effect, and actions
- Conflict detection and resolution
- Rule versioning and auditing
- Inheritance from parent scopes
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class RuleEffect(str, Enum):
    """Effect of a rule when triggered."""

    ALLOW = "allow"
    DENY = "deny"
    AUDIT = "audit"
    PROMPT = "prompt"  # Ask user for confirmation
    TRANSFORM = "transform"  # Modify the operation


class RuleAction(str, Enum):
    """Actions that rules can govern."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    CREATE = "create"
    MODIFY = "modify"
    OVERWRITE = "overwrite"
    RENAME = "rename"
    COPY = "copy"
    MOVE = "move"
    LINK = "link"
    CHMOD = "chmod"
    CHOWN = "chown"
    LIST = "list"
    STAT = "stat"
    AI_READ = "ai_read"
    AI_INDEX = "ai_index"
    AI_EMBED = "ai_embed"
    NETWORK = "network"
    SYSCALL = "syscall"


class RuleScope(str, Enum):
    """Scope levels for rules."""

    SYSTEM = "system"  # Applies globally
    USER = "user"  # Applies to specific user
    FOLDER = "folder"  # Applies to directory tree
    FILE = "file"  # Applies to specific file
    AGENT = "agent"  # Applies to specific agent
    PROCESS = "process"  # Applies to process type


class RuleValidationError(Exception):
    """Error during rule validation."""


# Condition operator dispatch - returns True if condition passes
_CONDITION_OPERATORS = {
    "eq": lambda actual, value: actual == value,
    "ne": lambda actual, value: actual != value,
    "in": lambda actual, value: actual in value,
    "not_in": lambda actual, value: actual not in value,
    "contains": lambda actual, value: value in actual,
    "gt": lambda actual, value: actual > value,
    "lt": lambda actual, value: actual < value,
    "ge": lambda actual, value: actual >= value,
    "le": lambda actual, value: actual <= value,
}

    def __init__(
        self,
        message: str,
        rule_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.rule_id = rule_id
        self.details = details or {}


class RuleConflict(Exception):
    """Conflict between rules detected."""

    def __init__(
        self,
        message: str,
        conflicting_rules: List[str],
        resolution_suggestions: Optional[List[str]] = None,
    ):
        super().__init__(message)
        self.conflicting_rules = conflicting_rules
        self.resolution_suggestions = resolution_suggestions or []


@dataclass
class Rule:
    """A constitutional rule defining system behavior."""

    rule_id: str
    scope: RuleScope
    target: str  # Path, user, agent name, etc.
    effect: RuleEffect
    actions: List[RuleAction]
    reason: str
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority: int = 100  # Higher = more important
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    version: int = 1
    parent_rule_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate rule after initialization."""
        if not self.rule_id:
            self.rule_id = f"r_{uuid.uuid4().hex[:8]}"
        if isinstance(self.scope, str):
            self.scope = RuleScope(self.scope)
        if isinstance(self.effect, str):
            self.effect = RuleEffect(self.effect)
        if self.actions and isinstance(self.actions[0], str):
            self.actions = [RuleAction(a) for a in self.actions]

    def matches(
        self,
        target: str,
        action: RuleAction,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check if this rule matches the given operation.

        Args:
            target: The target path/resource
            action: The action being performed
            context: Additional context for condition evaluation

        Returns:
            True if rule matches
        """
        if not self.enabled:
            return False

        # Check action
        if action not in self.actions:
            return False

        # Check target based on scope
        if self.scope == RuleScope.SYSTEM:
            target_matches = True
        elif self.scope in (RuleScope.FOLDER, RuleScope.FILE):
            # Path matching
            target_matches = self._path_matches(target)
        elif self.scope == RuleScope.USER:
            target_matches = context and context.get("user") == self.target
        elif self.scope == RuleScope.AGENT:
            target_matches = context and context.get("agent") == self.target
        elif self.scope == RuleScope.PROCESS:
            target_matches = context and context.get("process_type") == self.target
        else:
            target_matches = False

        if not target_matches:
            return False

        # Check conditions
        if self.conditions and context:
            return self._evaluate_conditions(context)

        return True

    def _path_matches(self, target: str) -> bool:
        """Check if target path matches rule target."""
        rule_path = Path(self.target)
        target_path = Path(target)

        if self.scope == RuleScope.FILE:
            return target_path == rule_path
        else:  # FOLDER - match if target is under folder
            try:
                target_path.relative_to(rule_path)
                return True
            except ValueError:
                return target_path == rule_path

    def _evaluate_conditions(self, context: Dict[str, Any]) -> bool:
        """Evaluate rule conditions against context."""
        for key, expected in self.conditions.items():
            actual = context.get(key)

            if isinstance(expected, dict):
                # Complex condition - use operator dispatch
                op = expected.get("op", "eq")
                value = expected.get("value")
                evaluator = _CONDITION_OPERATORS.get(op)
                if evaluator and not evaluator(actual, value):
                    return False
            else:
                if actual != expected:
                    return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary."""
        return {
            "rule_id": self.rule_id,
            "scope": self.scope.value,
            "target": self.target,
            "effect": self.effect.value,
            "actions": [a.value for a in self.actions],
            "reason": self.reason,
            "conditions": self.conditions,
            "priority": self.priority,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "version": self.version,
            "parent_rule_id": self.parent_rule_id,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rule":
        """Create rule from dictionary."""
        # Convert datetime strings
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        return cls(**data)


class RuleRegistry:
    """Registry for storing and managing rules.

    Supports:
    - Persistent storage in SQLite
    - Rule versioning
    - Conflict detection
    - Inheritance resolution
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize rule registry.

        Args:
            db_path: Path to SQLite database (None for in-memory)
        """
        self.db_path = db_path
        self._rules: Dict[str, Rule] = {}
        self._rules_by_target: Dict[str, List[str]] = {}
        self._conflict_handlers: List[Callable[[RuleConflict], Optional[str]]] = []
        self._conn: Optional[sqlite3.Connection] = None

        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        db_str = str(self.db_path) if self.db_path else ":memory:"
        self._conn = sqlite3.connect(db_str, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS rules (
                rule_id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                target TEXT NOT NULL,
                effect TEXT NOT NULL,
                actions TEXT NOT NULL,
                reason TEXT,
                conditions TEXT,
                priority INTEGER DEFAULT 100,
                enabled INTEGER DEFAULT 1,
                created_at TEXT,
                updated_at TEXT,
                created_by TEXT,
                version INTEGER DEFAULT 1,
                parent_rule_id TEXT,
                tags TEXT,
                metadata TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_rules_target ON rules(target);
            CREATE INDEX IF NOT EXISTS idx_rules_scope ON rules(scope);
            CREATE INDEX IF NOT EXISTS idx_rules_enabled ON rules(enabled);

            CREATE TABLE IF NOT EXISTS rule_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                data TEXT NOT NULL,
                changed_at TEXT NOT NULL,
                changed_by TEXT,
                change_type TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_history_rule ON rule_history(rule_id);
        """
        )

        # Load existing rules
        self._load_rules()

    def _load_rules(self) -> None:
        """Load rules from database."""
        if not self._conn:
            return

        cursor = self._conn.execute("SELECT * FROM rules WHERE enabled = 1")

        for row in cursor.fetchall():
            rule = self._row_to_rule(row)
            self._rules[rule.rule_id] = rule
            self._index_rule(rule)

    def _row_to_rule(self, row: sqlite3.Row) -> Rule:
        """Convert database row to Rule object."""
        return Rule(
            rule_id=row["rule_id"],
            scope=RuleScope(row["scope"]),
            target=row["target"],
            effect=RuleEffect(row["effect"]),
            actions=[RuleAction(a) for a in json.loads(row["actions"])],
            reason=row["reason"] or "",
            conditions=json.loads(row["conditions"]) if row["conditions"] else {},
            priority=row["priority"],
            enabled=bool(row["enabled"]),
            created_at=(
                datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now()
            ),
            updated_at=(
                datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else datetime.now()
            ),
            created_by=row["created_by"] or "system",
            version=row["version"],
            parent_rule_id=row["parent_rule_id"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def _index_rule(self, rule: Rule) -> None:
        """Add rule to target index."""
        if rule.target not in self._rules_by_target:
            self._rules_by_target[rule.target] = []
        if rule.rule_id not in self._rules_by_target[rule.target]:
            self._rules_by_target[rule.target].append(rule.rule_id)

    def _unindex_rule(self, rule: Rule) -> None:
        """Remove rule from target index."""
        if rule.target in self._rules_by_target:
            if rule.rule_id in self._rules_by_target[rule.target]:
                self._rules_by_target[rule.target].remove(rule.rule_id)

    def add_rule(self, rule: Rule, check_conflicts: bool = True) -> str:
        """Add a new rule to the registry.

        Args:
            rule: The rule to add
            check_conflicts: Whether to check for conflicts

        Returns:
            The rule ID

        Raises:
            RuleConflict: If conflicting rules exist
            RuleValidationError: If rule is invalid
        """
        # Validate rule
        self._validate_rule(rule)

        # Check for conflicts
        if check_conflicts:
            conflicts = self._detect_conflicts(rule)
            if conflicts:
                raise RuleConflict(
                    f"Rule conflicts with existing rules: {conflicts}",
                    conflicting_rules=conflicts,
                    resolution_suggestions=self._suggest_resolutions(rule, conflicts),
                )

        # Save to database
        self._save_rule(rule)

        # Add to in-memory storage
        self._rules[rule.rule_id] = rule
        self._index_rule(rule)

        # Record history
        self._record_history(rule, "create")

        logger.info(f"Added rule {rule.rule_id}: {rule.reason}")
        return rule.rule_id

    def update_rule(
        self, rule_id: str, updates: Dict[str, Any], check_conflicts: bool = True
    ) -> Rule:
        """Update an existing rule.

        Args:
            rule_id: ID of rule to update
            updates: Fields to update
            check_conflicts: Whether to check for conflicts

        Returns:
            Updated rule
        """
        if rule_id not in self._rules:
            raise RuleValidationError(f"Rule not found: {rule_id}", rule_id=rule_id)

        old_rule = self._rules[rule_id]
        self._unindex_rule(old_rule)

        # Create updated rule
        rule_data = old_rule.to_dict()
        rule_data.update(updates)
        rule_data["version"] = old_rule.version + 1
        rule_data["updated_at"] = datetime.now()

        new_rule = Rule.from_dict(rule_data)

        # Check conflicts
        if check_conflicts:
            conflicts = self._detect_conflicts(new_rule, exclude=[rule_id])
            if conflicts:
                self._index_rule(old_rule)
                raise RuleConflict(
                    f"Update would conflict with: {conflicts}",
                    conflicting_rules=conflicts,
                )

        # Save and update
        self._save_rule(new_rule)
        self._rules[rule_id] = new_rule
        self._index_rule(new_rule)
        self._record_history(new_rule, "update")

        return new_rule

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule from the registry.

        Args:
            rule_id: ID of rule to delete

        Returns:
            True if deleted
        """
        if rule_id not in self._rules:
            return False

        rule = self._rules[rule_id]
        self._unindex_rule(rule)
        del self._rules[rule_id]

        # Disable in database (soft delete)
        if self._conn:
            self._conn.execute(
                "UPDATE rules SET enabled = 0, updated_at = ? WHERE rule_id = ?",
                (datetime.now().isoformat(), rule_id),
            )
            self._conn.commit()

        self._record_history(rule, "delete")
        logger.info(f"Deleted rule {rule_id}")
        return True

    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """Get a rule by ID."""
        return self._rules.get(rule_id)

    def get_rules_for_target(self, target: str) -> List[Rule]:
        """Get all rules that apply to a target.

        Includes inherited rules from parent paths.
        """
        rules = []
        target_path = Path(target)

        # Get direct rules
        if target in self._rules_by_target:
            for rule_id in self._rules_by_target[target]:
                if rule_id in self._rules:
                    rules.append(self._rules[rule_id])

        # Get inherited rules from parent paths
        for parent in target_path.parents:
            parent_str = str(parent)
            if parent_str in self._rules_by_target:
                for rule_id in self._rules_by_target[parent_str]:
                    if rule_id in self._rules:
                        rule = self._rules[rule_id]
                        if rule.scope == RuleScope.FOLDER:
                            rules.append(rule)

        # Get system-wide rules
        for rule in self._rules.values():
            if rule.scope == RuleScope.SYSTEM and rule not in rules:
                rules.append(rule)

        # Sort by priority (higher first)
        rules.sort(key=lambda r: r.priority, reverse=True)
        return rules

    def find_matching_rules(
        self,
        target: str,
        action: RuleAction,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Rule]:
        """Find all rules matching an operation.

        Args:
            target: The target path/resource
            action: The action being performed
            context: Additional context

        Returns:
            Matching rules sorted by priority
        """
        candidates = self.get_rules_for_target(target)
        matching = [r for r in candidates if r.matches(target, action, context)]
        return matching

    def evaluate(
        self,
        target: str,
        action: RuleAction,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[RuleEffect, Optional[Rule]]:
        """Evaluate rules for an operation.

        Args:
            target: The target path/resource
            action: The action being performed
            context: Additional context

        Returns:
            Tuple of (effect, matching_rule)
        """
        matching = self.find_matching_rules(target, action, context)

        if not matching:
            return RuleEffect.ALLOW, None

        # Return highest priority matching rule's effect
        rule = matching[0]
        return rule.effect, rule

    def _validate_rule(self, rule: Rule) -> None:
        """Validate a rule."""
        if not rule.target:
            raise RuleValidationError("Rule must have a target", rule_id=rule.rule_id)

        if not rule.actions:
            raise RuleValidationError("Rule must have at least one action", rule_id=rule.rule_id)

        # Validate path for file/folder scope
        if rule.scope in (RuleScope.FILE, RuleScope.FOLDER):
            path = Path(rule.target)
            if not path.is_absolute():
                raise RuleValidationError(
                    f"Target must be absolute path for {rule.scope.value} scope",
                    rule_id=rule.rule_id,
                )

    def _detect_conflicts(self, rule: Rule, exclude: Optional[List[str]] = None) -> List[str]:
        """Detect conflicting rules."""
        exclude = exclude or []
        conflicts = []

        existing = self.get_rules_for_target(rule.target)

        for other in existing:
            if other.rule_id in exclude:
                continue

            # Check for conflicting effects on same actions
            common_actions = set(rule.actions) & set(other.actions)
            if common_actions and rule.effect != other.effect:
                # Same target, same actions, different effects = conflict
                if rule.scope == other.scope or rule.priority == other.priority:
                    conflicts.append(other.rule_id)

        return conflicts

    def _suggest_resolutions(self, rule: Rule, conflicts: List[str]) -> List[str]:
        """Suggest ways to resolve conflicts."""
        suggestions = []

        for conflict_id in conflicts:
            conflict = self.get_rule(conflict_id)
            if not conflict:
                continue

            if rule.priority == conflict.priority:
                suggestions.append(
                    f"Change priority of new rule or {conflict_id} to establish precedence"
                )

            if set(rule.actions) == set(conflict.actions):
                suggestions.append(
                    f"Consider disabling {conflict_id} if this rule should replace it"
                )

            if rule.scope == conflict.scope == RuleScope.FOLDER:
                suggestions.append("Consider using more specific paths to avoid overlap")

        return suggestions

    def _save_rule(self, rule: Rule) -> None:
        """Save rule to database."""
        if not self._conn:
            return

        self._conn.execute(
            """
            INSERT OR REPLACE INTO rules (
                rule_id, scope, target, effect, actions, reason,
                conditions, priority, enabled, created_at, updated_at,
                created_by, version, parent_rule_id, tags, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rule.rule_id,
                rule.scope.value,
                rule.target,
                rule.effect.value,
                json.dumps([a.value for a in rule.actions]),
                rule.reason,
                json.dumps(rule.conditions),
                rule.priority,
                1 if rule.enabled else 0,
                rule.created_at.isoformat(),
                rule.updated_at.isoformat(),
                rule.created_by,
                rule.version,
                rule.parent_rule_id,
                json.dumps(rule.tags),
                json.dumps(rule.metadata),
            ),
        )
        self._conn.commit()

    def _record_history(self, rule: Rule, change_type: str) -> None:
        """Record rule change in history."""
        if not self._conn:
            return

        self._conn.execute(
            """
            INSERT INTO rule_history (rule_id, version, data, changed_at, changed_by, change_type)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                rule.rule_id,
                rule.version,
                json.dumps(rule.to_dict()),
                datetime.now().isoformat(),
                rule.created_by,
                change_type,
            ),
        )
        self._conn.commit()

    def get_history(self, rule_id: str) -> List[Dict[str, Any]]:
        """Get version history for a rule."""
        if not self._conn:
            return []

        cursor = self._conn.execute(
            "SELECT * FROM rule_history WHERE rule_id = ? ORDER BY version DESC",
            (rule_id,),
        )

        return [
            {
                "version": row["version"],
                "data": json.loads(row["data"]),
                "changed_at": row["changed_at"],
                "changed_by": row["changed_by"],
                "change_type": row["change_type"],
            }
            for row in cursor.fetchall()
        ]

    def list_rules(
        self,
        scope: Optional[RuleScope] = None,
        enabled_only: bool = True,
        tags: Optional[List[str]] = None,
    ) -> List[Rule]:
        """List rules with optional filtering."""
        rules = list(self._rules.values())

        if enabled_only:
            rules = [r for r in rules if r.enabled]

        if scope:
            rules = [r for r in rules if r.scope == scope]

        if tags:
            rules = [r for r in rules if any(t in r.tags for t in tags)]

        return sorted(rules, key=lambda r: (r.scope.value, r.target))

    def export_rules(self) -> List[Dict[str, Any]]:
        """Export all rules as dictionaries."""
        return [r.to_dict() for r in self._rules.values()]

    def import_rules(self, rules_data: List[Dict[str, Any]], overwrite: bool = False) -> int:
        """Import rules from dictionaries.

        Args:
            rules_data: List of rule dictionaries
            overwrite: Whether to overwrite existing rules

        Returns:
            Number of rules imported
        """
        count = 0

        for data in rules_data:
            rule = Rule.from_dict(data)

            if rule.rule_id in self._rules and not overwrite:
                continue

            try:
                if rule.rule_id in self._rules:
                    self.update_rule(rule.rule_id, data, check_conflicts=False)
                else:
                    self.add_rule(rule, check_conflicts=False)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to import rule {rule.rule_id}: {e}")

        return count

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def on_conflict(self, handler: Callable[[RuleConflict], Optional[str]]) -> None:
        """Register a conflict resolution handler.

        Handler receives conflict and returns rule_id to keep, or None.
        """
        self._conflict_handlers.append(handler)
