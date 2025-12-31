"""Context Memory for Conversational Kernel.

Maintains user preferences, rule history, and contextual information
for intelligent policy suggestions and reasoning.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .rules import Rule, RuleScope

logger = logging.getLogger(__name__)


@dataclass
class UserContext:
    """Context for a specific user."""

    user_id: str
    username: str
    home_directory: str = ""
    shell: str = "/bin/bash"
    groups: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    rule_history: List[str] = field(default_factory=list)
    last_active: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "home_directory": self.home_directory,
            "shell": self.shell,
            "groups": self.groups,
            "preferences": self.preferences,
            "rule_history": self.rule_history,
            "last_active": self.last_active.isoformat(),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserContext":
        """Create from dictionary."""
        if isinstance(data.get("last_active"), str):
            data["last_active"] = datetime.fromisoformat(data["last_active"])
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class FolderContext:
    """Context for a specific folder."""

    path: str
    rule_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    description: str = ""
    owner: str = ""
    sensitivity: str = "normal"  # normal, sensitive, confidential
    inherited_from: Optional[str] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "rule_ids": self.rule_ids,
            "tags": self.tags,
            "description": self.description,
            "owner": self.owner,
            "sensitivity": self.sensitivity,
            "inherited_from": self.inherited_from,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FolderContext":
        """Create from dictionary."""
        if isinstance(data.get("last_accessed"), str):
            data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class AgentContext:
    """Context for an AI agent."""

    agent_id: str
    agent_name: str
    agent_type: str = "general"
    capabilities: List[str] = field(default_factory=list)
    restrictions: List[str] = field(default_factory=list)
    allowed_paths: List[str] = field(default_factory=list)
    denied_paths: List[str] = field(default_factory=list)
    max_file_size: int = 0  # 0 = no limit
    can_network: bool = False
    can_execute: bool = False
    trust_level: int = 50  # 0-100
    last_active: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "restrictions": self.restrictions,
            "allowed_paths": self.allowed_paths,
            "denied_paths": self.denied_paths,
            "max_file_size": self.max_file_size,
            "can_network": self.can_network,
            "can_execute": self.can_execute,
            "trust_level": self.trust_level,
            "last_active": self.last_active.isoformat(),
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentContext":
        """Create from dictionary."""
        if isinstance(data.get("last_active"), str):
            data["last_active"] = datetime.fromisoformat(data["last_active"])
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class ContextMemory:
    """Persistent context memory for the conversational kernel.

    Stores and retrieves contextual information including:
    - User preferences and history
    - Folder-specific context
    - Agent configurations
    - Rule patterns for suggestions
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize context memory.

        Args:
            db_path: Path to SQLite database (None for in-memory)
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._users: Dict[str, UserContext] = {}
        self._folders: Dict[str, FolderContext] = {}
        self._agents: Dict[str, AgentContext] = {}
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database."""
        db_str = str(self.db_path) if self.db_path else ":memory:"
        self._conn = sqlite3.connect(db_str, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS user_context (
                user_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS folder_context (
                path TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS agent_context (
                agent_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS rule_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                usage_count INTEGER DEFAULT 0,
                last_used TEXT
            );

            CREATE TABLE IF NOT EXISTS interaction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                interaction_type TEXT,
                input_text TEXT,
                output_rule_id TEXT,
                context_data TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_folder_path ON folder_context(path);
            CREATE INDEX IF NOT EXISTS idx_history_user ON interaction_history(user_id);
            CREATE INDEX IF NOT EXISTS idx_history_time ON interaction_history(timestamp);
        """
        )

        # Load existing context
        self._load_context()

    def _load_context(self) -> None:
        """Load context from database."""
        if not self._conn:
            return

        # Load users
        cursor = self._conn.execute("SELECT data FROM user_context")
        for row in cursor.fetchall():
            user = UserContext.from_dict(json.loads(row["data"]))
            self._users[user.user_id] = user

        # Load folders
        cursor = self._conn.execute("SELECT data FROM folder_context")
        for row in cursor.fetchall():
            folder = FolderContext.from_dict(json.loads(row["data"]))
            self._folders[folder.path] = folder

        # Load agents
        cursor = self._conn.execute("SELECT data FROM agent_context")
        for row in cursor.fetchall():
            agent = AgentContext.from_dict(json.loads(row["data"]))
            self._agents[agent.agent_id] = agent

    def get_user(self, user_id: str) -> Optional[UserContext]:
        """Get user context."""
        return self._users.get(user_id)

    def set_user(self, context: UserContext) -> None:
        """Set user context."""
        self._users[context.user_id] = context
        self._save_user(context)

    def _save_user(self, context: UserContext) -> None:
        """Save user context to database."""
        if not self._conn:
            return

        self._conn.execute(
            """
            INSERT OR REPLACE INTO user_context (user_id, data, updated_at)
            VALUES (?, ?, ?)
            """,
            (context.user_id, json.dumps(context.to_dict()), datetime.now().isoformat()),
        )
        self._conn.commit()

    def get_folder(self, path: str) -> Optional[FolderContext]:
        """Get folder context."""
        return self._folders.get(path)

    def set_folder(self, context: FolderContext) -> None:
        """Set folder context."""
        self._folders[context.path] = context
        self._save_folder(context)

    def _save_folder(self, context: FolderContext) -> None:
        """Save folder context to database."""
        if not self._conn:
            return

        self._conn.execute(
            """
            INSERT OR REPLACE INTO folder_context (path, data, updated_at)
            VALUES (?, ?, ?)
            """,
            (context.path, json.dumps(context.to_dict()), datetime.now().isoformat()),
        )
        self._conn.commit()

    def get_agent(self, agent_id: str) -> Optional[AgentContext]:
        """Get agent context."""
        return self._agents.get(agent_id)

    def set_agent(self, context: AgentContext) -> None:
        """Set agent context."""
        self._agents[context.agent_id] = context
        self._save_agent(context)

    def _save_agent(self, context: AgentContext) -> None:
        """Save agent context to database."""
        if not self._conn:
            return

        self._conn.execute(
            """
            INSERT OR REPLACE INTO agent_context (agent_id, data, updated_at)
            VALUES (?, ?, ?)
            """,
            (context.agent_id, json.dumps(context.to_dict()), datetime.now().isoformat()),
        )
        self._conn.commit()

    def get_inherited_context(self, path: str) -> List[FolderContext]:
        """Get context from parent folders.

        Args:
            path: Path to get inherited context for

        Returns:
            List of parent folder contexts, nearest first
        """
        contexts = []
        current = Path(path)

        while current != current.parent:
            current = current.parent
            if str(current) in self._folders:
                contexts.append(self._folders[str(current)])

        return contexts

    def find_similar_folders(self, context: FolderContext) -> List[Tuple[FolderContext, float]]:
        """Find folders with similar rules/tags.

        Args:
            context: Folder context to compare

        Returns:
            List of (folder_context, similarity_score) tuples
        """
        similar = []

        for folder in self._folders.values():
            if folder.path == context.path:
                continue

            score = 0.0

            # Compare tags
            if context.tags and folder.tags:
                common_tags = set(context.tags) & set(folder.tags)
                if common_tags:
                    score += len(common_tags) / max(len(context.tags), len(folder.tags))

            # Compare sensitivity
            if context.sensitivity == folder.sensitivity:
                score += 0.2

            # Compare rule count
            if context.rule_ids and folder.rule_ids:
                score += 0.1 * min(len(context.rule_ids), len(folder.rule_ids))

            if score > 0.3:
                similar.append((folder, score))

        # Sort by score descending
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar[:10]

    def record_interaction(
        self,
        user_id: str,
        interaction_type: str,
        input_text: str,
        output_rule_id: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a user interaction.

        Args:
            user_id: User who interacted
            interaction_type: Type of interaction
            input_text: User input
            output_rule_id: Resulting rule ID if any
            context_data: Additional context
        """
        if not self._conn:
            return

        self._conn.execute(
            """
            INSERT INTO interaction_history
            (timestamp, user_id, interaction_type, input_text, output_rule_id, context_data)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(),
                user_id,
                interaction_type,
                input_text,
                output_rule_id,
                json.dumps(context_data) if context_data else None,
            ),
        )
        self._conn.commit()

    def get_user_history(
        self,
        user_id: str,
        limit: int = 50,
        interaction_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get user interaction history.

        Args:
            user_id: User to get history for
            limit: Maximum entries
            interaction_type: Filter by type

        Returns:
            List of interaction records
        """
        if not self._conn:
            return []

        query = """
            SELECT * FROM interaction_history
            WHERE user_id = ?
        """
        params: List[Any] = [user_id]

        if interaction_type:
            query += " AND interaction_type = ?"
            params.append(interaction_type)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(query, params)

        return [
            {
                "timestamp": row["timestamp"],
                "interaction_type": row["interaction_type"],
                "input_text": row["input_text"],
                "output_rule_id": row["output_rule_id"],
                "context_data": json.loads(row["context_data"]) if row["context_data"] else None,
            }
            for row in cursor.fetchall()
        ]

    def suggest_rules_for_folder(
        self,
        path: str,
        existing_rules: List[Rule],
    ) -> List[Dict[str, Any]]:
        """Suggest rules based on similar folders.

        Args:
            path: Path to suggest rules for
            existing_rules: Existing rules in registry

        Returns:
            List of rule suggestions
        """
        suggestions = []
        folder_context = self.get_folder(path)

        if not folder_context:
            folder_context = FolderContext(path=path)

        # Find similar folders
        similar = self.find_similar_folders(folder_context)

        for similar_folder, score in similar:
            # Get rules for similar folder
            for rule_id in similar_folder.rule_ids:
                for rule in existing_rules:
                    if rule.rule_id == rule_id:
                        suggestions.append(
                            {
                                "rule": rule,
                                "source_folder": similar_folder.path,
                                "similarity_score": score,
                                "reason": f"Similar to rule for {similar_folder.path}",
                            }
                        )
                        break

        return suggestions

    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze rule patterns across the system.

        Returns:
            Analysis results
        """
        analysis = {
            "total_users": len(self._users),
            "total_folders": len(self._folders),
            "total_agents": len(self._agents),
            "common_tags": {},
            "sensitivity_distribution": {},
            "folders_by_depth": {},
        }

        # Count tags
        tag_counts: Dict[str, int] = {}
        for folder in self._folders.values():
            for tag in folder.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        analysis["common_tags"] = dict(
            sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )

        # Sensitivity distribution
        for folder in self._folders.values():
            sensitivity = folder.sensitivity
            analysis["sensitivity_distribution"][sensitivity] = (
                analysis["sensitivity_distribution"].get(sensitivity, 0) + 1
            )

        # Folder depth distribution
        for folder in self._folders.values():
            depth = len(Path(folder.path).parts)
            analysis["folders_by_depth"][depth] = analysis["folders_by_depth"].get(depth, 0) + 1

        return analysis

    def get_context_for_evaluation(
        self,
        path: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get full context for rule evaluation.

        Args:
            path: File/folder path
            user_id: Optional user ID
            agent_id: Optional agent ID

        Returns:
            Combined context dictionary
        """
        context: Dict[str, Any] = {"path": path}

        # Add folder context
        folder = self.get_folder(path)
        if folder:
            context["folder"] = folder.to_dict()
            context["sensitivity"] = folder.sensitivity
            context["folder_tags"] = folder.tags

        # Add inherited context
        inherited = self.get_inherited_context(path)
        if inherited:
            context["inherited_sensitivity"] = max(
                (f.sensitivity for f in inherited),
                key=lambda s: (
                    ["normal", "sensitive", "confidential"].index(s)
                    if s in ["normal", "sensitive", "confidential"]
                    else 0
                ),
            )

        # Add user context
        if user_id:
            user = self.get_user(user_id)
            if user:
                context["user"] = user.username
                context["user_groups"] = user.groups
                context["user_preferences"] = user.preferences

        # Add agent context
        if agent_id:
            agent = self.get_agent(agent_id)
            if agent:
                context["agent"] = agent.agent_name
                context["agent_type"] = agent.agent_type
                context["trust_level"] = agent.trust_level
                context["agent_can_network"] = agent.can_network
                context["agent_can_execute"] = agent.can_execute

        return context

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
