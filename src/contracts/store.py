"""
Learning Contracts Store

Manages learning contracts with support for active, expired, and revoked states.
Contracts control what data can be learned/stored by AI systems.
"""

import hashlib
import json
import logging
import secrets
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set


logger = logging.getLogger(__name__)


class ContractStatus(Enum):
    """Status of a learning contract."""
    PENDING = auto()      # Contract awaiting user consent
    ACTIVE = auto()       # Contract is active and enforceable
    EXPIRED = auto()      # Contract has expired (time-based)
    REVOKED = auto()      # Contract was revoked by user
    SUPERSEDED = auto()   # Contract replaced by newer version
    SUSPENDED = auto()    # Temporarily suspended


class ContractType(Enum):
    """
    Type of learning contract.

    Aligned with learning-contracts specification:
    https://github.com/kase1111-hash/learning-contracts
    """
    # Core contract types from learning-contracts spec
    OBSERVATION = auto()    # Permits watching signals only; forbids storage/generalization
    EPISODIC = auto()       # Allows storing specific instances without cross-context application
    PROCEDURAL = auto()     # Enables deriving reusable heuristics within defined scopes
    STRATEGIC = auto()      # Permits long-term pattern inference (high-trust only)
    PROHIBITED = auto()     # Explicitly blocks learning in sensitive domains

    # Legacy types (mapped for backwards compatibility)
    FULL_CONSENT = auto()         # -> Maps to STRATEGIC
    LIMITED_CONSENT = auto()      # -> Maps to PROCEDURAL
    ABSTRACTION_ONLY = auto()     # -> Maps to EPISODIC
    NO_LEARNING = auto()          # -> Maps to PROHIBITED
    TEMPORARY = auto()            # Time-limited (any type)
    SESSION_ONLY = auto()         # Session-scoped (any type)

    @classmethod
    def from_legacy(cls, legacy_type: "ContractType") -> "ContractType":
        """Map legacy contract types to learning-contracts types."""
        mapping = {
            cls.FULL_CONSENT: cls.STRATEGIC,
            cls.LIMITED_CONSENT: cls.PROCEDURAL,
            cls.ABSTRACTION_ONLY: cls.EPISODIC,
            cls.NO_LEARNING: cls.PROHIBITED,
        }
        return mapping.get(legacy_type, legacy_type)

    def allows_storage(self) -> bool:
        """Check if this contract type allows any storage."""
        return self not in [self.OBSERVATION, self.PROHIBITED, self.NO_LEARNING]

    def allows_generalization(self) -> bool:
        """Check if this contract type allows generalization/abstraction."""
        return self in [self.PROCEDURAL, self.STRATEGIC, self.LIMITED_CONSENT, self.FULL_CONSENT]

    def allows_cross_context(self) -> bool:
        """Check if this contract type allows cross-context application."""
        return self in [self.PROCEDURAL, self.STRATEGIC, self.FULL_CONSENT]

    def allows_long_term_patterns(self) -> bool:
        """Check if this contract type allows long-term pattern inference."""
        return self in [self.STRATEGIC, self.FULL_CONSENT]


class LearningScope(Enum):
    """Scope of what can be learned."""
    ALL = auto()                   # All interactions
    DOMAIN_SPECIFIC = auto()       # Specific domains only
    TASK_SPECIFIC = auto()         # Specific task types
    CONTENT_SPECIFIC = auto()      # Specific content categories
    AGENT_SPECIFIC = auto()        # Specific agent interactions


@dataclass
class ContractScope:
    """Defines the scope of a learning contract."""
    scope_type: LearningScope
    domains: Set[str] = field(default_factory=set)
    tasks: Set[str] = field(default_factory=set)
    agents: Set[str] = field(default_factory=set)
    content_types: Set[str] = field(default_factory=set)
    excluded_domains: Set[str] = field(default_factory=set)

    def matches(self, domain: str = "", task: str = "", agent: str = "", content_type: str = "") -> bool:
        """Check if the given parameters match this scope."""
        # Check exclusions first
        if domain and domain in self.excluded_domains:
            return False

        if self.scope_type == LearningScope.ALL:
            return True
        elif self.scope_type == LearningScope.DOMAIN_SPECIFIC:
            return domain in self.domains if domain else not self.domains
        elif self.scope_type == LearningScope.TASK_SPECIFIC:
            return task in self.tasks if task else not self.tasks
        elif self.scope_type == LearningScope.AGENT_SPECIFIC:
            return agent in self.agents if agent else not self.agents
        elif self.scope_type == LearningScope.CONTENT_SPECIFIC:
            return content_type in self.content_types if content_type else not self.content_types

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scope_type": self.scope_type.name,
            "domains": list(self.domains),
            "tasks": list(self.tasks),
            "agents": list(self.agents),
            "content_types": list(self.content_types),
            "excluded_domains": list(self.excluded_domains),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContractScope":
        """Create from dictionary."""
        return cls(
            scope_type=LearningScope[data["scope_type"]],
            domains=set(data.get("domains", [])),
            tasks=set(data.get("tasks", [])),
            agents=set(data.get("agents", [])),
            content_types=set(data.get("content_types", [])),
            excluded_domains=set(data.get("excluded_domains", [])),
        )


@dataclass
class LearningContract:
    """A learning contract defining consent for AI learning."""
    contract_id: str
    user_id: str
    contract_type: ContractType
    scope: ContractScope
    status: ContractStatus = ContractStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    activated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    version: int = 1
    previous_contract_id: Optional[str] = None
    description: str = ""
    consent_method: str = "explicit"  # explicit, implicit, inherited
    revocation_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    signature_hash: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if contract is currently valid."""
        if self.status != ContractStatus.ACTIVE:
            return False

        if self.expires_at and datetime.now() > self.expires_at:
            return False

        return True

    def allows_learning(self, domain: str = "", task: str = "", agent: str = "") -> bool:
        """Check if this contract allows learning for given parameters."""
        if not self.is_valid():
            return False

        if self.contract_type == ContractType.NO_LEARNING:
            return False

        return self.scope.matches(domain=domain, task=task, agent=agent)

    def allows_raw_storage(self) -> bool:
        """Check if raw data storage is allowed."""
        if not self.is_valid():
            return False

        return self.contract_type.allows_storage()

    def allows_generalization(self) -> bool:
        """Check if generalization/abstraction is allowed."""
        if not self.is_valid():
            return False

        return self.contract_type.allows_generalization()

    def allows_cross_context(self) -> bool:
        """Check if cross-context application is allowed."""
        if not self.is_valid():
            return False

        return self.contract_type.allows_cross_context()

    def allows_long_term_patterns(self) -> bool:
        """Check if long-term pattern inference is allowed."""
        if not self.is_valid():
            return False

        return self.contract_type.allows_long_term_patterns()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "contract_id": self.contract_id,
            "user_id": self.user_id,
            "contract_type": self.contract_type.name,
            "scope": self.scope.to_dict(),
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "version": self.version,
            "previous_contract_id": self.previous_contract_id,
            "description": self.description,
            "consent_method": self.consent_method,
            "revocation_reason": self.revocation_reason,
            "metadata": self.metadata,
            "signature_hash": self.signature_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningContract":
        """Create from dictionary."""
        return cls(
            contract_id=data["contract_id"],
            user_id=data["user_id"],
            contract_type=ContractType[data["contract_type"]],
            scope=ContractScope.from_dict(data["scope"]),
            status=ContractStatus[data["status"]],
            created_at=datetime.fromisoformat(data["created_at"]),
            activated_at=datetime.fromisoformat(data["activated_at"]) if data.get("activated_at") else None,
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            revoked_at=datetime.fromisoformat(data["revoked_at"]) if data.get("revoked_at") else None,
            version=data.get("version", 1),
            previous_contract_id=data.get("previous_contract_id"),
            description=data.get("description", ""),
            consent_method=data.get("consent_method", "explicit"),
            revocation_reason=data.get("revocation_reason"),
            metadata=data.get("metadata", {}),
            signature_hash=data.get("signature_hash"),
        )

    def compute_signature(self) -> str:
        """Compute signature hash for the contract."""
        content = f"{self.contract_id}:{self.user_id}:{self.contract_type.name}:{self.created_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class ContractQuery:
    """Query parameters for finding contracts."""
    user_id: Optional[str] = None
    status: Optional[ContractStatus] = None
    statuses: Optional[Set[ContractStatus]] = None
    contract_type: Optional[ContractType] = None
    domain: Optional[str] = None
    agent: Optional[str] = None
    active_only: bool = False
    include_expired: bool = False
    limit: int = 100


class ContractStore:
    """
    Persistent store for learning contracts.

    Uses SQLite for storage with support for:
    - Contract lifecycle management
    - Status transitions
    - Query by various criteria
    - Audit logging
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        default_deny: bool = True,
    ):
        """
        Initialize contract store.

        Args:
            db_path: Path to SQLite database (None for in-memory)
            default_deny: Default to denying learning if no contract exists
        """
        self.db_path = db_path
        self.default_deny = default_deny
        self._lock = threading.Lock()
        self._connection: Optional[sqlite3.Connection] = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the contract store."""
        try:
            db_str = str(self.db_path) if self.db_path else ":memory:"
            self._connection = sqlite3.connect(db_str, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row

            self._create_tables()
            self._initialized = True
            logger.info("Contract store initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize contract store: {e}")
            return False

    def close(self) -> None:
        """Close the store."""
        if self._connection:
            self._connection.close()
            self._connection = None
        self._initialized = False

    def create_contract(
        self,
        user_id: str,
        contract_type: ContractType,
        scope: ContractScope,
        duration: Optional[timedelta] = None,
        description: str = "",
        consent_method: str = "explicit",
        metadata: Optional[Dict[str, Any]] = None,
        auto_activate: bool = False,
    ) -> LearningContract:
        """
        Create a new learning contract.

        Args:
            user_id: User creating the contract
            contract_type: Type of contract
            scope: Scope of the contract
            duration: Optional duration before expiry
            description: Contract description
            consent_method: How consent was obtained
            metadata: Additional metadata
            auto_activate: Automatically activate contract

        Returns:
            Created LearningContract
        """
        if not self._initialized:
            raise RuntimeError("Contract store not initialized")

        contract_id = f"LC-{secrets.token_hex(8)}"

        contract = LearningContract(
            contract_id=contract_id,
            user_id=user_id,
            contract_type=contract_type,
            scope=scope,
            status=ContractStatus.PENDING,
            description=description,
            consent_method=consent_method,
            metadata=metadata or {},
        )

        if duration:
            contract.expires_at = datetime.now() + duration

        contract.signature_hash = contract.compute_signature()

        with self._lock:
            self._insert_contract(contract)
            self._log_event(contract_id, "created", user_id)

        if auto_activate:
            self.activate_contract(contract_id, user_id)
            contract.status = ContractStatus.ACTIVE
            contract.activated_at = datetime.now()

        return contract

    def activate_contract(
        self,
        contract_id: str,
        activated_by: str,
    ) -> bool:
        """
        Activate a pending contract.

        Args:
            contract_id: Contract to activate
            activated_by: Who is activating

        Returns:
            True if activation successful
        """
        with self._lock:
            contract = self._get_contract(contract_id)
            if not contract:
                return False

            if contract.status != ContractStatus.PENDING:
                logger.warning(f"Cannot activate contract in {contract.status} status")
                return False

            cursor = self._connection.cursor()
            cursor.execute("""
                UPDATE contracts
                SET status = ?, activated_at = ?
                WHERE contract_id = ?
            """, (ContractStatus.ACTIVE.name, datetime.now().isoformat(), contract_id))
            self._connection.commit()

            self._log_event(contract_id, "activated", activated_by)

        return True

    def revoke_contract(
        self,
        contract_id: str,
        revoked_by: str,
        reason: str = "",
    ) -> bool:
        """
        Revoke a contract.

        Args:
            contract_id: Contract to revoke
            revoked_by: Who is revoking
            reason: Reason for revocation

        Returns:
            True if revocation successful
        """
        with self._lock:
            contract = self._get_contract(contract_id)
            if not contract:
                return False

            if contract.status in [ContractStatus.REVOKED, ContractStatus.EXPIRED]:
                logger.warning(f"Contract already in terminal state: {contract.status}")
                return False

            cursor = self._connection.cursor()
            cursor.execute("""
                UPDATE contracts
                SET status = ?, revoked_at = ?, revocation_reason = ?
                WHERE contract_id = ?
            """, (ContractStatus.REVOKED.name, datetime.now().isoformat(), reason, contract_id))
            self._connection.commit()

            self._log_event(contract_id, "revoked", revoked_by, reason)

        return True

    def expire_contract(self, contract_id: str) -> bool:
        """Mark a contract as expired."""
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute("""
                UPDATE contracts
                SET status = ?
                WHERE contract_id = ? AND status = ?
            """, (ContractStatus.EXPIRED.name, contract_id, ContractStatus.ACTIVE.name))
            self._connection.commit()

            if cursor.rowcount > 0:
                self._log_event(contract_id, "expired", "system")
                return True

        return False

    def suspend_contract(
        self,
        contract_id: str,
        suspended_by: str,
        reason: str = "",
    ) -> bool:
        """Suspend a contract temporarily."""
        with self._lock:
            contract = self._get_contract(contract_id)
            if not contract:
                return False

            if contract.status != ContractStatus.ACTIVE:
                return False

            cursor = self._connection.cursor()
            cursor.execute("""
                UPDATE contracts
                SET status = ?
                WHERE contract_id = ?
            """, (ContractStatus.SUSPENDED.name, contract_id))
            self._connection.commit()

            self._log_event(contract_id, "suspended", suspended_by, reason)

        return True

    def resume_contract(
        self,
        contract_id: str,
        resumed_by: str,
    ) -> bool:
        """Resume a suspended contract."""
        with self._lock:
            contract = self._get_contract(contract_id)
            if not contract:
                return False

            if contract.status != ContractStatus.SUSPENDED:
                return False

            cursor = self._connection.cursor()
            cursor.execute("""
                UPDATE contracts
                SET status = ?
                WHERE contract_id = ?
            """, (ContractStatus.ACTIVE.name, contract_id))
            self._connection.commit()

            self._log_event(contract_id, "resumed", resumed_by)

        return True

    def get_contract(self, contract_id: str) -> Optional[LearningContract]:
        """Get a contract by ID."""
        with self._lock:
            return self._get_contract(contract_id)

    def query_contracts(self, query: ContractQuery) -> List[LearningContract]:
        """Query contracts based on criteria."""
        with self._lock:
            return self._query_contracts(query)

    def get_active_contracts(self, user_id: str) -> List[LearningContract]:
        """Get all active contracts for a user."""
        query = ContractQuery(
            user_id=user_id,
            status=ContractStatus.ACTIVE,
            active_only=True,
        )
        return self.query_contracts(query)

    def get_valid_contract(
        self,
        user_id: str,
        domain: str = "",
        task: str = "",
        agent: str = "",
    ) -> Optional[LearningContract]:
        """
        Get a valid contract for the given parameters.

        Returns the most specific matching contract or None if no valid contract.
        """
        contracts = self.get_active_contracts(user_id)

        # Check for expired contracts and expire them
        for contract in contracts:
            if contract.expires_at and datetime.now() > contract.expires_at:
                self.expire_contract(contract.contract_id)

        # Filter to valid contracts
        valid = [c for c in contracts if c.is_valid()]

        # Find matching contracts
        matching = [
            c for c in valid
            if c.allows_learning(domain=domain, task=task, agent=agent)
        ]

        if not matching:
            return None

        # Return most specific match (most recent if tie)
        matching.sort(key=lambda c: (
            c.scope.scope_type != LearningScope.ALL,  # Prefer specific scopes
            c.version,
            c.created_at,
        ), reverse=True)

        return matching[0]

    def check_learning_allowed(
        self,
        user_id: str,
        domain: str = "",
        task: str = "",
        agent: str = "",
    ) -> tuple[bool, Optional[LearningContract]]:
        """
        Check if learning is allowed for the given parameters.

        Returns:
            Tuple of (allowed, contract)
        """
        contract = self.get_valid_contract(
            user_id=user_id,
            domain=domain,
            task=task,
            agent=agent,
        )

        if contract:
            allowed = contract.contract_type != ContractType.NO_LEARNING
            return allowed, contract

        # No contract - use default policy
        return not self.default_deny, None

    def supersede_contract(
        self,
        old_contract_id: str,
        new_contract: LearningContract,
        superseded_by: str,
    ) -> bool:
        """Supersede an old contract with a new one."""
        with self._lock:
            old = self._get_contract(old_contract_id)
            if not old:
                return False

            # Mark old contract as superseded
            cursor = self._connection.cursor()
            cursor.execute("""
                UPDATE contracts
                SET status = ?
                WHERE contract_id = ?
            """, (ContractStatus.SUPERSEDED.name, old_contract_id))

            # Update new contract
            new_contract.previous_contract_id = old_contract_id
            new_contract.version = old.version + 1

            self._insert_contract(new_contract)
            self._connection.commit()

            self._log_event(old_contract_id, "superseded", superseded_by)
            self._log_event(new_contract.contract_id, "created", superseded_by)

        return True

    def get_contract_history(self, contract_id: str) -> List[LearningContract]:
        """Get the history chain of a contract."""
        history = []
        current_id = contract_id

        while current_id:
            contract = self.get_contract(current_id)
            if not contract:
                break
            history.append(contract)
            current_id = contract.previous_contract_id

        return list(reversed(history))

    def cleanup_expired(self) -> int:
        """Clean up expired contracts. Returns count of expired contracts."""
        count = 0
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute("""
                SELECT contract_id FROM contracts
                WHERE status = ? AND expires_at < ?
            """, (ContractStatus.ACTIVE.name, datetime.now().isoformat()))

            expired = cursor.fetchall()
            for row in expired:
                self.expire_contract(row[0])
                count += 1

        return count

    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics."""
        with self._lock:
            cursor = self._connection.cursor()

            stats = {"by_status": {}, "by_type": {}}

            cursor.execute("""
                SELECT status, COUNT(*) FROM contracts GROUP BY status
            """)
            for row in cursor.fetchall():
                stats["by_status"][row[0]] = row[1]

            cursor.execute("""
                SELECT contract_type, COUNT(*) FROM contracts GROUP BY contract_type
            """)
            for row in cursor.fetchall():
                stats["by_type"][row[0]] = row[1]

            cursor.execute("SELECT COUNT(*) FROM contracts")
            stats["total"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM contract_events")
            stats["total_events"] = cursor.fetchone()[0]

        return stats

    def _create_tables(self) -> None:
        """Create database tables."""
        cursor = self._connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contracts (
                contract_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                contract_type TEXT NOT NULL,
                scope_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                activated_at TEXT,
                expires_at TEXT,
                revoked_at TEXT,
                version INTEGER DEFAULT 1,
                previous_contract_id TEXT,
                description TEXT,
                consent_method TEXT,
                revocation_reason TEXT,
                metadata_json TEXT,
                signature_hash TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contract_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                actor TEXT NOT NULL,
                details TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (contract_id) REFERENCES contracts(contract_id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_contracts_user_id
            ON contracts(user_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_contracts_status
            ON contracts(status)
        """)

        self._connection.commit()

    def _insert_contract(self, contract: LearningContract) -> None:
        """Insert a contract into the database."""
        cursor = self._connection.cursor()
        cursor.execute("""
            INSERT INTO contracts (
                contract_id, user_id, contract_type, scope_json, status,
                created_at, activated_at, expires_at, revoked_at,
                version, previous_contract_id, description, consent_method,
                revocation_reason, metadata_json, signature_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            contract.contract_id,
            contract.user_id,
            contract.contract_type.name,
            json.dumps(contract.scope.to_dict()),
            contract.status.name,
            contract.created_at.isoformat(),
            contract.activated_at.isoformat() if contract.activated_at else None,
            contract.expires_at.isoformat() if contract.expires_at else None,
            contract.revoked_at.isoformat() if contract.revoked_at else None,
            contract.version,
            contract.previous_contract_id,
            contract.description,
            contract.consent_method,
            contract.revocation_reason,
            json.dumps(contract.metadata),
            contract.signature_hash,
        ))
        self._connection.commit()

    def _get_contract(self, contract_id: str) -> Optional[LearningContract]:
        """Get a contract from the database."""
        cursor = self._connection.cursor()
        cursor.execute("SELECT * FROM contracts WHERE contract_id = ?", (contract_id,))
        row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_contract(row)

    def _query_contracts(self, query: ContractQuery) -> List[LearningContract]:
        """Query contracts from the database."""
        sql = "SELECT * FROM contracts WHERE 1=1"
        params = []

        if query.user_id:
            sql += " AND user_id = ?"
            params.append(query.user_id)

        if query.status:
            sql += " AND status = ?"
            params.append(query.status.name)

        if query.statuses:
            placeholders = ",".join("?" * len(query.statuses))
            sql += f" AND status IN ({placeholders})"
            params.extend(s.name for s in query.statuses)

        if query.contract_type:
            sql += " AND contract_type = ?"
            params.append(query.contract_type.name)

        if query.active_only:
            sql += " AND status = ?"
            params.append(ContractStatus.ACTIVE.name)

        if not query.include_expired:
            sql += " AND (expires_at IS NULL OR expires_at > ?)"
            params.append(datetime.now().isoformat())

        sql += f" ORDER BY created_at DESC LIMIT {query.limit}"

        cursor = self._connection.cursor()
        cursor.execute(sql, params)

        return [self._row_to_contract(row) for row in cursor.fetchall()]

    def _row_to_contract(self, row: sqlite3.Row) -> LearningContract:
        """Convert database row to LearningContract."""
        return LearningContract(
            contract_id=row["contract_id"],
            user_id=row["user_id"],
            contract_type=ContractType[row["contract_type"]],
            scope=ContractScope.from_dict(json.loads(row["scope_json"])),
            status=ContractStatus[row["status"]],
            created_at=datetime.fromisoformat(row["created_at"]),
            activated_at=datetime.fromisoformat(row["activated_at"]) if row["activated_at"] else None,
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            revoked_at=datetime.fromisoformat(row["revoked_at"]) if row["revoked_at"] else None,
            version=row["version"],
            previous_contract_id=row["previous_contract_id"],
            description=row["description"] or "",
            consent_method=row["consent_method"] or "explicit",
            revocation_reason=row["revocation_reason"],
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
            signature_hash=row["signature_hash"],
        )

    def _log_event(
        self,
        contract_id: str,
        event_type: str,
        actor: str,
        details: str = "",
    ) -> None:
        """Log a contract event."""
        cursor = self._connection.cursor()
        cursor.execute("""
            INSERT INTO contract_events (contract_id, event_type, actor, details, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (contract_id, event_type, actor, details, datetime.now().isoformat()))
        self._connection.commit()


def create_contract_store(
    db_path: Optional[Path] = None,
    default_deny: bool = True,
) -> ContractStore:
    """
    Factory function to create a contract store.

    Args:
        db_path: Path to database (None for in-memory)
        default_deny: Default to denying learning without contract

    Returns:
        Initialized ContractStore
    """
    store = ContractStore(db_path=db_path, default_deny=default_deny)
    if not store.initialize():
        raise RuntimeError("Failed to initialize contract store")
    return store


# =============================================================================
# Pre-built Contract Templates
# Aligned with learning-contracts specification
# =============================================================================


@dataclass
class ContractTemplate:
    """
    Pre-built contract template for common use cases.

    Templates from learning-contracts spec:
    https://github.com/kase1111-hash/learning-contracts
    """
    name: str
    description: str
    contract_type: ContractType
    scope: ContractScope
    default_duration: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def create_contract(
        self,
        store: ContractStore,
        user_id: str,
        duration: Optional[timedelta] = None,
        auto_activate: bool = True,
    ) -> LearningContract:
        """Create a contract from this template."""
        return store.create_contract(
            user_id=user_id,
            contract_type=self.contract_type,
            scope=self.scope,
            duration=duration or self.default_duration,
            description=self.description,
            metadata={**self.metadata, "template": self.name},
            auto_activate=auto_activate,
        )


# Seven pre-built templates from learning-contracts specification
CONTRACT_TEMPLATES: Dict[str, ContractTemplate] = {
    # 1. Coding Template
    "coding": ContractTemplate(
        name="coding",
        description="Learning contract for coding assistance. Allows learning coding patterns, "
                   "preferences, and project context. Enables procedural heuristics for better suggestions.",
        contract_type=ContractType.PROCEDURAL,
        scope=ContractScope(
            scope_type=LearningScope.DOMAIN_SPECIFIC,
            domains={"coding", "programming", "development", "debugging"},
            content_types={"code", "technical", "documentation"},
            excluded_domains={"credentials", "secrets", "api_keys"},
        ),
        metadata={"category": "technical", "trust_level": "medium"},
    ),

    # 2. Gaming Template
    "gaming": ContractTemplate(
        name="gaming",
        description="Learning contract for gaming sessions. Allows episodic memory of game states "
                   "and preferences without cross-session generalization.",
        contract_type=ContractType.EPISODIC,
        scope=ContractScope(
            scope_type=LearningScope.DOMAIN_SPECIFIC,
            domains={"gaming", "games", "entertainment"},
            content_types={"game_state", "preferences", "achievements"},
        ),
        default_duration=timedelta(hours=24),  # Session-based
        metadata={"category": "entertainment", "trust_level": "low"},
    ),

    # 3. Journaling Template
    "journaling": ContractTemplate(
        name="journaling",
        description="Learning contract for personal journaling. Allows long-term memory of "
                   "personal notes and reflections with high privacy protection.",
        contract_type=ContractType.EPISODIC,
        scope=ContractScope(
            scope_type=LearningScope.DOMAIN_SPECIFIC,
            domains={"personal", "journal", "diary", "notes"},
            content_types={"personal", "reflections", "goals"},
            excluded_domains={"medical", "financial", "legal"},
        ),
        metadata={"category": "personal", "trust_level": "high", "privacy": "strict"},
    ),

    # 4. Work Projects Template
    "work_projects": ContractTemplate(
        name="work_projects",
        description="Learning contract for work-related projects. Enables procedural learning "
                   "of workflows and project patterns within defined work domains.",
        contract_type=ContractType.PROCEDURAL,
        scope=ContractScope(
            scope_type=LearningScope.TASK_SPECIFIC,
            tasks={"project_management", "documentation", "analysis", "reporting"},
            domains={"work", "professional", "business"},
            excluded_domains={"hr", "confidential", "proprietary"},
        ),
        metadata={"category": "professional", "trust_level": "medium"},
    ),

    # 5. Restricted Domains Template
    "restricted": ContractTemplate(
        name="restricted",
        description="Explicitly prohibits learning in sensitive domains. Use this to block "
                   "learning for medical, financial, legal, or other restricted areas.",
        contract_type=ContractType.PROHIBITED,
        scope=ContractScope(
            scope_type=LearningScope.DOMAIN_SPECIFIC,
            domains={"medical", "financial", "legal", "credentials", "passwords", "secrets"},
        ),
        metadata={"category": "restricted", "trust_level": "none", "immutable": True},
    ),

    # 6. Study Template
    "study": ContractTemplate(
        name="study",
        description="Learning contract for educational content. Allows procedural learning "
                   "of study material with cross-context application for better retention.",
        contract_type=ContractType.PROCEDURAL,
        scope=ContractScope(
            scope_type=LearningScope.DOMAIN_SPECIFIC,
            domains={"education", "study", "learning", "courses", "tutorials"},
            content_types={"educational", "reference", "examples"},
        ),
        metadata={"category": "education", "trust_level": "medium"},
    ),

    # 7. Strategy Template
    "strategy": ContractTemplate(
        name="strategy",
        description="High-trust strategic learning contract. Enables long-term pattern inference "
                   "and cross-context application. Use only for trusted, long-term relationships.",
        contract_type=ContractType.STRATEGIC,
        scope=ContractScope(
            scope_type=LearningScope.ALL,
            excluded_domains={"credentials", "passwords", "secrets", "medical", "financial"},
        ),
        metadata={"category": "strategic", "trust_level": "high", "requires_explicit_consent": True},
    ),
}


def get_template(name: str) -> Optional[ContractTemplate]:
    """Get a contract template by name."""
    return CONTRACT_TEMPLATES.get(name.lower())


def list_templates() -> List[str]:
    """List all available template names."""
    return list(CONTRACT_TEMPLATES.keys())


def create_contract_from_template(
    store: ContractStore,
    template_name: str,
    user_id: str,
    duration: Optional[timedelta] = None,
    auto_activate: bool = True,
) -> Optional[LearningContract]:
    """
    Create a contract from a pre-built template.

    Args:
        store: Contract store to use
        template_name: Name of the template (coding, gaming, etc.)
        user_id: User creating the contract
        duration: Optional override for default duration
        auto_activate: Whether to automatically activate

    Returns:
        Created LearningContract or None if template not found
    """
    template = get_template(template_name)
    if not template:
        logger.warning(f"Unknown contract template: {template_name}")
        return None

    return template.create_contract(
        store=store,
        user_id=user_id,
        duration=duration,
        auto_activate=auto_activate,
    )
