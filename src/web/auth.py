"""
User Authentication Module for Agent OS Web Interface.

Provides:
- User account management (create, authenticate, update)
- Password hashing with bcrypt-like security
- Session token management
- Role-based access control
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """User roles for access control."""

    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


class AuthError(Exception):
    """Authentication error."""

    def __init__(self, message: str, error_code: str = "AUTH_ERROR"):
        super().__init__(message)
        self.error_code = error_code


@dataclass
class User:
    """User account model."""

    user_id: str
    username: str
    email: Optional[str]
    password_hash: str
    salt: str
    role: UserRole = UserRole.USER
    display_name: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "display_name": self.display_name or self.username,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "is_active": self.is_active,
        }
        if include_sensitive:
            data["metadata"] = self.metadata
        return data


@dataclass
class Session:
    """User session model."""

    session_id: str
    user_id: str
    token: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_activity: datetime = field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True

    @property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() >= self.expires_at

    @property
    def is_valid(self) -> bool:
        """Check if session is valid."""
        return self.is_active and not self.is_expired

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_activity": self.last_activity.isoformat(),
            "is_active": self.is_active,
        }


@dataclass
class LoginAttempt:
    """Record of a login attempt for rate limiting."""

    identifier: str  # Username or IP address
    attempt_time: datetime
    success: bool


class RateLimiter:
    """Rate limiter for authentication attempts.

    Implements:
    - Exponential backoff after failed attempts
    - Account lockout after max failures
    """

    MAX_ATTEMPTS = 5  # Max failed attempts before lockout
    LOCKOUT_DURATION = timedelta(minutes=15)  # Lockout duration
    ATTEMPT_WINDOW = timedelta(minutes=15)  # Window to count failed attempts

    # Exponential backoff delays (in seconds) after each failed attempt
    BACKOFF_DELAYS = [0, 1, 2, 4, 8]  # 0s, 1s, 2s, 4s, 8s

    def __init__(self):
        self._attempts: Dict[str, List[LoginAttempt]] = {}
        self._lockouts: Dict[str, datetime] = {}
        self._lock = threading.Lock()

    def is_locked_out(self, identifier: str) -> Tuple[bool, Optional[int]]:
        """Check if identifier is locked out.

        Args:
            identifier: Username or IP address

        Returns:
            Tuple of (is_locked, seconds_remaining)
        """
        with self._lock:
            if identifier not in self._lockouts:
                return False, None

            lockout_end = self._lockouts[identifier]
            if datetime.utcnow() >= lockout_end:
                del self._lockouts[identifier]
                return False, None

            remaining = int((lockout_end - datetime.utcnow()).total_seconds())
            return True, remaining

    def get_backoff_delay(self, identifier: str) -> int:
        """Get required backoff delay in seconds.

        Args:
            identifier: Username or IP address

        Returns:
            Delay in seconds before next attempt is allowed
        """
        with self._lock:
            recent_failures = self._get_recent_failures(identifier)
            if recent_failures >= len(self.BACKOFF_DELAYS):
                return self.BACKOFF_DELAYS[-1]
            return self.BACKOFF_DELAYS[recent_failures]

    def record_attempt(self, identifier: str, success: bool) -> None:
        """Record a login attempt.

        Args:
            identifier: Username or IP address
            success: Whether the attempt was successful
        """
        with self._lock:
            attempt = LoginAttempt(
                identifier=identifier,
                attempt_time=datetime.utcnow(),
                success=success,
            )

            if identifier not in self._attempts:
                self._attempts[identifier] = []

            self._attempts[identifier].append(attempt)
            self._cleanup_old_attempts(identifier)

            if success:
                # Clear lockout and failed attempts on success
                if identifier in self._lockouts:
                    del self._lockouts[identifier]
                self._attempts[identifier] = [a for a in self._attempts[identifier] if a.success]
            else:
                # Check if lockout should be applied
                failures = self._get_recent_failures(identifier)
                if failures >= self.MAX_ATTEMPTS:
                    self._lockouts[identifier] = datetime.utcnow() + self.LOCKOUT_DURATION
                    logger.warning(f"Account locked out: {identifier} (too many failed attempts)")

    def _get_recent_failures(self, identifier: str) -> int:
        """Get count of recent failed attempts (within window)."""
        if identifier not in self._attempts:
            return 0

        cutoff = datetime.utcnow() - self.ATTEMPT_WINDOW
        return sum(
            1 for a in self._attempts[identifier] if not a.success and a.attempt_time > cutoff
        )

    def _cleanup_old_attempts(self, identifier: str) -> None:
        """Remove attempts older than the window."""
        if identifier not in self._attempts:
            return

        cutoff = datetime.utcnow() - self.ATTEMPT_WINDOW
        self._attempts[identifier] = [
            a for a in self._attempts[identifier] if a.attempt_time > cutoff
        ]


class PasswordHasher:
    """Secure password hashing using PBKDF2-SHA256."""

    ITERATIONS = 600000  # NIST SP 800-132 recommends >= 600,000 for SHA-256
    HASH_LENGTH = 32
    SALT_LENGTH = 32  # Standardized 32-byte salt (64 hex chars)
    MIN_PASSWORD_LENGTH = 12
    MAX_PASSWORD_LENGTH = 128

    @classmethod
    def validate_password(cls, password: str) -> tuple[bool, str]:
        """
        Validate password meets security requirements.

        Requirements:
        - At least 12 characters
        - At most 128 characters
        - At least one uppercase letter
        - At least one lowercase letter
        - At least one digit
        - At least one special character

        Args:
            password: Password to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not password:
            return False, "Password is required"

        if len(password) < cls.MIN_PASSWORD_LENGTH:
            return False, f"Password must be at least {cls.MIN_PASSWORD_LENGTH} characters"

        if len(password) > cls.MAX_PASSWORD_LENGTH:
            return False, f"Password must be at most {cls.MAX_PASSWORD_LENGTH} characters"

        if not re.search(r"[A-Z]", password):
            return False, "Password must contain at least one uppercase letter"

        if not re.search(r"[a-z]", password):
            return False, "Password must contain at least one lowercase letter"

        if not re.search(r"\d", password):
            return False, "Password must contain at least one digit"

        if not re.search(r'[!@#$%^&*(),.?":{}|<>_\-+=\[\]\\;\'`~]', password):
            return False, "Password must contain at least one special character"

        return True, ""

    @classmethod
    def hash_password(cls, password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """
        Hash a password with a salt.

        Args:
            password: Plain text password
            salt: Optional salt (generated if not provided)

        Returns:
            Tuple of (password_hash, salt)
        """
        if salt is None:
            salt = secrets.token_hex(cls.SALT_LENGTH)  # 32 bytes = 64 hex chars

        password_bytes = password.encode("utf-8")
        salt_bytes = salt.encode("utf-8")

        hash_bytes = hashlib.pbkdf2_hmac(
            "sha256", password_bytes, salt_bytes, cls.ITERATIONS, dklen=cls.HASH_LENGTH
        )

        return hash_bytes.hex(), salt

    @classmethod
    def verify_password(cls, password: str, password_hash: str, salt: str) -> bool:
        """
        Verify a password against a hash.

        Args:
            password: Plain text password to verify
            password_hash: Stored password hash
            salt: Salt used for hashing

        Returns:
            True if password matches
        """
        computed_hash, _ = cls.hash_password(password, salt)
        return hmac.compare_digest(computed_hash, password_hash)


class UserStore:
    """
    SQLite-based user account store.

    Provides CRUD operations for user accounts and sessions.
    Session tokens are cryptographically bound to session metadata
    (user ID, IP, expiry) using HMAC.
    """

    # Token secret is generated per-instance and should be stored securely
    # in production (e.g., from environment variable or keyring)
    _TOKEN_SECRET_ENV = "AGENT_OS_SESSION_SECRET"

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize user store.

        Args:
            db_path: Path to SQLite database (None for in-memory)
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        self._connection: Optional[sqlite3.Connection] = None
        self._initialized = False
        self._rate_limiter = RateLimiter()

        # Session token signing secret
        env_secret = os.environ.get(self._TOKEN_SECRET_ENV)
        if env_secret:
            self._token_secret = env_secret.encode()
        else:
            # Generate random secret for this instance
            # Note: In production, this should be persistent
            self._token_secret = secrets.token_bytes(32)
            logger.warning(
                f"No {self._TOKEN_SECRET_ENV} set - using ephemeral session secret. "
                "Sessions will not survive restarts."
            )

    def _generate_bound_token(
        self,
        session_id: str,
        user_id: str,
        expires_at: datetime,
        ip_address: Optional[str] = None,
    ) -> str:
        """
        Generate a cryptographically bound session token.

        The token includes an HMAC that binds it to the session metadata,
        preventing token theft/replay attacks.

        Args:
            session_id: Unique session ID
            user_id: User ID
            expires_at: Token expiration time
            ip_address: Optional client IP (bound to token if provided)

        Returns:
            Base64-encoded token with embedded HMAC
        """
        # Create binding data
        binding_data = f"{session_id}:{user_id}:{expires_at.isoformat()}"
        if ip_address:
            binding_data += f":{ip_address}"

        # Generate random component
        random_component = secrets.token_bytes(16)

        # Generate HMAC over binding data + random component
        message = binding_data.encode() + random_component
        signature = hmac.new(self._token_secret, message, hashlib.sha256).digest()

        # Token format: random_component || signature (both in base64)
        token_bytes = random_component + signature
        return base64.urlsafe_b64encode(token_bytes).decode()

    def _verify_bound_token(
        self,
        token: str,
        session_id: str,
        user_id: str,
        expires_at: datetime,
        ip_address: Optional[str] = None,
    ) -> bool:
        """
        Verify a cryptographically bound session token.

        Args:
            token: The session token to verify
            session_id: Expected session ID
            user_id: Expected user ID
            expires_at: Expected expiration time
            ip_address: Expected client IP (if bound)

        Returns:
            True if token is valid and matches binding data
        """
        try:
            token_bytes = base64.urlsafe_b64decode(token.encode())

            if len(token_bytes) != 48:  # 16 random + 32 signature
                return False

            random_component = token_bytes[:16]
            provided_signature = token_bytes[16:]

            # Recreate binding data
            binding_data = f"{session_id}:{user_id}:{expires_at.isoformat()}"
            if ip_address:
                binding_data += f":{ip_address}"

            # Recompute expected signature
            message = binding_data.encode() + random_component
            expected_signature = hmac.new(self._token_secret, message, hashlib.sha256).digest()

            return hmac.compare_digest(provided_signature, expected_signature)

        except Exception as e:
            logger.warning(f"Token verification failed: {e}")
            return False

    def initialize(self) -> bool:
        """Initialize the user store."""
        try:
            db_str = str(self.db_path) if self.db_path else ":memory:"
            self._connection = sqlite3.connect(db_str, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row

            self._create_tables()
            self._initialized = True
            logger.info(f"User store initialized: {db_str}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize user store: {e}")
            return False

    def close(self) -> None:
        """Close the store."""
        if self._connection:
            self._connection.close()
            self._connection = None
        self._initialized = False

    def _create_tables(self) -> None:
        """Create database tables."""
        cursor = self._connection.cursor()

        # Users table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                display_name TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_login TEXT,
                is_active INTEGER DEFAULT 1,
                metadata_json TEXT
            )
        """
        )

        # Sessions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                last_activity TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                is_active INTEGER DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """
        )

        # Indexes
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)
        """
        )

        self._connection.commit()

    def create_user(
        self,
        username: str,
        password: str,
        email: Optional[str] = None,
        role: UserRole = UserRole.USER,
        display_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> User:
        """
        Create a new user account.

        Args:
            username: Unique username
            password: Plain text password (will be hashed)
            email: Optional email address
            role: User role
            display_name: Optional display name
            metadata: Optional metadata

        Returns:
            Created User

        Raises:
            AuthError: If username or email already exists
        """
        if not self._initialized:
            raise RuntimeError("User store not initialized")

        # Validate username
        if not username or len(username) < 3:
            raise AuthError("Username must be at least 3 characters", "INVALID_USERNAME")

        if len(username) > 50:
            raise AuthError("Username must be at most 50 characters", "INVALID_USERNAME")

        # Validate password
        is_valid, error_msg = PasswordHasher.validate_password(password)
        if not is_valid:
            raise AuthError(error_msg, "INVALID_PASSWORD")

        # Check for existing username
        if self.get_user_by_username(username):
            raise AuthError("Username already exists", "USERNAME_EXISTS")

        # Check for existing email
        if email and self.get_user_by_email(email):
            raise AuthError("Email already exists", "EMAIL_EXISTS")

        # Hash password
        password_hash, salt = PasswordHasher.hash_password(password)

        # Generate user ID
        user_id = f"user_{secrets.token_hex(8)}"
        now = datetime.utcnow()

        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            salt=salt,
            role=role,
            display_name=display_name,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )

        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(
                """
                INSERT INTO users (
                    user_id, username, email, password_hash, salt, role,
                    display_name, created_at, updated_at, last_login,
                    is_active, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    user.user_id,
                    user.username,
                    user.email,
                    user.password_hash,
                    user.salt,
                    user.role.value,
                    user.display_name,
                    user.created_at.isoformat(),
                    user.updated_at.isoformat(),
                    None,
                    1 if user.is_active else 0,
                    json.dumps(user.metadata),
                ),
            )
            self._connection.commit()

        logger.info(f"Created user: {username} ({user_id})")
        return user

    def authenticate(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
    ) -> Tuple[Optional[User], Optional[str]]:
        """
        Authenticate a user with username and password.

        Includes rate limiting with exponential backoff and account lockout.

        Args:
            username: Username or email
            password: Plain text password
            ip_address: Optional client IP for rate limiting

        Returns:
            Tuple of (User, None) if successful, (None, error_message) otherwise
        """
        # Check rate limiting by username
        is_locked, remaining = self._rate_limiter.is_locked_out(username)
        if is_locked:
            logger.warning(f"Login attempt for locked account: {username}")
            return None, f"Account temporarily locked. Try again in {remaining} seconds."

        # Also check by IP if provided
        if ip_address:
            ip_locked, ip_remaining = self._rate_limiter.is_locked_out(ip_address)
            if ip_locked:
                logger.warning(f"Login attempt from locked IP: {ip_address}")
                return None, f"Too many failed attempts. Try again in {ip_remaining} seconds."

        # Try to find user by username or email
        user = self.get_user_by_username(username)
        if not user:
            user = self.get_user_by_email(username)

        if not user:
            # Record failed attempt even for non-existent users (timing attack prevention)
            self._rate_limiter.record_attempt(username, success=False)
            if ip_address:
                self._rate_limiter.record_attempt(ip_address, success=False)
            return None, "Invalid username or password"

        if not user.is_active:
            self._rate_limiter.record_attempt(username, success=False)
            if ip_address:
                self._rate_limiter.record_attempt(ip_address, success=False)
            return None, "Account is inactive"

        # Verify password
        if not PasswordHasher.verify_password(password, user.password_hash, user.salt):
            self._rate_limiter.record_attempt(username, success=False)
            if ip_address:
                self._rate_limiter.record_attempt(ip_address, success=False)
            logger.warning(f"Failed login attempt for user: {username}")
            return None, "Invalid username or password"

        # Success - record and clear any lockouts
        self._rate_limiter.record_attempt(username, success=True)
        if ip_address:
            self._rate_limiter.record_attempt(ip_address, success=True)

        # Update last login
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(
                """
                UPDATE users SET last_login = ? WHERE user_id = ?
            """,
                (datetime.utcnow().isoformat(), user.user_id),
            )
            self._connection.commit()

        user.last_login = datetime.utcnow()
        logger.info(f"Successful login for user: {username}")
        return user, None

    def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()

        return self._row_to_user(row) if row else None

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username."""
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()

        return self._row_to_user(row) if row else None

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email."""
        if not email:
            return None
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
            row = cursor.fetchone()

        return self._row_to_user(row) if row else None

    def update_user(
        self,
        user_id: str,
        display_name: Optional[str] = None,
        email: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> bool:
        """Update user details."""
        updates = []
        params = []

        if display_name is not None:
            updates.append("display_name = ?")
            params.append(display_name)

        if email is not None:
            # Check email uniqueness
            existing = self.get_user_by_email(email)
            if existing and existing.user_id != user_id:
                raise AuthError("Email already exists", "EMAIL_EXISTS")
            updates.append("email = ?")
            params.append(email)

        if is_active is not None:
            updates.append("is_active = ?")
            params.append(1 if is_active else 0)

        if not updates:
            return False

        updates.append("updated_at = ?")
        params.append(datetime.utcnow().isoformat())
        params.append(user_id)

        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(f"UPDATE users SET {', '.join(updates)} WHERE user_id = ?", params)  # nosec B608 - column names are hardcoded
            self._connection.commit()
            return cursor.rowcount > 0

    def change_password(self, user_id: str, new_password: str) -> bool:
        """Change a user's password.

        Also invalidates all existing sessions for security, forcing
        re-authentication with the new password.
        """
        is_valid, error_msg = PasswordHasher.validate_password(new_password)
        if not is_valid:
            raise AuthError(error_msg, "INVALID_PASSWORD")

        password_hash, salt = PasswordHasher.hash_password(new_password)

        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(
                """
                UPDATE users SET password_hash = ?, salt = ?, updated_at = ?
                WHERE user_id = ?
            """,
                (password_hash, salt, datetime.utcnow().isoformat(), user_id),
            )
            self._connection.commit()
            success = cursor.rowcount > 0

        # Invalidate all existing sessions for security
        if success:
            sessions_invalidated = self.invalidate_all_user_sessions(user_id)
            logger.info(
                f"Password changed for user {user_id}, "
                f"invalidated {sessions_invalidated} sessions"
            )

        return success

    def list_users(self, include_inactive: bool = False) -> List[User]:
        """List all users."""
        with self._lock:
            cursor = self._connection.cursor()
            if include_inactive:
                cursor.execute("SELECT * FROM users ORDER BY created_at DESC")
            else:
                cursor.execute("SELECT * FROM users WHERE is_active = 1 ORDER BY created_at DESC")
            rows = cursor.fetchall()

        return [self._row_to_user(row) for row in rows]

    def get_user_count(self) -> int:
        """Get total user count."""
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
            return cursor.fetchone()[0]

    def _row_to_user(self, row: sqlite3.Row) -> User:
        """Convert database row to User object."""
        return User(
            user_id=row["user_id"],
            username=row["username"],
            email=row["email"],
            password_hash=row["password_hash"],
            salt=row["salt"],
            role=UserRole(row["role"]),
            display_name=row["display_name"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_login=datetime.fromisoformat(row["last_login"]) if row["last_login"] else None,
            is_active=bool(row["is_active"]),
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        )

    # Session management

    def create_session(
        self,
        user_id: str,
        duration_hours: int = 24,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        bind_to_ip: bool = True,
    ) -> Session:
        """
        Create a new session for a user.

        The session token is cryptographically bound to the session metadata
        (user ID, session ID, expiry, and optionally IP address) using HMAC.
        This prevents token theft and replay attacks.

        Args:
            user_id: User ID
            duration_hours: Session duration in hours
            ip_address: Client IP address
            user_agent: Client user agent
            bind_to_ip: Whether to bind token to IP address

        Returns:
            Created Session
        """
        session_id = f"sess_{secrets.token_hex(16)}"
        now = datetime.utcnow()
        expires_at = now + timedelta(hours=duration_hours)

        # Generate cryptographically bound token
        bound_ip = ip_address if bind_to_ip else None
        token = self._generate_bound_token(
            session_id=session_id,
            user_id=user_id,
            expires_at=expires_at,
            ip_address=bound_ip,
        )

        session = Session(
            session_id=session_id,
            user_id=user_id,
            token=token,
            created_at=now,
            expires_at=expires_at,
            last_activity=now,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(
                """
                INSERT INTO sessions (
                    session_id, user_id, token, created_at, expires_at,
                    last_activity, ip_address, user_agent, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session.session_id,
                    session.user_id,
                    session.token,
                    session.created_at.isoformat(),
                    session.expires_at.isoformat() if session.expires_at else None,
                    session.last_activity.isoformat(),
                    session.ip_address,
                    session.user_agent,
                    1,
                ),
            )
            self._connection.commit()

        logger.info(f"Created session: {session_id} for user {user_id}")
        return session

    def get_session_by_token(self, token: str) -> Optional[Session]:
        """Get a session by token."""
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute("SELECT * FROM sessions WHERE token = ? AND is_active = 1", (token,))
            row = cursor.fetchone()

        if not row:
            return None

        session = self._row_to_session(row)

        # Check expiration
        if session.is_expired:
            self.invalidate_session(session.session_id)
            return None

        return session

    def validate_session(
        self,
        token: str,
        ip_address: Optional[str] = None,
        verify_binding: bool = True,
    ) -> Optional[User]:
        """
        Validate a session token and return the associated user.

        Verifies the cryptographic binding of the token to ensure it
        hasn't been tampered with or stolen.

        Args:
            token: Session token
            ip_address: Client IP (for binding verification)
            verify_binding: Whether to verify token binding (default: True)

        Returns:
            User if session is valid, None otherwise
        """
        session = self.get_session_by_token(token)
        if not session or not session.is_valid:
            return None

        # Verify cryptographic binding if enabled
        if verify_binding and session.expires_at:
            # Use stored IP if available for verification
            bound_ip = session.ip_address if session.ip_address else None

            # If caller provided IP, use that for stricter verification
            if ip_address and session.ip_address:
                # Check if IP matches the bound IP
                if ip_address != session.ip_address:
                    logger.warning(
                        f"Session IP mismatch: expected {session.ip_address}, " f"got {ip_address}"
                    )
                    return None

            if not self._verify_bound_token(
                token=token,
                session_id=session.session_id,
                user_id=session.user_id,
                expires_at=session.expires_at,
                ip_address=bound_ip,
            ):
                logger.warning(f"Session token binding verification failed")
                return None

        user = self.get_user(session.user_id)
        if not user or not user.is_active:
            return None

        # Update last activity
        self.touch_session(session.session_id)

        return user

    def touch_session(self, session_id: str) -> None:
        """Update session last activity time."""
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(
                """
                UPDATE sessions SET last_activity = ? WHERE session_id = ?
            """,
                (datetime.utcnow().isoformat(), session_id),
            )
            self._connection.commit()

    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session."""
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(
                """
                UPDATE sessions SET is_active = 0 WHERE session_id = ?
            """,
                (session_id,),
            )
            self._connection.commit()
            return cursor.rowcount > 0

    def invalidate_session_by_token(self, token: str) -> bool:
        """Invalidate a session by token."""
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(
                """
                UPDATE sessions SET is_active = 0 WHERE token = ?
            """,
                (token,),
            )
            self._connection.commit()
            return cursor.rowcount > 0

    def invalidate_all_user_sessions(self, user_id: str) -> int:
        """Invalidate all sessions for a user."""
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(
                """
                UPDATE sessions SET is_active = 0 WHERE user_id = ?
            """,
                (user_id,),
            )
            self._connection.commit()
            return cursor.rowcount

    def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all active sessions for a user."""
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(
                """
                SELECT * FROM sessions WHERE user_id = ? AND is_active = 1
                ORDER BY last_activity DESC
            """,
                (user_id,),
            )
            rows = cursor.fetchall()

        return [self._row_to_session(row) for row in rows]

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(
                """
                UPDATE sessions SET is_active = 0
                WHERE is_active = 1 AND expires_at < ?
            """,
                (datetime.utcnow().isoformat(),),
            )
            self._connection.commit()
            return cursor.rowcount

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        """Convert database row to Session object."""
        return Session(
            session_id=row["session_id"],
            user_id=row["user_id"],
            token=row["token"],
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            last_activity=datetime.fromisoformat(row["last_activity"]),
            ip_address=row["ip_address"],
            user_agent=row["user_agent"],
            is_active=bool(row["is_active"]),
        )


# Global user store instance
_user_store: Optional[UserStore] = None


def get_user_store() -> UserStore:
    """Get the global user store instance."""
    global _user_store
    if _user_store is None:
        # Default to file-based storage in the current directory
        from .config import get_config

        config = get_config()
        db_path = config.static_dir.parent / "users.db"
        _user_store = UserStore(db_path)
        _user_store.initialize()
    return _user_store


def set_user_store(store: UserStore) -> None:
    """Set the global user store instance."""
    global _user_store
    _user_store = store


def create_user_store(db_path: Optional[Path] = None) -> UserStore:
    """
    Factory function to create a user store.

    Args:
        db_path: Path to database (None for in-memory)

    Returns:
        Initialized UserStore
    """
    store = UserStore(db_path=db_path)
    if not store.initialize():
        raise RuntimeError("Failed to initialize user store")
    return store
