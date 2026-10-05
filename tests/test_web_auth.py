"""
Tests for the web authentication layer.

Covers:
- src/web/auth.py: PasswordHasher, login RateLimiter, UserStore (accounts, authentication,
  sessions, scoped API keys, the legacy AGENT_OS_API_KEY path) and the store factories.
- src/web/auth_helpers.py: token extraction, API key auth, the require_* FastAPI
  dependencies and WebSocket authentication.
- src/web/routes/auth.py: the /api/auth/* endpoints, through the real application.

Includes regression tests for the configured AGENT_OS_API_KEY (which has no ``aos_``
prefix) being rejected by ``_try_api_key_auth``, which left admin-only endpoints
unreachable when authentication was enabled.

All state is isolated: in-memory or tmp_path SQLite stores, DI overrides restored after
each test, the session secret supplied via environment, and no writes to src/web/data/.
"""

import hashlib
import logging
import secrets
import stat
from datetime import datetime, timedelta

import pytest
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.core.exceptions import AuthError
from src.web.auth import (
    ApiKeyScope,
    PasswordHasher,
    RateLimiter,
    ScopedApiKey,
    UserRole,
    UserStore,
    create_user_store,
    get_user_store,
    reset_user_store,
    set_user_store,
)
from src.web.auth_helpers import (
    _extract_token,
    _try_api_key_auth,
    authenticate_websocket,
    require_admin_user,
    require_authenticated_user,
    require_scope,
)
from src.web.config import ConfigurationError, WebConfig, set_config
from src.web.dependencies import DependencyOverrides, _container

# The README tells users to generate AGENT_OS_API_KEY this way; it has no "aos_" prefix.
CONFIGURED_API_KEY = secrets.token_urlsafe(32)

STRONG_PASSWORD = "Corr3ct-Horse!"
NEW_PASSWORD = "N3w-Battery#Staple"
SESSION_SECRET = "web-auth-tests-session-secret"


# =============================================================================
# Helpers
# =============================================================================


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _make_request(headers: dict = None, cookies: dict = None) -> Request:
    """Build a bare Starlette request carrying the given headers and cookies."""
    raw_headers = [
        (name.lower().encode("latin-1"), value.encode("latin-1"))
        for name, value in (headers or {}).items()
    ]
    if cookies:
        cookie_header = "; ".join(f"{name}={value}" for name, value in cookies.items())
        raw_headers.append((b"cookie", cookie_header.encode("latin-1")))
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": raw_headers,
            "query_string": b"",
        }
    )


def _login(client: TestClient, username: str, password: str = STRONG_PASSWORD) -> str:
    """Log in through the API and return the session token.

    The cookie jar is cleared so later requests only carry the credentials a test
    passes explicitly.
    """
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    client.cookies.clear()
    return response.json()["token"]


def _build_helper_app() -> FastAPI:
    """Minimal app exercising the auth helpers as real FastAPI dependencies."""
    helper_app = FastAPI()

    @helper_app.post("/chat")
    def write_chat(
        user_id: str = Depends(require_authenticated_user),
        _scope: None = Depends(require_scope(ApiKeyScope.WRITE_CHAT.value)),
    ):
        return {"user_id": user_id}

    @helper_app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        user_id = await authenticate_websocket(websocket)
        if not user_id:
            await websocket.close(code=4001)
            return
        await websocket.accept()
        await websocket.send_json({"user_id": user_id})
        await websocket.close()

    return helper_app


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _isolated_auth_env(monkeypatch, tmp_path):
    """Keep session secrets and machine salts out of the home directory."""
    monkeypatch.setenv("AGENT_OS_SESSION_SECRET", SESSION_SECRET)
    monkeypatch.setenv("AGENT_OS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("AGENT_OS_API_KEY", raising=False)


@pytest.fixture
def web_config(tmp_path):
    """Production-like config: auth required, API key configured."""
    return WebConfig(
        debug=True,
        require_auth=True,
        api_key=CONFIGURED_API_KEY,
        force_https=False,
        rate_limit_enabled=False,
        data_dir=tmp_path / "data",
    )


@pytest.fixture(autouse=True)
def config_override(web_config):
    """Install web_config for every test; restore prior DI overrides afterwards."""
    with DependencyOverrides() as overrides:
        overrides.set_config(web_config)
        yield web_config


@pytest.fixture
def store(config_override):
    """In-memory UserStore installed as the application's user store."""
    user_store = create_user_store()
    set_user_store(user_store)
    yield user_store
    user_store.close()


@pytest.fixture
def user(store):
    return store.create_user("alice", STRONG_PASSWORD, email="alice@example.com")


@pytest.fixture
def admin(store):
    return store.create_user("root", STRONG_PASSWORD, role=UserRole.ADMIN)


@pytest.fixture
def intent_store(monkeypatch):
    """In-memory intent log so auth routes don't write to the configured data dir."""
    from src.web import intent_log

    log_store = intent_log.IntentLogStore()
    log_store.initialize()
    monkeypatch.setattr(intent_log, "_intent_log_store", log_store)
    return log_store


@pytest.fixture
def client(web_config, store, intent_store, monkeypatch):
    """TestClient for the real application, over https so Secure cookies round-trip."""
    from src.web import app as app_module

    # create_app() replaces module-level globals; restore them after the test.
    monkeypatch.setattr(app_module, "_app", app_module._app)
    monkeypatch.setattr(app_module._app_state, "config", app_module._app_state.config)
    application = app_module.create_app(web_config)
    return TestClient(application, base_url="https://testserver")


@pytest.fixture
def helper_client(store):
    return TestClient(_build_helper_app(), base_url="https://testserver")


# =============================================================================
# PasswordHasher
# =============================================================================


class TestPasswordHasher:
    """Password policy and PBKDF2 hashing."""

    @pytest.mark.parametrize(
        "password, message",
        [
            ("", "Password is required"),
            ("Sh0rt!pass", "at least 12 characters"),
            ("A1!" + "a" * 126, "at most 128 characters"),
            ("all-lower-case-1!", "uppercase"),
            ("ALL-UPPER-CASE-1!", "lowercase"),
            ("No-Digits-Here!!", "digit"),
            ("NoSpecialChars123", "special character"),
        ],
    )
    def test_rejects_weak_passwords(self, password, message):
        is_valid, error = PasswordHasher.validate_password(password)
        assert is_valid is False
        assert message in error

    def test_accepts_strong_password(self):
        assert PasswordHasher.validate_password(STRONG_PASSWORD) == (True, "")

    def test_accepts_password_at_length_bounds(self):
        assert PasswordHasher.validate_password("Abcdefgh12!x")[0] is True  # 12 chars
        assert PasswordHasher.validate_password("A1!" + "a" * 125)[0] is True  # 128 chars

    def test_hash_and_verify(self):
        password_hash, salt = PasswordHasher.hash_password(STRONG_PASSWORD)

        assert len(salt) == 64  # 32 random bytes, hex encoded
        assert len(password_hash) == 64  # 32-byte PBKDF2-SHA256 digest, hex encoded
        assert STRONG_PASSWORD not in password_hash
        assert PasswordHasher.verify_password(STRONG_PASSWORD, password_hash, salt)
        assert not PasswordHasher.verify_password(NEW_PASSWORD, password_hash, salt)

    def test_salt_controls_hash(self):
        first, salt = PasswordHasher.hash_password(STRONG_PASSWORD)
        again, same_salt = PasswordHasher.hash_password(STRONG_PASSWORD, salt)
        other, other_salt = PasswordHasher.hash_password(STRONG_PASSWORD)

        assert same_salt == salt and again == first
        assert other_salt != salt and other != first


# =============================================================================
# UserStore: accounts
# =============================================================================


class TestUserStoreAccounts:
    """User creation, lookup and profile management."""

    def test_create_user_persists_hashed_credentials(self, store):
        created = store.create_user(
            "alice",
            STRONG_PASSWORD,
            email="alice@example.com",
            display_name="Alice",
            metadata={"team": "ops"},
        )

        assert created.user_id.startswith("user_")
        assert created.role == UserRole.USER
        assert created.is_active is True
        assert created.password_hash != STRONG_PASSWORD

        loaded = store.get_user(created.user_id)
        assert loaded.username == "alice"
        assert loaded.email == "alice@example.com"
        assert loaded.display_name == "Alice"
        assert loaded.metadata == {"team": "ops"}
        assert loaded.last_login is None
        assert PasswordHasher.verify_password(STRONG_PASSWORD, loaded.password_hash, loaded.salt)

        assert store.get_user_by_username("alice").user_id == created.user_id
        assert store.get_user_by_email("alice@example.com").user_id == created.user_id

    def test_create_user_with_role(self, store):
        created = store.create_user("root", STRONG_PASSWORD, role=UserRole.ADMIN)
        assert store.get_user(created.user_id).role == UserRole.ADMIN

    def test_lookups_for_missing_users(self, store):
        assert store.get_user("user_missing") is None
        assert store.get_user_by_username("nobody") is None
        assert store.get_user_by_email("nobody@example.com") is None
        assert store.get_user_by_email("") is None

    def test_to_dict_hides_credentials(self, user):
        data = user.to_dict()

        assert data["username"] == "alice"
        assert data["role"] == "user"
        assert data["display_name"] == "alice"  # falls back to the username
        assert data["last_login"] is None
        assert "password_hash" not in data
        assert "salt" not in data
        assert "metadata" not in data
        assert user.to_dict(include_sensitive=True)["metadata"] == {}

    @pytest.mark.parametrize("username", ["", "ab", "x" * 51])
    def test_rejects_invalid_username(self, store, username):
        with pytest.raises(AuthError) as exc_info:
            store.create_user(username, STRONG_PASSWORD)
        assert exc_info.value.error_code == "INVALID_USERNAME"

    def test_rejects_password_failing_policy(self, store):
        with pytest.raises(AuthError) as exc_info:
            store.create_user("alice", "lowercase-only-1!")
        assert exc_info.value.error_code == "INVALID_PASSWORD"
        assert "uppercase" in str(exc_info.value)
        assert store.get_user_by_username("alice") is None

    def test_rejects_duplicate_username(self, store, user):
        with pytest.raises(AuthError) as exc_info:
            store.create_user("alice", STRONG_PASSWORD)
        assert exc_info.value.error_code == "USERNAME_EXISTS"

    def test_rejects_duplicate_email(self, store, user):
        with pytest.raises(AuthError) as exc_info:
            store.create_user("alice2", STRONG_PASSWORD, email="alice@example.com")
        assert exc_info.value.error_code == "EMAIL_EXISTS"

    def test_uninitialized_store_refuses_writes(self):
        with pytest.raises(RuntimeError, match="not initialized"):
            UserStore().create_user("alice", STRONG_PASSWORD)

    def test_update_user_fields(self, store, user):
        assert store.update_user(user.user_id, display_name="Alice A.", email="a@example.com")

        updated = store.get_user(user.user_id)
        assert updated.display_name == "Alice A."
        assert updated.email == "a@example.com"
        assert updated.updated_at >= user.updated_at

    def test_update_user_keeps_own_email(self, store, user):
        assert store.update_user(user.user_id, email="alice@example.com") is True

    def test_update_user_rejects_taken_email(self, store, user):
        other = store.create_user("bob", STRONG_PASSWORD, email="bob@example.com")
        with pytest.raises(AuthError) as exc_info:
            store.update_user(other.user_id, email="alice@example.com")
        assert exc_info.value.error_code == "EMAIL_EXISTS"

    def test_update_user_without_changes_or_unknown_user(self, store, user):
        assert store.update_user(user.user_id) is False
        assert store.update_user("user_missing", display_name="Ghost") is False

    def test_deactivation_affects_counts_and_listing(self, store, user):
        bob = store.create_user("bob", STRONG_PASSWORD)
        assert store.get_user_count() == 2

        assert store.update_user(bob.user_id, is_active=False)

        assert store.get_user(bob.user_id).is_active is False
        assert store.get_user_count() == 1
        assert [u.username for u in store.list_users()] == ["alice"]
        assert {u.username for u in store.list_users(include_inactive=True)} == {"alice", "bob"}

    def test_change_password_rotates_credentials_and_sessions(self, store, user):
        session = store.create_session(user.user_id)

        assert store.change_password(user.user_id, NEW_PASSWORD) is True

        assert store.validate_session(session.token) is None
        assert store.get_user_sessions(user.user_id) == []
        assert store.authenticate("alice", NEW_PASSWORD)[0].user_id == user.user_id
        assert store.authenticate("alice", STRONG_PASSWORD) == (
            None,
            "Invalid username or password",
        )

    def test_change_password_enforces_policy(self, store, user):
        session = store.create_session(user.user_id)

        with pytest.raises(AuthError) as exc_info:
            store.change_password(user.user_id, "weakpassword")
        assert exc_info.value.error_code == "INVALID_PASSWORD"

        # Nothing changed: old password and session still work.
        assert store.validate_session(session.token).user_id == user.user_id
        assert store.authenticate("alice", STRONG_PASSWORD)[0] is not None

    def test_change_password_for_unknown_user(self, store):
        assert store.change_password("user_missing", NEW_PASSWORD) is False


# =============================================================================
# UserStore: authentication and lockout
# =============================================================================


class TestUserStoreAuthenticate:
    """Password authentication, including brute-force lockout."""

    def test_authenticate_by_username(self, store, user):
        authenticated, error = store.authenticate("alice", STRONG_PASSWORD)

        assert error is None
        assert authenticated.user_id == user.user_id
        assert authenticated.last_login is not None
        assert store.get_user(user.user_id).last_login is not None

    def test_authenticate_by_email(self, store, user):
        authenticated, error = store.authenticate("alice@example.com", STRONG_PASSWORD)
        assert error is None
        assert authenticated.user_id == user.user_id

    def test_wrong_password(self, store, user):
        assert store.authenticate("alice", NEW_PASSWORD) == (None, "Invalid username or password")

    def test_unknown_user_gets_same_message(self, store):
        assert store.authenticate("nobody", STRONG_PASSWORD) == (
            None,
            "Invalid username or password",
        )

    def test_inactive_user(self, store, user):
        store.update_user(user.user_id, is_active=False)
        assert store.authenticate("alice", STRONG_PASSWORD) == (None, "Account is inactive")

    def test_account_locks_after_repeated_failures(self, store, user):
        for _ in range(RateLimiter.MAX_ATTEMPTS):
            assert store.authenticate("alice", "Wrong-Passw0rd!")[0] is None

        authenticated, error = store.authenticate("alice", STRONG_PASSWORD)

        assert authenticated is None
        assert "Account temporarily locked" in error

    def test_ip_locks_after_repeated_failures(self, store, user):
        for attempt in range(RateLimiter.MAX_ATTEMPTS):
            store.authenticate(f"ghost{attempt}", STRONG_PASSWORD, ip_address="203.0.113.9")

        locked_out = store.authenticate("alice", STRONG_PASSWORD, ip_address="203.0.113.9")
        other_ip = store.authenticate("alice", STRONG_PASSWORD, ip_address="198.51.100.1")

        assert locked_out[0] is None
        assert "Too many failed attempts" in locked_out[1]
        assert other_ip[0] is not None


class TestLoginRateLimiter:
    """The in-memory login RateLimiter used by UserStore.authenticate."""

    def test_backoff_grows_and_caps(self):
        limiter = RateLimiter()
        delays = [limiter.get_backoff_delay("alice")]
        for _ in range(6):
            limiter.record_attempt("alice", success=False)
            delays.append(limiter.get_backoff_delay("alice"))

        assert delays == [0, 1, 2, 4, 8, 8, 8]

    def test_lockout_after_max_failures(self):
        limiter = RateLimiter()
        for _ in range(RateLimiter.MAX_ATTEMPTS - 1):
            limiter.record_attempt("alice", success=False)
        assert limiter.is_locked_out("alice") == (False, None)

        limiter.record_attempt("alice", success=False)
        is_locked, remaining = limiter.is_locked_out("alice")

        assert is_locked is True
        assert 0 < remaining <= RateLimiter.LOCKOUT_DURATION.total_seconds()
        assert limiter.is_locked_out("bob") == (False, None)

    def test_success_clears_failures_and_lockout(self):
        limiter = RateLimiter()
        for _ in range(RateLimiter.MAX_ATTEMPTS):
            limiter.record_attempt("alice", success=False)

        limiter.record_attempt("alice", success=True)

        assert limiter.is_locked_out("alice") == (False, None)
        assert limiter.get_backoff_delay("alice") == 0

    def test_lockout_expires(self):
        limiter = RateLimiter()
        limiter.LOCKOUT_DURATION = timedelta(0)
        for _ in range(RateLimiter.MAX_ATTEMPTS):
            limiter.record_attempt("alice", success=False)

        assert limiter.is_locked_out("alice") == (False, None)


# =============================================================================
# UserStore: sessions
# =============================================================================


class TestUserStoreSessions:
    """Session creation, validation, binding, expiry and invalidation."""

    def test_create_and_validate_session(self, store, user):
        session = store.create_session(user.user_id, ip_address="10.0.0.1", user_agent="pytest")

        assert session.session_id.startswith("sess_")
        assert session.is_valid
        assert session.expires_at - session.created_at == timedelta(hours=24)
        assert store.validate_session(session.token).user_id == user.user_id

        stored = store.get_session_by_token(session.token)
        assert stored.session_id == session.session_id
        assert stored.ip_address == "10.0.0.1"
        assert stored.user_agent == "pytest"

        data = stored.to_dict()
        assert data["session_id"] == session.session_id
        assert data["user_id"] == user.user_id
        assert data["is_active"] is True
        assert "token" not in data

    def test_custom_duration(self, store, user):
        session = store.create_session(user.user_id, duration_hours=24 * 30)
        assert session.expires_at - session.created_at == timedelta(days=30)

    def test_validation_touches_last_activity(self, store, user):
        session = store.create_session(user.user_id)
        before = store.get_session_by_token(session.token).last_activity

        store.validate_session(session.token)

        assert store.get_session_by_token(session.token).last_activity >= before

    def test_unknown_or_malformed_token(self, store, user):
        store.create_session(user.user_id)
        assert store.validate_session("not-a-real-token") is None
        assert store.validate_session("") is None
        assert store.get_session_by_token("not-a-real-token") is None

    def test_expired_session_is_rejected_and_deactivated(self, store, user):
        session = store.create_session(user.user_id, duration_hours=0)

        assert session.is_expired
        assert store.validate_session(session.token) is None
        assert store.get_user_sessions(user.user_id) == []

    def test_cleanup_expired_sessions(self, store, user):
        live = store.create_session(user.user_id)
        store.create_session(user.user_id, duration_hours=-1)
        store.create_session(user.user_id, duration_hours=-2)

        assert store.cleanup_expired_sessions() == 2
        assert store.cleanup_expired_sessions() == 0
        assert [s.session_id for s in store.get_user_sessions(user.user_id)] == [live.session_id]

    def test_ip_binding(self, store, user):
        session = store.create_session(user.user_id, ip_address="10.0.0.1")

        assert store.validate_session(session.token, ip_address="10.0.0.1") is not None
        assert store.validate_session(session.token, ip_address="10.9.9.9") is None

    def test_inactive_user_session_rejected(self, store, user):
        session = store.create_session(user.user_id)
        store.update_user(user.user_id, is_active=False)
        assert store.validate_session(session.token) is None

    def test_invalidate_session(self, store, user):
        first = store.create_session(user.user_id)
        second = store.create_session(user.user_id)

        assert store.invalidate_session(first.session_id) is True
        assert store.invalidate_session("sess_missing") is False

        assert store.validate_session(first.token) is None
        assert store.validate_session(second.token) is not None
        assert [s.session_id for s in store.get_user_sessions(user.user_id)] == [second.session_id]

    def test_invalidate_session_by_token(self, store, user):
        session = store.create_session(user.user_id)

        assert store.invalidate_session_by_token(session.token) is True
        assert store.invalidate_session_by_token("unknown") is False
        assert store.validate_session(session.token) is None

    def test_invalidate_all_user_sessions(self, store, user):
        other = store.create_user("bob", STRONG_PASSWORD)
        tokens = [store.create_session(user.user_id).token for _ in range(3)]
        other_session = store.create_session(other.user_id)

        assert store.invalidate_all_user_sessions(user.user_id) == 3

        assert all(store.validate_session(token) is None for token in tokens)
        assert store.validate_session(other_session.token).user_id == other.user_id

    def test_token_is_bound_to_signing_secret(self, tmp_path, monkeypatch):
        db_path = tmp_path / "shared-users.db"
        issuer = create_user_store(db_path)
        owner = issuer.create_user("alice", STRONG_PASSWORD)
        token = issuer.create_session(owner.user_id).token

        same_secret = create_user_store(db_path)
        monkeypatch.setenv("AGENT_OS_SESSION_SECRET", "a-different-secret")
        other_secret = create_user_store(db_path)
        try:
            assert same_secret.validate_session(token).user_id == owner.user_id
            # Same row in the database, but the HMAC binding no longer verifies.
            assert other_secret.validate_session(token) is None
            assert other_secret.validate_session(token, verify_binding=False) is not None
        finally:
            for opened in (issuer, same_secret, other_secret):
                opened.close()


class TestSessionSecretPersistence:
    """Session secret persisted (encrypted) next to the database when not in env."""

    def test_sessions_survive_restart(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AGENT_OS_SESSION_SECRET")
        db_path = tmp_path / "users.db"

        first = create_user_store(db_path)
        owner = first.create_user("alice", STRONG_PASSWORD)
        token = first.create_session(owner.user_id).token
        first.close()

        secret_file = tmp_path / ".session_secret"
        assert secret_file.exists()
        assert stat.S_IMODE(secret_file.stat().st_mode) == 0o600
        assert len(secret_file.read_bytes()) == 12 + 32 + 16  # nonce + ciphertext + GCM tag
        assert (tmp_path / "config" / ".machine_salt").exists()

        restarted = create_user_store(db_path)
        try:
            assert restarted.validate_session(token).user_id == owner.user_id
        finally:
            restarted.close()

    def test_legacy_secret_file_falls_back_to_ephemeral_secret(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AGENT_OS_SESSION_SECRET")
        db_path = tmp_path / "users.db"
        legacy_secret = b"\x01" * 64
        (tmp_path / ".session_secret").write_bytes(legacy_secret)

        first = create_user_store(db_path)
        owner = first.create_user("alice", STRONG_PASSWORD)
        token = first.create_session(owner.user_id).token
        second = create_user_store(db_path)
        try:
            assert first.validate_session(token) is not None
            assert second.validate_session(token) is None
            assert (tmp_path / ".session_secret").read_bytes() == legacy_secret
        finally:
            first.close()
            second.close()

    def test_in_memory_store_uses_ephemeral_secret(self, monkeypatch):
        monkeypatch.delenv("AGENT_OS_SESSION_SECRET")
        first = create_user_store()
        second = create_user_store()
        owner = first.create_user("alice", STRONG_PASSWORD)
        token = first.create_session(owner.user_id).token
        try:
            assert first.validate_session(token) is not None
            assert first._token_secret != second._token_secret
        finally:
            first.close()
            second.close()


# =============================================================================
# UserStore: scoped API keys
# =============================================================================


class TestScopedApiKeys:
    """create_api_key / validate_api_key / revoke_api_key / list_api_keys."""

    def test_create_and_validate(self, store, user):
        raw_key, api_key = store.create_api_key(
            user.user_id, [ApiKeyScope.READ_CHAT.value], description="ci"
        )

        assert raw_key.startswith("aos_")
        assert api_key.user_id == user.user_id
        assert api_key.key_hash == hashlib.sha256(raw_key.encode()).hexdigest()
        assert api_key.expires_at is None

        validated = store.validate_api_key(raw_key)
        assert validated.key_id == api_key.key_id
        assert validated.user_id == user.user_id
        assert validated.scopes == ["read:chat"]
        assert validated.description == "ci"
        assert validated.is_active is True

    def test_unknown_key(self, store, user):
        store.create_api_key(user.user_id, ["read:chat"])
        assert store.validate_api_key("aos_" + secrets.token_urlsafe(32)) is None
        assert store.validate_api_key("") is None

    def test_rejects_invalid_scope(self, store, user):
        with pytest.raises(AuthError) as exc_info:
            store.create_api_key(user.user_id, ["read:chat", "root"])
        assert exc_info.value.error_code == "INVALID_SCOPE"
        assert store.list_api_keys(user.user_id) == []

    def test_rejects_unknown_user(self, store):
        with pytest.raises(AuthError) as exc_info:
            store.create_api_key("user_missing", ["read:chat"])
        assert exc_info.value.error_code == "USER_NOT_FOUND"

    def test_expiry(self, store, user):
        live_raw, live_key = store.create_api_key(user.user_id, ["read:chat"], expires_in_days=30)
        expired_raw, expired_key = store.create_api_key(
            user.user_id, ["read:chat"], expires_in_days=-1
        )

        assert live_key.expires_at - live_key.created_at == timedelta(days=30)
        assert store.validate_api_key(live_raw) is not None
        assert expired_key.is_expired()
        assert store.validate_api_key(expired_raw) is None

    def test_revoke_only_by_owner(self, store, user):
        other = store.create_user("bob", STRONG_PASSWORD)
        raw_key, api_key = store.create_api_key(user.user_id, ["read:chat"])

        assert store.revoke_api_key(api_key.key_id, other.user_id) is False
        assert store.validate_api_key(raw_key) is not None

        assert store.revoke_api_key(api_key.key_id, user.user_id) is True
        assert store.validate_api_key(raw_key) is None
        assert store.revoke_api_key(api_key.key_id, user.user_id) is True  # idempotent update

    def test_list_masks_hash_and_omits_revoked(self, store, user):
        other = store.create_user("bob", STRONG_PASSWORD)
        _, kept = store.create_api_key(user.user_id, ["read:chat", "write:chat"], "kept")
        _, revoked = store.create_api_key(user.user_id, ["admin"], "revoked")
        store.create_api_key(other.user_id, ["read:memory"])
        store.revoke_api_key(revoked.key_id, user.user_id)

        listed = store.list_api_keys(user.user_id)

        assert [k.key_id for k in listed] == [kept.key_id]
        assert listed[0].key_hash == "***"
        assert listed[0].scopes == ["read:chat", "write:chat"]
        assert listed[0].description == "kept"

    def test_has_scope_and_admin_implies_all(self):
        limited = ScopedApiKey("k1", "u1", "h", scopes=["read:chat"])
        admin_key = ScopedApiKey("k2", "u1", "h", scopes=["admin"])

        assert limited.has_scope("read:chat")
        assert not limited.has_scope("write:chat")
        assert admin_key.has_scope("write:memory")

    def test_is_expired(self):
        key = ScopedApiKey("k1", "u1", "h", scopes=[])
        assert key.is_expired() is False
        key.expires_at = datetime.utcnow() - timedelta(seconds=1)
        assert key.is_expired() is True
        key.expires_at = datetime.utcnow() + timedelta(hours=1)
        assert key.is_expired() is False


class TestLegacyConfiguredApiKey:
    """validate_api_key() accepts the configured AGENT_OS_API_KEY as an admin key."""

    def test_configured_key_is_admin_scoped(self, store, caplog):
        with caplog.at_level(logging.WARNING, logger="src.web.auth"):
            api_key = store.validate_api_key(CONFIGURED_API_KEY)

        assert api_key.key_id == "legacy"
        assert api_key.user_id == "admin"
        assert api_key.scopes == [ApiKeyScope.ADMIN.value]
        assert api_key.description == "Legacy AGENT_OS_API_KEY"
        assert api_key.has_scope(ApiKeyScope.WRITE_MEMORY.value)
        assert "DEPRECATED" in caplog.text

    def test_near_miss_is_rejected(self, store):
        assert store.validate_api_key(CONFIGURED_API_KEY[:-1]) is None
        assert store.validate_api_key(CONFIGURED_API_KEY + "x") is None

    def test_disabled_when_no_key_configured(self, store, tmp_path):
        set_config(WebConfig(require_auth=False, api_key=None, debug=True, data_dir=tmp_path))
        assert store.validate_api_key(CONFIGURED_API_KEY) is None


# =============================================================================
# Store factories and DI integration
# =============================================================================


class TestUserStoreFactories:
    """create_user_store / get_user_store / set_user_store / reset_user_store."""

    def test_create_user_store_in_memory(self):
        created = create_user_store()
        try:
            assert created.db_path is None
            assert created.get_user_count() == 0
        finally:
            created.close()

    def test_create_user_store_failure(self, tmp_path):
        with pytest.raises(RuntimeError, match="Failed to initialize user store"):
            create_user_store(tmp_path / "missing-dir" / "users.db")

    def test_close_marks_store_uninitialized(self):
        created = create_user_store()
        created.close()
        with pytest.raises(RuntimeError, match="not initialized"):
            created.create_user("alice", STRONG_PASSWORD)

    def test_set_user_store_overrides_get_user_store(self, store):
        assert get_user_store() is store

    def test_default_store_lives_in_configured_data_dir(self, web_config):
        _container.clear_override("user_store")
        reset_user_store()

        first = get_user_store()
        try:
            db_path = web_config.data_dir / "users.db"
            assert first.db_path == db_path
            assert db_path.exists()
            assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
            assert get_user_store() is first

            reset_user_store()
            second = get_user_store()
            assert second is not first
            second.close()
        finally:
            first.close()


# =============================================================================
# auth_helpers: token extraction and API key auth
# =============================================================================


class TestTokenExtraction:
    """_extract_token and _try_api_key_auth."""

    def test_explicit_session_token_wins(self):
        request = _make_request(_bearer("from-header"), cookies={"session_token": "from-cookie"})
        assert _extract_token(request, "from-param") == "from-param"

    def test_cookie_before_header(self):
        request = _make_request(_bearer("from-header"), cookies={"session_token": "from-cookie"})
        assert _extract_token(request) == "from-cookie"

    def test_bearer_header(self):
        assert _extract_token(_make_request(_bearer("from-header"))) == "from-header"

    @pytest.mark.parametrize("header", ["Basic dXNlcjpwYXNz", "Token abc", "Bearer"])
    def test_non_bearer_header_ignored(self, header):
        assert _extract_token(_make_request({"Authorization": header})) is None

    def test_no_credentials(self):
        assert _extract_token(_make_request()) is None

    def test_api_key_auth_requires_bearer_header(self, store):
        assert _try_api_key_auth(_make_request()) is None
        request = _make_request({"Authorization": f"Basic {CONFIGURED_API_KEY}"})
        assert _try_api_key_auth(request) is None

    def test_api_key_auth_accepts_configured_key(self, store):
        user_id, api_key = _try_api_key_auth(_make_request(_bearer(CONFIGURED_API_KEY)))
        assert user_id == "admin"
        assert api_key.scopes == ["admin"]

    def test_api_key_auth_accepts_scoped_key(self, store, user):
        raw_key, created = store.create_api_key(user.user_id, ["read:memory"])

        user_id, api_key = _try_api_key_auth(_make_request(_bearer(raw_key)))

        assert user_id == user.user_id
        assert api_key.key_id == created.key_id

    def test_api_key_auth_rejects_other_tokens(self, store, user):
        session = store.create_session(user.user_id)
        for token in ("wrong", "aos_" + secrets.token_urlsafe(32), session.token):
            assert _try_api_key_auth(_make_request(_bearer(token))) is None


# =============================================================================
# auth_helpers: require_authenticated_user / require_admin_user
# =============================================================================


class TestAuthDependencies:
    """The require_* dependencies, called directly with real requests."""

    # --- Regression: configured AGENT_OS_API_KEY (no aos_ prefix) -------------

    def test_configured_api_key_authenticates(self, store):
        assert not CONFIGURED_API_KEY.startswith("aos_")
        request = _make_request(_bearer(CONFIGURED_API_KEY))

        assert require_authenticated_user(request, session_token=None) == "admin"
        assert request.state.api_key_scopes == ["admin"]

    def test_configured_api_key_passes_admin_check(self, store):
        request = _make_request(_bearer(CONFIGURED_API_KEY))

        assert require_admin_user(request, session_token=None) == "admin"
        assert request.state.api_key_scopes == ["admin"]

    # --- Scoped keys -------------------------------------------------------------

    def test_scoped_key_without_admin_scope(self, store, user):
        raw_key, _ = store.create_api_key(user.user_id, ["read:chat", "write:chat"])

        request = _make_request(_bearer(raw_key))
        assert require_authenticated_user(request, session_token=None) == user.user_id
        assert request.state.api_key_scopes == ["read:chat", "write:chat"]

        with pytest.raises(HTTPException) as exc_info:
            require_admin_user(_make_request(_bearer(raw_key)), session_token=None)
        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "API key lacks admin scope"

    def test_scoped_key_with_admin_scope(self, store, user):
        raw_key, _ = store.create_api_key(user.user_id, ["admin"])
        request = _make_request(_bearer(raw_key))

        assert require_admin_user(request, session_token=None) == user.user_id
        assert request.state.api_key_scopes == ["admin"]

    def test_revoked_key_is_rejected(self, store, user):
        raw_key, api_key = store.create_api_key(user.user_id, ["admin"])
        store.revoke_api_key(api_key.key_id, user.user_id)

        for dependency in (require_authenticated_user, require_admin_user):
            with pytest.raises(HTTPException) as exc_info:
                dependency(_make_request(_bearer(raw_key)), session_token=None)
            assert exc_info.value.status_code == 401

    # --- Sessions ----------------------------------------------------------------

    def test_session_via_bearer_header(self, store, user):
        session = store.create_session(user.user_id)
        request = _make_request(_bearer(session.token))

        assert require_authenticated_user(request, session_token=None) == user.user_id
        assert request.state.api_key_scopes is None

    def test_session_via_cookie(self, store, user):
        session = store.create_session(user.user_id)
        request = _make_request(cookies={"session_token": session.token})
        assert require_authenticated_user(request, session_token=session.token) == user.user_id

    def test_regular_user_session_is_not_admin(self, store, user):
        session = store.create_session(user.user_id)

        with pytest.raises(HTTPException) as exc_info:
            require_admin_user(_make_request(_bearer(session.token)), session_token=None)
        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "Admin privileges required for this operation"

    def test_admin_user_session_is_admin(self, store, admin):
        session = store.create_session(admin.user_id)
        request = _make_request(_bearer(session.token))

        assert require_admin_user(request, session_token=None) == admin.user_id
        assert request.state.api_key_scopes is None

    # --- Failures ----------------------------------------------------------------

    @pytest.mark.parametrize(
        "dependency, detail",
        [
            (require_authenticated_user, "Authentication required"),
            (require_admin_user, "Authentication required for admin operations"),
        ],
    )
    def test_missing_credentials(self, store, dependency, detail):
        with pytest.raises(HTTPException) as exc_info:
            dependency(_make_request(), session_token=None)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == detail

    @pytest.mark.parametrize("dependency", [require_authenticated_user, require_admin_user])
    @pytest.mark.parametrize(
        "token",
        [
            "wrong-token",
            "aos_unknown_scoped_key",
            CONFIGURED_API_KEY[:-1],
            CONFIGURED_API_KEY.upper() + "Z",
        ],
    )
    def test_unknown_bearer_token(self, store, dependency, token):
        with pytest.raises(HTTPException) as exc_info:
            dependency(_make_request(_bearer(token)), session_token=None)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Session expired or invalid"

    def test_expired_session(self, store, user):
        session = store.create_session(user.user_id, duration_hours=0)
        with pytest.raises(HTTPException) as exc_info:
            require_authenticated_user(_make_request(_bearer(session.token)), session_token=None)
        assert exc_info.value.status_code == 401


# =============================================================================
# auth_helpers: require_scope
# =============================================================================


class TestRequireScope:
    """require_scope() as a direct check and as a route dependency."""

    @staticmethod
    def _request_with_scopes(scopes):
        request = _make_request()
        request.state.api_key_scopes = scopes
        return request

    def test_session_auth_grants_all_scopes(self):
        assert require_scope("write:memory")(self._request_with_scopes(None)) is None

    def test_matching_scope(self):
        check = require_scope("write:chat")
        assert check(self._request_with_scopes(["read:chat", "write:chat"])) is None

    def test_admin_scope_grants_everything(self):
        assert require_scope("write:memory")(self._request_with_scopes(["admin"])) is None

    def test_missing_scope(self):
        with pytest.raises(HTTPException) as exc_info:
            require_scope("write:chat")(self._request_with_scopes(["read:chat"]))
        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "API key lacks required scope: write:chat"

    def test_route_rejects_key_without_scope(self, helper_client, store, user):
        raw_key, _ = store.create_api_key(user.user_id, ["read:chat"])

        response = helper_client.post("/chat", headers=_bearer(raw_key))

        assert response.status_code == 403
        assert response.json()["detail"] == "API key lacks required scope: write:chat"

    def test_route_accepts_key_with_scope(self, helper_client, store, user):
        raw_key, _ = store.create_api_key(user.user_id, ["write:chat"])

        response = helper_client.post("/chat", headers=_bearer(raw_key))

        assert response.status_code == 200
        assert response.json() == {"user_id": user.user_id}

    def test_route_accepts_admin_keys(self, helper_client, store, user):
        raw_key, _ = store.create_api_key(user.user_id, ["admin"])

        assert helper_client.post("/chat", headers=_bearer(raw_key)).status_code == 200
        configured = helper_client.post("/chat", headers=_bearer(CONFIGURED_API_KEY))
        assert configured.status_code == 200
        assert configured.json() == {"user_id": "admin"}

    def test_route_accepts_session(self, helper_client, store, user):
        session = store.create_session(user.user_id)
        response = helper_client.post("/chat", headers=_bearer(session.token))
        assert response.status_code == 200

    def test_route_requires_authentication(self, helper_client):
        assert helper_client.post("/chat").status_code == 401


# =============================================================================
# auth_helpers: authenticate_websocket
# =============================================================================


class TestWebSocketAuthentication:
    """authenticate_websocket() via a real WebSocket handshake."""

    def test_token_query_parameter(self, helper_client, store, user):
        token = store.create_session(user.user_id).token
        with helper_client.websocket_connect(f"/ws?token={token}") as websocket:
            assert websocket.receive_json() == {"user_id": user.user_id}

    def test_session_cookie(self, helper_client, store, user):
        token = store.create_session(user.user_id).token
        headers = {"cookie": f"session_token={token}"}
        with helper_client.websocket_connect("/ws", headers=headers) as websocket:
            assert websocket.receive_json() == {"user_id": user.user_id}

    @pytest.mark.parametrize("path", ["/ws", "/ws?token=not-a-session"])
    def test_rejects_missing_or_invalid_token(self, helper_client, store, path):
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with helper_client.websocket_connect(path):
                pass
        assert exc_info.value.code == 4001

    def test_rejects_invalidated_session(self, helper_client, store, user):
        session = store.create_session(user.user_id)
        store.invalidate_session(session.session_id)
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with helper_client.websocket_connect(f"/ws?token={session.token}"):
                pass
        assert exc_info.value.code == 4001

    def test_chat_websocket_requires_session(self, client, store, user):
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/api/chat/ws"):
                pass
        assert exc_info.value.code == 4001

        token = store.create_session(user.user_id).token
        with client.websocket_connect(f"/api/chat/ws?token={token}") as websocket:
            assert websocket.receive_json()["type"] == "connected"


# =============================================================================
# routes/auth.py endpoints
# =============================================================================


class TestRegisterEndpoint:
    """POST /api/auth/register."""

    def test_register_creates_user_and_session(self, client, store, intent_store):
        response = client.post(
            "/api/auth/register",
            json={
                "username": "alice",
                "password": STRONG_PASSWORD,
                "email": "alice@example.com",
                "display_name": "Alice",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["message"] == "Registration successful"
        assert body["user"]["username"] == "alice"
        assert body["user"]["display_name"] == "Alice"
        assert body["user"]["role"] == "user"
        assert body["expires_at"]

        created = store.get_user_by_username("alice")
        assert created.email == "alice@example.com"
        assert store.validate_session(body["token"]).user_id == created.user_id

        cookie = response.headers["set-cookie"]
        assert f"session_token={body['token']}" in cookie
        assert "HttpOnly" in cookie
        assert "Secure" in cookie
        assert "samesite=lax" in cookie.lower()
        assert "Max-Age=86400" in cookie

        entries = intent_store.get_user_entries(created.user_id)
        assert [e.intent_type.name for e in entries] == ["AUTH_REGISTER"]

    def test_register_cannot_self_assign_admin(self, client, store):
        response = client.post(
            "/api/auth/register",
            json={"username": "mallory", "password": STRONG_PASSWORD, "role": "admin"},
        )

        assert response.status_code == 200
        assert response.json()["user"]["role"] == "user"
        assert store.get_user_by_username("mallory").role == UserRole.USER

    def test_register_enforces_password_policy(self, client, store):
        response = client.post(
            "/api/auth/register",
            json={"username": "alice", "password": "lowercase-only-1!"},
        )

        assert response.status_code == 400
        assert "uppercase" in response.json()["detail"]
        assert store.get_user_by_username("alice") is None

    def test_register_validates_request_body(self, client):
        too_short = client.post("/api/auth/register", json={"username": "al", "password": "x"})
        missing = client.post("/api/auth/register", json={"username": "alice"})

        assert too_short.status_code == 422
        assert missing.status_code == 422

    def test_register_duplicate_username(self, client, user):
        response = client.post(
            "/api/auth/register", json={"username": "alice", "password": STRONG_PASSWORD}
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "Username already exists"

    def test_register_duplicate_email(self, client, user):
        response = client.post(
            "/api/auth/register",
            json={"username": "alice2", "password": STRONG_PASSWORD, "email": "alice@example.com"},
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "Email already exists"


class TestLoginLogoutEndpoints:
    """POST /api/auth/login, /logout and GET /me, /status."""

    def test_login_with_username(self, client, store, user):
        response = client.post(
            "/api/auth/login", json={"username": "alice", "password": STRONG_PASSWORD}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["user"]["user_id"] == user.user_id
        assert body["user"]["last_login"] is not None
        assert store.validate_session(body["token"]).user_id == user.user_id

        cookie = response.headers["set-cookie"]
        assert "Secure" in cookie and "HttpOnly" in cookie
        assert "Max-Age=86400" in cookie

    def test_login_with_email(self, client, user):
        response = client.post(
            "/api/auth/login",
            json={"username": "alice@example.com", "password": STRONG_PASSWORD},
        )
        assert response.status_code == 200
        assert response.json()["user"]["user_id"] == user.user_id

    def test_remember_me_extends_session(self, client, store, user):
        response = client.post(
            "/api/auth/login",
            json={"username": "alice", "password": STRONG_PASSWORD, "remember_me": True},
        )

        assert response.status_code == 200
        assert "Max-Age=2592000" in response.headers["set-cookie"]
        session = store.get_session_by_token(response.json()["token"])
        assert session.expires_at - session.created_at == timedelta(days=30)

    @pytest.mark.parametrize(
        "username, password",
        [("alice", "Wrong-Passw0rd!"), ("nobody", STRONG_PASSWORD)],
    )
    def test_login_rejects_bad_credentials(self, client, user, username, password):
        response = client.post("/api/auth/login", json={"username": username, "password": password})

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid username or password"
        assert "set-cookie" not in response.headers

    def test_login_rejects_inactive_account(self, client, store, user):
        store.update_user(user.user_id, is_active=False)
        response = client.post(
            "/api/auth/login", json={"username": "alice", "password": STRONG_PASSWORD}
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Account is inactive"

    def test_login_lockout(self, client, user):
        for _ in range(RateLimiter.MAX_ATTEMPTS):
            client.post("/api/auth/login", json={"username": "alice", "password": "Wrong-Pa55!"})

        response = client.post(
            "/api/auth/login", json={"username": "alice", "password": STRONG_PASSWORD}
        )

        assert response.status_code == 401
        assert "Account temporarily locked" in response.json()["detail"]

    def test_session_cookie_authenticates_me(self, client, user):
        client.post("/api/auth/login", json={"username": "alice", "password": STRONG_PASSWORD})

        response = client.get("/api/auth/me")

        assert response.status_code == 200
        assert response.json()["user_id"] == user.user_id

    def test_me_with_bearer_token(self, client, user):
        token = _login(client, "alice")

        response = client.get("/api/auth/me", headers=_bearer(token))

        assert response.status_code == 200
        body = response.json()
        assert body["username"] == "alice"
        assert body["email"] == "alice@example.com"
        assert body["display_name"] == "alice"
        assert body["role"] == "user"
        assert body["last_login"] is not None

    def test_me_requires_valid_session(self, client):
        missing = client.get("/api/auth/me")
        invalid = client.get("/api/auth/me", headers=_bearer("not-a-session"))

        assert missing.status_code == 401
        assert missing.json()["detail"] == "Not authenticated"
        assert invalid.status_code == 401
        assert invalid.json()["detail"] == "Session expired or invalid"

    def test_logout_invalidates_session(self, client, store, user):
        token = _login(client, "alice")

        response = client.post("/api/auth/logout", headers=_bearer(token))

        assert response.status_code == 200
        assert response.json()["success"] is True
        assert "Max-Age=0" in response.headers["set-cookie"]
        assert store.validate_session(token) is None
        assert client.get("/api/auth/me", headers=_bearer(token)).status_code == 401

    def test_logout_with_cookie(self, client, store, user):
        response = client.post(
            "/api/auth/login", json={"username": "alice", "password": STRONG_PASSWORD}
        )
        token = response.json()["token"]

        assert client.post("/api/auth/logout").status_code == 200
        assert store.validate_session(token) is None

    @pytest.mark.parametrize("headers", [{}, _bearer("not-a-session")])
    def test_logout_without_valid_session(self, client, headers):
        response = client.post("/api/auth/logout", headers=headers)
        assert response.status_code == 200
        assert response.json() == {"success": True, "message": "Logged out successfully"}

    @pytest.mark.parametrize("headers", [{}, _bearer("not-a-session")])
    def test_status_unauthenticated(self, client, headers):
        response = client.get("/api/auth/status", headers=headers)

        assert response.status_code == 200
        assert response.json() == {"authenticated": False, "user": None, "require_auth": True}

    def test_status_authenticated(self, client, user):
        token = _login(client, "alice")

        response = client.get("/api/auth/status", headers=_bearer(token))

        body = response.json()
        assert body["authenticated"] is True
        assert body["user"]["user_id"] == user.user_id
        assert body["require_auth"] is True


class TestAccountEndpoints:
    """PUT /profile, POST /change-password, sessions, /logout-all, /users/count."""

    def test_update_profile(self, client, store, user):
        token = _login(client, "alice")

        response = client.put(
            "/api/auth/profile",
            json={"display_name": "Alice A.", "email": "alice.a@example.com"},
            headers=_bearer(token),
        )

        assert response.status_code == 200
        assert response.json()["user"]["display_name"] == "Alice A."
        assert response.json()["user"]["email"] == "alice.a@example.com"
        assert store.get_user(user.user_id).email == "alice.a@example.com"

    def test_update_profile_email_conflict(self, client, store, user):
        store.create_user("bob", STRONG_PASSWORD, email="bob@example.com")
        token = _login(client, "alice")

        response = client.put(
            "/api/auth/profile", json={"email": "bob@example.com"}, headers=_bearer(token)
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Email already exists"
        assert store.get_user(user.user_id).email == "alice@example.com"

    def test_change_password(self, client, store, user):
        token = _login(client, "alice")

        response = client.post(
            "/api/auth/change-password",
            json={"current_password": STRONG_PASSWORD, "new_password": NEW_PASSWORD},
            headers=_bearer(token),
        )

        assert response.status_code == 200
        assert response.json()["success"] is True
        # Every session, including the one used for the change, is revoked.
        assert client.get("/api/auth/me", headers=_bearer(token)).status_code == 401
        old = client.post(
            "/api/auth/login", json={"username": "alice", "password": STRONG_PASSWORD}
        )
        assert old.status_code == 401
        assert _login(client, "alice", NEW_PASSWORD)

    def test_change_password_requires_current_password(self, client, user):
        token = _login(client, "alice")

        response = client.post(
            "/api/auth/change-password",
            json={"current_password": "Wrong-Passw0rd!", "new_password": NEW_PASSWORD},
            headers=_bearer(token),
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Current password is incorrect"
        assert client.get("/api/auth/me", headers=_bearer(token)).status_code == 200

    def test_change_password_enforces_policy(self, client, user):
        token = _login(client, "alice")

        response = client.post(
            "/api/auth/change-password",
            json={"current_password": STRONG_PASSWORD, "new_password": "nouppercase1!"},
            headers=_bearer(token),
        )

        assert response.status_code == 400
        assert "uppercase" in response.json()["detail"]
        assert client.get("/api/auth/me", headers=_bearer(token)).status_code == 200

    def test_list_and_revoke_sessions(self, client, store, user):
        first = _login(client, "alice")
        second = _login(client, "alice")

        listed = client.get("/api/auth/sessions", headers=_bearer(first))
        assert listed.status_code == 200
        assert listed.json()["count"] == 2

        second_id = store.get_session_by_token(second).session_id
        response = client.delete(f"/api/auth/sessions/{second_id}", headers=_bearer(first))

        assert response.status_code == 200
        assert client.get("/api/auth/me", headers=_bearer(second)).status_code == 401
        assert client.get("/api/auth/me", headers=_bearer(first)).status_code == 200
        assert client.get("/api/auth/sessions", headers=_bearer(first)).json()["count"] == 1

    def test_cannot_revoke_another_users_session(self, client, store, user):
        bob = store.create_user("bob", STRONG_PASSWORD)
        bob_session = store.create_session(bob.user_id)
        token = _login(client, "alice")

        response = client.delete(
            f"/api/auth/sessions/{bob_session.session_id}", headers=_bearer(token)
        )

        assert response.status_code == 404
        assert store.validate_session(bob_session.token) is not None

    def test_logout_all(self, client, store, user):
        tokens = [_login(client, "alice"), _login(client, "alice")]

        response = client.post("/api/auth/logout-all", headers=_bearer(tokens[0]))

        assert response.status_code == 200
        assert response.json()["sessions_invalidated"] == 2
        for token in tokens:
            assert client.get("/api/auth/me", headers=_bearer(token)).status_code == 401

    @pytest.mark.parametrize(
        "method, path, payload",
        [
            ("put", "/api/auth/profile", {"display_name": "x"}),
            (
                "post",
                "/api/auth/change-password",
                {"current_password": STRONG_PASSWORD, "new_password": NEW_PASSWORD},
            ),
            ("get", "/api/auth/sessions", None),
            ("delete", "/api/auth/sessions/sess_x", None),
            ("post", "/api/auth/logout-all", None),
        ],
    )
    def test_account_endpoints_require_session(self, client, method, path, payload):
        kwargs = {"json": payload} if payload is not None else {}

        missing = getattr(client, method)(path, **kwargs)
        invalid = getattr(client, method)(path, headers=_bearer("not-a-session"), **kwargs)

        assert missing.status_code == 401
        assert missing.json()["detail"] == "Not authenticated"
        assert invalid.status_code == 401
        assert invalid.json()["detail"] == "Session expired or invalid"

    def test_user_count(self, client, store, user, admin):
        bob = store.create_user("bob", STRONG_PASSWORD)
        store.update_user(bob.user_id, is_active=False)

        assert client.get("/api/auth/users/count").json() == {"count": 2}
        for username in ("root", "alice"):
            token = _login(client, username)
            response = client.get("/api/auth/users/count", headers=_bearer(token))
            assert response.json() == {"count": 2}


# =============================================================================
# Admin access through the real app (regression for the AGENT_OS_API_KEY bug)
# =============================================================================


class TestAdminAccessEndToEnd:
    """Admin-only endpoints (require_admin_user) through the full application."""

    ADMIN_ENDPOINT = "/api/system/settings"
    USER_ENDPOINT = "/api/system/info"

    def test_config_requires_api_key_when_auth_enabled(self, web_config):
        web_config.validate()  # the fixture's key satisfies validation
        with pytest.raises(ConfigurationError):
            WebConfig(require_auth=True, api_key=None).validate()

    def test_configured_api_key_reaches_admin_endpoint(self, client):
        assert not CONFIGURED_API_KEY.startswith("aos_")

        response = client.get(self.ADMIN_ENDPOINT, headers=_bearer(CONFIGURED_API_KEY))

        assert response.status_code == 200
        assert isinstance(response.json(), list) and response.json()

    def test_configured_api_key_reaches_other_admin_endpoints(self, client):
        response = client.get("/api/system/logs", headers=_bearer(CONFIGURED_API_KEY))
        assert response.status_code == 200

    def test_configured_api_key_reaches_authenticated_endpoint(self, client):
        response = client.get(self.USER_ENDPOINT, headers=_bearer(CONFIGURED_API_KEY))
        assert response.status_code == 200
        assert "version" in response.json()

    def test_regular_user_session_is_forbidden(self, client, user):
        token = _login(client, "alice")

        forbidden = client.get(self.ADMIN_ENDPOINT, headers=_bearer(token))
        allowed = client.get(self.USER_ENDPOINT, headers=_bearer(token))

        assert forbidden.status_code == 403
        assert forbidden.json()["detail"] == "Admin privileges required for this operation"
        assert allowed.status_code == 200

    def test_registered_user_cookie_is_forbidden(self, client):
        registered = client.post(
            "/api/auth/register", json={"username": "newbie", "password": STRONG_PASSWORD}
        )
        assert registered.status_code == 200

        response = client.get(self.ADMIN_ENDPOINT)  # session cookie from registration

        assert response.status_code == 403

    def test_no_credentials_is_unauthorized(self, client):
        for path in (self.ADMIN_ENDPOINT, self.USER_ENDPOINT):
            response = client.get(path)
            assert response.status_code == 401

    @pytest.mark.parametrize(
        "headers",
        [
            _bearer("wrong-token"),
            _bearer("aos_" + secrets.token_urlsafe(32)),
            _bearer(CONFIGURED_API_KEY[:-1]),
            _bearer(""),
            {"Authorization": f"Basic {CONFIGURED_API_KEY}"},
            {"Authorization": CONFIGURED_API_KEY},
            # Non-ASCII token: compare_digest on str raised TypeError (500)
            {"Authorization": b"Bearer caf\xc3\xa9"},
        ],
    )
    def test_wrong_credentials_are_unauthorized(self, client, headers):
        assert client.get(self.ADMIN_ENDPOINT, headers=headers).status_code == 401
        assert client.get(self.USER_ENDPOINT, headers=headers).status_code == 401

    def test_admin_role_session_is_allowed(self, client, admin):
        token = _login(client, "root")
        assert client.get(self.ADMIN_ENDPOINT, headers=_bearer(token)).status_code == 200

    def test_scoped_keys(self, client, store, user):
        limited_key, _ = store.create_api_key(user.user_id, ["read:chat"])
        admin_key, _ = store.create_api_key(user.user_id, ["admin"])

        limited = client.get(self.ADMIN_ENDPOINT, headers=_bearer(limited_key))
        assert limited.status_code == 403
        assert limited.json()["detail"] == "API key lacks admin scope"
        assert client.get(self.USER_ENDPOINT, headers=_bearer(limited_key)).status_code == 200
        assert client.get(self.ADMIN_ENDPOINT, headers=_bearer(admin_key)).status_code == 200

    def test_revoked_and_expired_keys_are_unauthorized(self, client, store, user):
        revoked_key, revoked = store.create_api_key(user.user_id, ["admin"])
        store.revoke_api_key(revoked.key_id, user.user_id)
        expired_key, _ = store.create_api_key(user.user_id, ["admin"], expires_in_days=-1)

        for raw_key in (revoked_key, expired_key):
            response = client.get(self.ADMIN_ENDPOINT, headers=_bearer(raw_key))
            assert response.status_code == 401

    def test_api_key_takes_precedence_over_cookie(self, client, user):
        client.post("/api/auth/login", json={"username": "alice", "password": STRONG_PASSWORD})
        assert client.get(self.ADMIN_ENDPOINT).status_code == 403  # cookie: regular user

        response = client.get(self.ADMIN_ENDPOINT, headers=_bearer(CONFIGURED_API_KEY))

        assert response.status_code == 200

    def test_old_key_rejected_when_no_key_configured(self, client, web_config):
        web_config.api_key = None
        response = client.get(self.ADMIN_ENDPOINT, headers=_bearer(CONFIGURED_API_KEY))
        assert response.status_code == 401
