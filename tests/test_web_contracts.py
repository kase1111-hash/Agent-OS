"""
Tests for the Learning Contracts web API (src/web/routes/contracts.py).

Exercises the routes end to end against the real src.contracts ContractStore,
persisted in an isolated temporary data directory, with real user sessions.
"""

from datetime import timedelta
from types import SimpleNamespace

import pytest

try:
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")

PASSWORD = "Contract-Test-Passw0rd!"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """App with an isolated data dir, a fresh contracts store and two logged-in users."""
    monkeypatch.setenv("AGENT_OS_REQUIRE_AUTH", "false")
    monkeypatch.setenv("AGENT_OS_WEB_DEBUG", "true")

    from src.web.app import create_app
    from src.web.auth import create_user_store, set_user_store
    from src.web.config import WebConfig, set_config
    from src.web.routes import contracts as contracts_routes

    config = WebConfig(
        debug=True,
        require_auth=False,
        rate_limit_enabled=False,
        static_dir=tmp_path / "static",
        templates_dir=tmp_path / "templates",
        data_dir=tmp_path / "data",
    )
    set_config(config)

    user_store = create_user_store(tmp_path / "users.db")
    set_user_store(user_store)
    contracts_routes.reset_store()

    def login(username):
        user = user_store.create_user(username, PASSWORD)
        session = user_store.create_session(user.user_id, bind_to_ip=False)
        return user.user_id, {"Authorization": f"Bearer {session.token}"}

    alice_id, alice = login("alice")
    bob_id, bob = login("bob")

    yield SimpleNamespace(
        client=TestClient(create_app(config)),
        routes=contracts_routes,
        data_dir=config.data_dir,
        alice_id=alice_id,
        alice=alice,
        bob_id=bob_id,
        bob=bob,
    )

    contracts_routes.reset_store()
    user_store.close()


def _create(env, headers, **body):
    body.setdefault("contract_type", "EPISODIC")
    response = env.client.post("/api/contracts/", json=body, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def _from_template(env, headers, template_id, **body):
    response = env.client.post(
        "/api/contracts/from-template",
        json={"template_id": template_id, **body},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return response.json()


def _list(env, headers, **params):
    response = env.client.get("/api/contracts/", params=params, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


class TestAuth:
    def test_endpoints_require_authentication(self, env):
        assert env.client.get("/api/contracts/").status_code == 401
        assert env.client.get("/api/contracts/stats").status_code == 401
        assert env.client.get("/api/contracts/templates").status_code == 401
        response = env.client.post("/api/contracts/", json={"contract_type": "EPISODIC"})
        assert response.status_code == 401

    def test_invalid_token_rejected(self, env):
        headers = {"Authorization": "Bearer not-a-real-token"}
        assert env.client.get("/api/contracts/", headers=headers).status_code == 401


class TestTemplatesAndTypes:
    def test_templates_match_real_module(self, env):
        from src.contracts import CONTRACT_TEMPLATES

        response = env.client.get("/api/contracts/templates", headers=env.alice)
        assert response.status_code == 200
        templates = {t["id"]: t for t in response.json()}

        assert set(templates) == set(CONTRACT_TEMPLATES)
        assert {"coding", "journaling", "restricted", "strategy"} <= set(templates)

        coding = templates["coding"]
        assert coding["name"] == "Coding Assistant"
        assert coding["contract_type"] == "PROCEDURAL"
        assert coding["description"]
        assert "coding" in coding["default_domains"]
        assert coding["default_duration_days"] is None

        # The gaming template expires after 24 hours.
        assert templates["gaming"]["default_duration_days"] == 1
        assert templates["restricted"]["contract_type"] == "PROHIBITED"
        assert templates["work_projects"]["name"] == "Work Projects"

    def test_get_single_template(self, env):
        response = env.client.get("/api/contracts/templates/journaling", headers=env.alice)
        assert response.status_code == 200
        assert response.json()["id"] == "journaling"
        assert response.json()["contract_type"] == "EPISODIC"

        response = env.client.get("/api/contracts/templates/nope", headers=env.alice)
        assert response.status_code == 404

    def test_contract_types(self, env):
        response = env.client.get("/api/contracts/types", headers=env.alice)
        assert response.status_code == 200
        types = {t["name"]: t for t in response.json()}

        assert set(types) == {"OBSERVATION", "EPISODIC", "PROCEDURAL", "STRATEGIC", "PROHIBITED"}
        assert types["OBSERVATION"]["allows_storage"] is False
        assert types["PROHIBITED"]["allows_storage"] is False
        assert types["EPISODIC"]["allows_storage"] is True
        assert types["EPISODIC"]["allows_generalization"] is False
        assert types["STRATEGIC"]["allows_long_term_patterns"] is True


class TestCreate:
    def test_create_direct_contract(self, env):
        contract = _create(
            env,
            env.alice,
            contract_type="PROCEDURAL",
            domains=["coding", " rust ", ""],
            duration_days=30,
            description="Learn my Rust style",
            metadata={"source": "test"},
        )

        assert contract["id"].startswith("LC-")
        assert contract["user_id"] == env.alice_id
        assert contract["contract_type"] == "PROCEDURAL"
        assert contract["status"] == "ACTIVE"
        assert contract["domains"] == ["coding", "rust"]
        assert contract["description"] == "Learn my Rust style"
        assert contract["metadata"] == {"source": "test"}
        assert contract["expires_at"] is not None

        listed = _list(env, env.alice)
        assert [c["id"] for c in listed] == [contract["id"]]

        detail = env.client.get(f"/api/contracts/{contract['id']}", headers=env.alice)
        assert detail.status_code == 200
        assert detail.json()["domains"] == ["coding", "rust"]

    def test_every_listed_type_is_accepted(self, env):
        types = env.client.get("/api/contracts/types", headers=env.alice).json()
        for contract_type in types:
            contract = _create(env, env.alice, contract_type=contract_type["name"])
            assert contract["contract_type"] == contract_type["name"]
        assert len(_list(env, env.alice)) == len(types)

    def test_body_user_id_is_ignored(self, env):
        contract = _create(env, env.alice, user_id=env.bob_id)
        assert contract["user_id"] == env.alice_id
        assert _list(env, env.bob) == []

    def test_invalid_contract_type(self, env):
        for bad_type in ("NOT_A_TYPE", "FULL_CONSENT"):
            response = env.client.post(
                "/api/contracts/", json={"contract_type": bad_type}, headers=env.alice
            )
            assert response.status_code == 400
            assert "Valid types" in response.json()["detail"]

    def test_invalid_duration(self, env):
        response = env.client.post(
            "/api/contracts/",
            json={"contract_type": "EPISODIC", "duration_days": -5},
            headers=env.alice,
        )
        assert response.status_code == 400
        assert _list(env, env.alice) == []

    def test_create_from_template_defaults(self, env):
        from src.contracts import CONTRACT_TEMPLATES

        contract = _from_template(env, env.alice, "coding")

        assert contract["user_id"] == env.alice_id
        assert contract["contract_type"] == "PROCEDURAL"
        assert contract["status"] == "ACTIVE"
        assert contract["domains"] == sorted(CONTRACT_TEMPLATES["coding"].scope.domains)
        assert contract["description"] == CONTRACT_TEMPLATES["coding"].description
        assert contract["metadata"]["template"] == "coding"
        assert contract["expires_at"] is None
        assert [c["id"] for c in _list(env, env.alice)] == [contract["id"]]

    def test_create_from_template_overrides(self, env):
        from src.contracts import CONTRACT_TEMPLATES

        contract = _from_template(env, env.alice, "journaling", domains=["dreams"], duration_days=7)
        assert contract["contract_type"] == "EPISODIC"
        assert contract["domains"] == ["dreams"]
        assert contract["expires_at"] is not None

        # The shared template definition is not mutated by the override.
        assert "dreams" not in CONTRACT_TEMPLATES["journaling"].scope.domains

        # Template default duration (24h) applies when none is given.
        gaming = _from_template(env, env.alice, "gaming")
        assert gaming["expires_at"] is not None

    def test_create_from_template_records_scope(self, env):
        from src.contracts import LearningScope

        contract = _from_template(env, env.alice, "strategy", domains=["research"])
        stored = env.routes.get_store()._store.get_contract(contract["id"])
        assert stored.scope.scope_type == LearningScope.DOMAIN_SPECIFIC
        assert stored.scope.domains == {"research"}
        assert "credentials" in stored.scope.excluded_domains

    def test_create_from_unknown_template(self, env):
        response = env.client.post(
            "/api/contracts/from-template", json={"template_id": "nope"}, headers=env.alice
        )
        assert response.status_code == 404


class TestUserIsolation:
    def test_users_only_see_their_own_contracts(self, env):
        alice_contract = _create(env, env.alice, domains=["alice-stuff"])
        bob_contract = _from_template(env, env.bob, "study")

        assert [c["id"] for c in _list(env, env.alice)] == [alice_contract["id"]]
        assert [c["id"] for c in _list(env, env.bob)] == [bob_contract["id"]]

    def test_other_users_contract_is_not_found(self, env):
        contract = _create(env, env.alice)
        cid = contract["id"]

        assert env.client.get(f"/api/contracts/{cid}", headers=env.bob).status_code == 404
        assert env.client.post(f"/api/contracts/{cid}/revoke", headers=env.bob).status_code == 404
        assert env.client.delete(f"/api/contracts/{cid}", headers=env.bob).status_code == 404

        # Still active for its owner.
        detail = env.client.get(f"/api/contracts/{cid}", headers=env.alice)
        assert detail.status_code == 200
        assert detail.json()["status"] == "ACTIVE"

    def test_unknown_contract_is_not_found(self, env):
        assert env.client.get("/api/contracts/LC-missing", headers=env.alice).status_code == 404
        response = env.client.post("/api/contracts/LC-missing/revoke", headers=env.alice)
        assert response.status_code == 404

    def test_new_user_has_no_default_contracts(self, env):
        assert _list(env, env.alice) == []
        stats = env.client.get("/api/contracts/stats", headers=env.alice).json()
        assert stats["total_contracts"] == 0


class TestRevokeAndStats:
    def test_revoke(self, env):
        contract = _create(env, env.alice)
        cid = contract["id"]

        response = env.client.post(f"/api/contracts/{cid}/revoke", headers=env.alice)
        assert response.status_code == 200
        assert response.json() == {"status": "revoked", "contract_id": cid}

        detail = env.client.get(f"/api/contracts/{cid}", headers=env.alice).json()
        assert detail["status"] == "REVOKED"

        response = env.client.post(f"/api/contracts/{cid}/revoke", headers=env.alice)
        assert response.json()["status"] == "already_revoked"

        assert [c["id"] for c in _list(env, env.alice, status="REVOKED")] == [cid]
        assert _list(env, env.alice, status="ACTIVE") == []

    def test_delete_revokes(self, env):
        contract = _create(env, env.alice)
        response = env.client.delete(f"/api/contracts/{contract['id']}", headers=env.alice)
        assert response.status_code == 200
        assert response.json()["status"] == "revoked"

    def test_status_filter(self, env):
        active = _create(env, env.alice)
        revoked = _create(env, env.alice)
        env.client.post(f"/api/contracts/{revoked['id']}/revoke", headers=env.alice)

        assert [c["id"] for c in _list(env, env.alice, status="ACTIVE")] == [active["id"]]
        assert [c["id"] for c in _list(env, env.alice, status="active")] == [active["id"]]
        assert _list(env, env.alice, status="PENDING") == []

        response = env.client.get("/api/contracts/?status=BOGUS", headers=env.alice)
        assert response.status_code == 400

    def test_stats(self, env):
        _create(env, env.alice, contract_type="EPISODIC")
        _from_template(env, env.alice, "coding")
        revoked = _create(env, env.alice, contract_type="PROHIBITED", domains=["medical"])
        env.client.post(f"/api/contracts/{revoked['id']}/revoke", headers=env.alice)
        _create(env, env.bob, contract_type="STRATEGIC")

        stats = env.client.get("/api/contracts/stats", headers=env.alice).json()
        assert stats["total_contracts"] == 3
        assert stats["active_contracts"] == 2
        assert stats["revoked_contracts"] == 1
        assert stats["pending_contracts"] == 0
        assert stats["expired_contracts"] == 0
        assert stats["contracts_by_type"] == {"EPISODIC": 1, "PROCEDURAL": 1, "PROHIBITED": 1}

        bob_stats = env.client.get("/api/contracts/stats", headers=env.bob).json()
        assert bob_stats["total_contracts"] == 1
        assert bob_stats["contracts_by_type"] == {"STRATEGIC": 1}

    def test_expired_contract(self, env):
        from src.contracts import ContractScope, ContractType, LearningScope

        real_store = env.routes.get_store()._store
        expired = real_store.create_contract(
            user_id=env.alice_id,
            contract_type=ContractType.EPISODIC,
            scope=ContractScope(scope_type=LearningScope.ALL),
            duration=timedelta(seconds=-1),
            auto_activate=True,
        )

        listed = _list(env, env.alice)
        assert [(c["id"], c["status"]) for c in listed] == [(expired.contract_id, "EXPIRED")]
        # The transition is persisted in the real store.
        assert real_store.get_contract(expired.contract_id).status.name == "EXPIRED"

        stats = env.client.get("/api/contracts/stats", headers=env.alice).json()
        assert stats["expired_contracts"] == 1
        assert stats["active_contracts"] == 0

        response = env.client.post(
            f"/api/contracts/{expired.contract_id}/revoke", headers=env.alice
        )
        assert response.status_code == 200
        assert response.json()["status"] == "already_expired"


class TestPersistence:
    def test_store_lives_in_data_dir(self, env):
        _create(env, env.alice)
        assert (env.data_dir / "contracts.db").exists()

    def test_contracts_survive_store_recreation(self, env):
        kept = _from_template(env, env.alice, "coding")
        revoked = _create(env, env.alice)
        env.client.post(f"/api/contracts/{revoked['id']}/revoke", headers=env.alice)
        bobs = _create(env, env.bob)

        # Simulate a server restart: drop the cached store and reopen from disk.
        env.routes.reset_store()

        listed = {c["id"]: c["status"] for c in _list(env, env.alice)}
        assert listed == {kept["id"]: "ACTIVE", revoked["id"]: "REVOKED"}
        assert [c["id"] for c in _list(env, env.bob)] == [bobs["id"]]

        detail = env.client.get(f"/api/contracts/{kept['id']}", headers=env.alice).json()
        assert detail["domains"] == kept["domains"]
        assert detail["metadata"]["template"] == "coding"

    def test_adapter_reopens_same_db_file(self, tmp_path):
        from src.web.routes.contracts import (
            ContractsStore,
            CreateContractRequest,
            CreateFromTemplateRequest,
        )

        db_path = tmp_path / "nested" / "contracts.db"
        store = ContractsStore(db_path=db_path)
        created = store.create_contract(
            "user-1", CreateContractRequest(contract_type="OBSERVATION", domains=["safety"])
        )
        store.create_from_template("user-2", CreateFromTemplateRequest(template_id="study"))
        store.close()

        reopened = ContractsStore(db_path=db_path)
        try:
            contracts = reopened.get_contracts("user-1")
            assert [c.id for c in contracts] == [created.id]
            assert contracts[0].domains == ["safety"]
            assert reopened.get_contract(created.id, "user-2") is None
            assert reopened.get_stats("user-2").total_contracts == 1
            assert reopened.get_contracts("") == []
        finally:
            reopened.close()
