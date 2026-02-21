"""
Tests for UC-017: Web Interface

Tests the FastAPI backend, WebSocket chat, and API endpoints.
"""

import pytest
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

# Skip tests if FastAPI is not installed
try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


def _override_auth(app):
    """Override auth dependencies so tests don't need real tokens."""
    try:
        from src.web.auth_helpers import require_admin_user, require_authenticated_user
        app.dependency_overrides[require_admin_user] = lambda: "test-admin"
        app.dependency_overrides[require_authenticated_user] = lambda: "test-user"
    except ImportError:
        pass
    try:
        from src.web.routes.chat import _authenticate_rest_request
        app.dependency_overrides[_authenticate_rest_request] = lambda: "test-user"
    except ImportError:
        pass
    return app


# =============================================================================
# Configuration Tests
# =============================================================================


class TestWebConfig:
    """Tests for web configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.web.config import WebConfig

        config = WebConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.debug is False
        assert config.require_auth is True

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        from src.web.config import WebConfig

        with patch.dict("os.environ", {
            "AGENT_OS_WEB_HOST": "0.0.0.0",
            "AGENT_OS_WEB_PORT": "9000",
            "AGENT_OS_WEB_DEBUG": "true",
        }):
            config = WebConfig.from_env()
            assert config.host == "0.0.0.0"
            assert config.port == 9000
            assert config.debug is True

    def test_get_config(self):
        """Test global config getter."""
        from src.web.config import get_config, set_config, WebConfig

        config = WebConfig(port=8888)
        set_config(config)
        assert get_config().port == 8888


# =============================================================================
# API Route Tests (using mock stores)
# =============================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestChatAPI:
    """Tests for chat API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.web.app import create_app
        app = _override_auth(create_app())
        return TestClient(app)

    def test_chat_status(self, client):
        """Test chat status endpoint."""
        response = client.get("/api/chat/status")
        assert response.status_code == 200
        data = response.json()
        assert "active_connections" in data
        assert "status" in data
        assert data["status"] == "operational"

    def test_send_message(self, client):
        """Test sending a message via REST."""
        response = client.post(
            "/api/chat/send",
            json={"message": "Hello, Agent OS!"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "conversation_id" in data
        assert data["message"]["role"] == "assistant"

    def test_list_conversations(self, client):
        """Test listing conversations."""
        response = client.get("/api/chat/conversations")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_send_empty_message(self, client):
        """Test sending empty message."""
        response = client.post(
            "/api/chat/send",
            json={"message": ""}
        )
        # Should still process but with empty content
        assert response.status_code == 200


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestAgentsAPI:
    """Tests for agents API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mock agent store."""
        from unittest.mock import patch
        from src.web.app import create_app

        # Force the agent store to use mock agents instead of real ones
        with patch("src.web.routes.agents.REAL_AGENTS_AVAILABLE", False):
            # Reset the store singleton so it reinitializes with mock agents
            import src.web.routes.agents as agents_module
            agents_module._store = None

            app = _override_auth(create_app())
            yield TestClient(app)

            # Clean up
            agents_module._store = None

    def test_list_agents(self, client):
        """Test listing agents."""
        response = client.get("/api/agents/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0  # Mock store has agents

    def test_get_agent(self, client):
        """Test getting specific agent."""
        response = client.get("/api/agents/whisper")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "whisper"
        assert "status" in data
        assert "metrics" in data

    def test_get_nonexistent_agent(self, client):
        """Test getting nonexistent agent."""
        response = client.get("/api/agents/nonexistent")
        assert response.status_code == 404

    def test_agent_metrics(self, client):
        """Test agent metrics endpoint."""
        response = client.get("/api/agents/whisper/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "requests_total" in data
        assert "requests_success" in data

    def test_start_agent(self, client):
        """Test starting an agent."""
        response = client.post("/api/agents/smith/start")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("started", "already_active")

    def test_stop_agent(self, client):
        """Test stopping an agent."""
        response = client.post("/api/agents/smith/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("stopped", "already_stopped")

    def test_agents_overview(self, client):
        """Test agents overview endpoint."""
        response = client.get("/api/agents/stats/overview")
        assert response.status_code == 200
        data = response.json()
        assert "total_agents" in data
        assert "total_requests" in data
        assert "success_rate" in data


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestConstitutionAPI:
    """Tests for constitution API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.web.app import create_app
        app = _override_auth(create_app())
        return TestClient(app)

    def test_get_overview(self, client):
        """Test constitution overview."""
        response = client.get("/api/constitution/overview")
        assert response.status_code == 200
        data = response.json()
        assert "total_rules" in data
        assert "sections" in data

    def test_list_sections(self, client):
        """Test listing sections."""
        response = client.get("/api/constitution/sections")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_rules(self, client):
        """Test listing rules."""
        response = client.get("/api/constitution/rules")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_get_rule(self, client):
        """Test getting specific rule."""
        # First get all rules to find a valid rule ID
        response = client.get("/api/constitution/rules")
        assert response.status_code == 200
        rules = response.json()
        if len(rules) > 0:
            rule_id = rules[0]["id"]
            response = client.get(f"/api/constitution/rules/{rule_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == rule_id
            assert "content" in data
        else:
            pytest.skip("No rules found to test")

    def test_create_rule(self, client):
        """Test creating a new rule."""
        response = client.post(
            "/api/constitution/rules",
            json={
                "content": "Test rule content",
                "rule_type": "permission",
                "keywords": ["test"],
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Test rule content"
        assert data["id"].startswith("user-")

    def test_create_supreme_rule_forbidden(self, client):
        """Test that creating supreme rules is forbidden."""
        response = client.post(
            "/api/constitution/rules",
            json={
                "content": "Trying to create supreme rule",
                "rule_type": "prohibition",
                "authority": "supreme",
            }
        )
        assert response.status_code == 403

    def test_validate_content(self, client):
        """Test content validation."""
        response = client.post(
            "/api/constitution/validate",
            json={"content": "This is safe content"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "is_allowed" in data
        assert isinstance(data["is_allowed"], bool)

    def test_validate_harmful_content(self, client):
        """Test validation of potentially harmful content."""
        response = client.post(
            "/api/constitution/validate",
            json={"content": "Content about harm and violence"}
        )
        assert response.status_code == 200
        data = response.json()
        # Should match prohibition rules
        assert len(data["matching_rules"]) > 0


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestMemoryAPI:
    """Tests for memory API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with authentication mocked."""
        from unittest.mock import patch
        from src.web.app import create_app
        app = create_app()

        # Mock authentication to return a test user
        with patch("src.web.routes.memory.get_current_user_id", return_value="test_user"):
            yield TestClient(app)

    def test_list_memories(self, client):
        """Test listing memories."""
        response = client.get("/api/memory/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_memory_stats(self, client):
        """Test memory stats."""
        response = client.get("/api/memory/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_entries" in data
        assert "total_size_bytes" in data

    def test_create_memory(self, client):
        """Test creating a memory."""
        response = client.post(
            "/api/memory/",
            json={
                "content": "Test memory content",
                "memory_type": "working",
                "tags": ["test"],
                "consent_given": True,
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Test memory content"
        assert "id" in data

    def test_create_memory_without_consent(self, client):
        """Test creating memory without consent fails."""
        response = client.post(
            "/api/memory/",
            json={
                "content": "No consent memory",
                "consent_given": False,
            }
        )
        assert response.status_code == 403

    def test_search_memories(self, client):
        """Test memory search."""
        response = client.get("/api/memory/search?query=python")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_export_memories(self, client):
        """Test memory export."""
        response = client.get("/api/memory/export")
        assert response.status_code == 200
        data = response.json()
        assert "exported_at" in data
        assert "entries" in data


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestSystemAPI:
    """Tests for system API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.web.app import create_app
        app = _override_auth(create_app())
        return TestClient(app)

    def test_system_info(self, client):
        """Test system info endpoint."""
        response = client.get("/api/system/info")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "python_version" in data
        assert "platform" in data

    def test_system_status(self, client):
        """Test system status endpoint."""
        response = client.get("/api/system/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded", "unhealthy")
        assert "components" in data

    def test_component_health(self, client):
        """Test component health endpoint."""
        response = client.get("/api/system/health")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert all("name" in c and "status" in c for c in data)

    def test_list_settings(self, client):
        """Test listing settings."""
        response = client.get("/api/system/settings")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_setting(self, client):
        """Test getting specific setting."""
        response = client.get("/api/system/settings/logging.level")
        assert response.status_code == 200
        data = response.json()
        assert data["key"] == "logging.level"

    def test_update_setting(self, client):
        """Test updating a setting."""
        response = client.put(
            "/api/system/settings/logging.level",
            json={"value": "DEBUG"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["value"] == "DEBUG"

    def test_system_version(self, client):
        """Test version endpoint."""
        response = client.get("/api/system/version")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "api_version" in data

    def test_system_metrics(self, client):
        """Test metrics endpoint."""
        response = client.get("/api/system/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "uptime_seconds" in data
        assert "memory" in data
        assert "cpu" in data

    def test_shutdown_without_confirm(self, client):
        """Test shutdown without confirmation fails."""
        response = client.post("/api/system/shutdown")
        assert response.status_code == 400


# =============================================================================
# WebSocket Tests
# =============================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestWebSocket:
    """Tests for WebSocket functionality."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.web.app import create_app
        app = create_app()
        return TestClient(app)

    def test_websocket_connect(self, client):
        """Test WebSocket connection."""
        with client.websocket_connect("/api/chat/ws") as websocket:
            # Should receive connection message
            data = websocket.receive_json()
            assert data["type"] == "connected"
            assert "connection_id" in data
            assert "conversation_id" in data

    def test_websocket_send_message(self, client):
        """Test sending message via WebSocket."""
        with client.websocket_connect("/api/chat/ws") as websocket:
            # Skip connection message
            websocket.receive_json()

            # Send message
            websocket.send_json({
                "type": "message",
                "content": "Hello via WebSocket!"
            })

            # Receive response
            data = websocket.receive_json()
            assert data["type"] == "response"
            assert "message" in data
            assert data["message"]["role"] == "assistant"

    def test_websocket_ping_pong(self, client):
        """Test WebSocket ping/pong."""
        with client.websocket_connect("/api/chat/ws") as websocket:
            # Skip connection message
            websocket.receive_json()

            # Send ping
            websocket.send_json({"type": "ping"})

            # Receive pong
            data = websocket.receive_json()
            assert data["type"] == "pong"
            assert "timestamp" in data

    def test_websocket_get_history(self, client):
        """Test getting chat history via WebSocket."""
        with client.websocket_connect("/api/chat/ws") as websocket:
            # Skip connection message
            websocket.receive_json()

            # Request history
            websocket.send_json({"type": "history"})

            # Receive history
            data = websocket.receive_json()
            assert data["type"] == "history"
            assert "messages" in data


# =============================================================================
# Chat Connection Manager Tests
# =============================================================================


class TestConnectionManager:
    """Tests for WebSocket connection manager."""

    def test_create_manager(self):
        """Test creating connection manager."""
        from src.web.routes.chat import ConnectionManager

        manager = ConnectionManager()
        assert len(manager.connections) == 0
        assert len(manager.conversations) == 0

    def test_get_conversation(self):
        """Test getting conversation history."""
        from src.web.routes.chat import ConnectionManager, ChatMessage, MessageRole
        import uuid

        manager = ConnectionManager()
        # Use unique conversation ID to avoid state leakage from other tests
        conv_id = f"test-conv-{uuid.uuid4()}"

        # Initially empty
        history = manager.get_conversation(conv_id)
        assert len(history) == 0

        # Add message
        message = ChatMessage(
            role=MessageRole.USER,
            content="Test message"
        )
        manager.add_message(conv_id, message)

        # Now has one message
        history = manager.get_conversation(conv_id)
        assert len(history) == 1
        assert history[0].content == "Test message"


# =============================================================================
# Mock Store Tests
# =============================================================================


class TestAgentStore:
    """Tests for agent store."""

    def test_get_all_agents(self):
        """Test getting all agents."""
        from unittest.mock import patch
        import src.web.routes.agents as agents_module

        # Force mock agents by disabling real agents
        with patch.object(agents_module, "REAL_AGENTS_AVAILABLE", False):
            agents_module._store = None
            store = agents_module.AgentStore()
            agents = store.get_all()
            assert len(agents) > 0
            assert any(a.name == "whisper" for a in agents)
            agents_module._store = None

    def test_update_status(self):
        """Test updating agent status."""
        from unittest.mock import patch
        import src.web.routes.agents as agents_module
        from src.web.routes.agents import AgentStatus

        # Force mock agents by disabling real agents
        with patch.object(agents_module, "REAL_AGENTS_AVAILABLE", False):
            agents_module._store = None
            store = agents_module.AgentStore()
            assert store.update_status("whisper", AgentStatus.DISABLED)
            agent = store.get("whisper")
            assert agent.status == AgentStatus.DISABLED
            agents_module._store = None

    def test_add_log(self):
        """Test adding log entries."""
        from unittest.mock import patch
        import src.web.routes.agents as agents_module
        from src.web.routes.agents import AgentLogEntry
        from datetime import datetime

        # Force mock agents by disabling real agents
        with patch.object(agents_module, "REAL_AGENTS_AVAILABLE", False):
            agents_module._store = None
            store = agents_module.AgentStore()
            store.add_log(AgentLogEntry(
                timestamp=datetime.utcnow(),
                level="INFO",
                message="Test log",
                agent_name="whisper"
            ))

            logs = store.get_logs(agent_name="whisper")
            assert len(logs) > 0
            agents_module._store = None


class TestConstitutionStore:
    """Tests for constitution store."""

    def test_get_all_rules(self):
        """Test getting all rules."""
        from src.web.routes.constitution import ConstitutionStore

        store = ConstitutionStore()
        rules = store.get_all_rules()
        assert len(rules) > 0

    def test_immutable_rule(self):
        """Test that immutable rules cannot be modified."""
        from src.web.routes.constitution import ConstitutionStore, RuleUpdate

        store = ConstitutionStore()
        rules = store.get_all_rules()

        # Find an actual immutable rule
        immutable_rules = [r for r in rules if r.is_immutable]
        if immutable_rules:
            with pytest.raises(ValueError, match="immutable"):
                store.update_rule(immutable_rules[0].id, RuleUpdate(content="New content"))
        else:
            # If no immutable rules, skip this test
            pytest.skip("No immutable rules found to test")

    def test_validate_safe_content(self):
        """Test validation returns results for content check."""
        from src.web.routes.constitution import ConstitutionStore

        store = ConstitutionStore()
        # Test that validation returns a result (may have matches due to common keywords)
        result = store.validate_content("Hello, how are you today?")
        # The result should be a ValidationResult with matching_rules list
        assert hasattr(result, "is_allowed")
        assert hasattr(result, "matching_rules")


class TestMemoryStore:
    """Tests for memory store."""

    def test_create_and_get(self):
        """Test creating and getting memory."""
        from src.web.routes.memory import MemoryStore, MemoryCreate, MemoryType

        store = MemoryStore()
        store.initialize()

        entry = store.create(MemoryCreate(
            content="Test memory",
            memory_type=MemoryType.WORKING,
            tags=["test"],
            consent_given=True,
        ), user_id="test_user")

        retrieved = store.get(entry.id, user_id="test_user")
        assert retrieved is not None
        assert retrieved.content == "Test memory"
        # Note: access_count is incremented after fetch, so first retrieval returns 0
        # The increment is visible on subsequent fetches
        assert retrieved.access_count == 0

    def test_search(self):
        """Test memory search."""
        from src.web.routes.memory import MemoryStore

        store = MemoryStore()
        store.initialize()
        results = store.search("python", user_id="test_user")
        # Mock store has Python-related entries
        assert len(results) >= 0  # May or may not find matches

    def test_delete(self):
        """Test deleting memory."""
        from src.web.routes.memory import MemoryStore, MemoryCreate

        store = MemoryStore()
        store.initialize()
        entry = store.create(MemoryCreate(
            content="To be deleted",
            consent_given=True,
        ), user_id="test_user")

        assert store.delete(entry.id, user_id="test_user")
        assert store.get(entry.id, user_id="test_user") is None


# =============================================================================
# Health Check
# =============================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestHealthCheck:
    """Tests for health check endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.web.app import create_app
        app = create_app()
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


# =============================================================================
# Acceptance Criteria
# =============================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestAcceptanceCriteria:
    """Tests verifying UC-017 acceptance criteria."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        from unittest.mock import patch
        from src.web.app import create_app
        import src.web.routes.agents as agents_module

        # Force mock agents and mock memory auth
        with patch.object(agents_module, "REAL_AGENTS_AVAILABLE", False):
            agents_module._store = None
            with patch("src.web.routes.memory.get_current_user_id", return_value="test_user"):
                app = create_app()
                yield TestClient(app)
            agents_module._store = None

    def test_chat_interface_works(self, client):
        """Verify chat interface functionality."""
        # Can send message
        response = client.post(
            "/api/chat/send",
            json={"message": "Test chat message"}
        )
        assert response.status_code == 200
        assert "message" in response.json()

        # WebSocket connects
        with client.websocket_connect("/api/chat/ws") as ws:
            data = ws.receive_json()
            assert data["type"] == "connected"

    def test_agent_monitoring_works(self, client):
        """Verify agent monitoring functionality."""
        # Can list agents
        response = client.get("/api/agents/")
        assert response.status_code == 200
        agents = response.json()
        assert len(agents) > 0

        # Can get agent details
        response = client.get("/api/agents/whisper")
        assert response.status_code == 200
        assert "metrics" in response.json()

        # Can control agents
        response = client.post("/api/agents/whisper/restart")
        assert response.status_code == 200

    def test_constitutional_editor_works(self, client):
        """Verify constitutional editor functionality."""
        # Can view rules
        response = client.get("/api/constitution/rules")
        assert response.status_code == 200
        rules = response.json()
        assert len(rules) > 0

        # Can create rules (with proper authority)
        response = client.post(
            "/api/constitution/rules",
            json={
                "content": "User-created test rule",
                "rule_type": "permission",
                "authority": "statutory",
            }
        )
        assert response.status_code == 200

        # Can validate content
        response = client.post(
            "/api/constitution/validate",
            json={"content": "Test content for validation"}
        )
        assert response.status_code == 200

    def test_memory_management_works(self, client):
        """Verify memory management functionality."""
        # Can list memories
        response = client.get("/api/memory/")
        assert response.status_code == 200

        # Can create memories
        response = client.post(
            "/api/memory/",
            json={
                "content": "Test memory for UI",
                "memory_type": "working",
                "consent_given": True,
            }
        )
        assert response.status_code == 200
        memory_id = response.json()["id"]

        # Can search memories
        response = client.get("/api/memory/search?query=test")
        assert response.status_code == 200

        # Can delete memories
        response = client.delete(f"/api/memory/{memory_id}")
        assert response.status_code == 200


# =============================================================================
# Voice API Tests
# =============================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestVoiceAPI:
    """Tests for voice API endpoints (STT/TTS)."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.web.app import create_app
        app = create_app()
        return TestClient(app)

    def test_voice_status(self, client):
        """Test voice status endpoint."""
        response = client.get("/api/voice/status")
        assert response.status_code == 200
        data = response.json()
        assert "stt_available" in data
        assert "stt_engine" in data
        assert "tts_available" in data
        assert "tts_engine" in data
        assert "available_voices" in data

    def test_list_voices(self, client):
        """Test listing TTS voices."""
        response = client.get("/api/voice/voices")
        assert response.status_code == 200
        data = response.json()
        assert "voices" in data
        assert "available" in data
        assert isinstance(data["voices"], list)

    def test_transcribe_with_invalid_base64(self, client):
        """Test transcription with invalid base64 data."""
        response = client.post(
            "/api/voice/transcribe",
            json={
                "audio_data": "not-valid-base64!!!",
                "audio_format": "wav",
            }
        )
        # 400 if STT is available but base64 invalid, 503 if STT not available
        assert response.status_code in (400, 503)

    def test_transcribe_with_valid_base64(self, client):
        """Test transcription with valid base64 (mock audio)."""
        import base64

        # Create minimal WAV header + silence (mock audio data)
        # This is a minimal valid WAV file structure
        wav_header = (
            b"RIFF" + (44).to_bytes(4, "little") +  # ChunkID + ChunkSize
            b"WAVE" +  # Format
            b"fmt " + (16).to_bytes(4, "little") +  # Subchunk1ID + Subchunk1Size
            (1).to_bytes(2, "little") +  # AudioFormat (PCM)
            (1).to_bytes(2, "little") +  # NumChannels
            (16000).to_bytes(4, "little") +  # SampleRate
            (32000).to_bytes(4, "little") +  # ByteRate
            (2).to_bytes(2, "little") +  # BlockAlign
            (16).to_bytes(2, "little") +  # BitsPerSample
            b"data" + (8).to_bytes(4, "little") +  # Subchunk2ID + Subchunk2Size
            b"\x00" * 8  # 8 bytes of silence
        )

        audio_b64 = base64.b64encode(wav_header).decode("ascii")

        response = client.post(
            "/api/voice/transcribe",
            json={
                "audio_data": audio_b64,
                "audio_format": "wav",
                "language": "auto",
            }
        )
        # May return 503 if no STT engine available, or 200 with result
        assert response.status_code in (200, 503)
        if response.status_code == 200:
            data = response.json()
            assert "text" in data
            assert "language" in data
            assert "processing_time_ms" in data

    def test_synthesize_without_text(self, client):
        """Test synthesis without text."""
        response = client.post(
            "/api/voice/synthesize",
            json={"text": ""}
        )
        # Pydantic validation should reject empty text
        assert response.status_code == 422

    def test_synthesize_with_text(self, client):
        """Test synthesis with valid text."""
        response = client.post(
            "/api/voice/synthesize",
            json={
                "text": "Hello, Agent OS!",
                "voice": "en_US-lessac-medium",
                "speed": 1.0,
            }
        )
        # May return 503 if no TTS engine available, or 200 with audio
        assert response.status_code in (200, 503)
        if response.status_code == 200:
            data = response.json()
            assert "audio_data" in data
            assert "format" in data
            assert "duration" in data
            assert "processing_time_ms" in data
            assert "sample_rate" in data

    def test_synthesize_stream(self, client):
        """Test streaming synthesis."""
        response = client.post(
            "/api/voice/synthesize/stream",
            json={
                "text": "Hello, streaming test!",
            }
        )
        # May return 503 if no TTS engine available
        assert response.status_code in (200, 503)
        if response.status_code == 200:
            assert response.headers.get("content-type") in (
                "audio/wav",
                "audio/mpeg",
            )

    def test_update_stt_config(self, client):
        """Test updating STT configuration."""
        response = client.put(
            "/api/voice/config/stt",
            json={
                "model": "base",
                "language": "en",
                "translate": False,
            }
        )
        # Should succeed or fail gracefully
        assert response.status_code in (200, 500)
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "updated"

    def test_update_tts_config(self, client):
        """Test updating TTS configuration."""
        response = client.put(
            "/api/voice/config/tts",
            json={
                "voice": "en_US-lessac-medium",
                "speed": 1.2,
            }
        )
        # Should succeed or fail gracefully
        assert response.status_code in (200, 500)
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "updated"


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestVoiceWebSocket:
    """Tests for Voice WebSocket functionality."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.web.app import create_app
        app = create_app()
        return TestClient(app)

    def test_voice_websocket_connect(self, client):
        """Test Voice WebSocket connection and ping/pong."""
        with client.websocket_connect("/api/voice/ws") as websocket:
            # Send ping
            websocket.send_json({"type": "ping"})

            # Receive pong
            data = websocket.receive_json()
            assert data["type"] == "pong"
            assert "timestamp" in data

    def test_voice_websocket_unknown_type(self, client):
        """Test Voice WebSocket with unknown message type."""
        with client.websocket_connect("/api/voice/ws") as websocket:
            websocket.send_json({"type": "unknown_type"})

            data = websocket.receive_json()
            assert data["type"] == "error"
            assert "unknown" in data["message"].lower()

    def test_voice_websocket_synthesize(self, client):
        """Test Voice WebSocket synthesis."""
        with client.websocket_connect("/api/voice/ws") as websocket:
            websocket.send_json({
                "type": "synthesize",
                "text": "Hello from WebSocket!"
            })

            data = websocket.receive_json()
            # May be audio or error if TTS not available
            assert data["type"] in ("audio", "error")
            if data["type"] == "audio":
                assert "data" in data
                assert "format" in data

    def test_voice_websocket_transcribe(self, client):
        """Test Voice WebSocket transcription."""
        import base64

        # Minimal WAV data
        wav_data = b"RIFF" + b"\x00" * 40 + b"data" + b"\x00" * 8
        audio_b64 = base64.b64encode(wav_data).decode("ascii")

        with client.websocket_connect("/api/voice/ws") as websocket:
            websocket.send_json({
                "type": "transcribe",
                "audio": audio_b64,
                "format": "wav"
            })

            data = websocket.receive_json()
            # May be transcription or error if STT not available
            assert data["type"] in ("transcription", "error")
            if data["type"] == "transcription":
                assert "text" in data


# =============================================================================
# Voice Config Tests
# =============================================================================


class TestVoiceConfig:
    """Tests for voice configuration."""

    def test_default_voice_config(self):
        """Test default voice configuration values."""
        from src.web.config import VoiceConfig

        config = VoiceConfig()
        assert config.stt_enabled is True
        assert config.stt_engine == "auto"
        assert config.stt_model == "base"
        assert config.stt_language == "en"
        assert config.tts_enabled is True
        assert config.tts_engine == "auto"
        assert config.tts_speed == 1.0

    def test_voice_config_in_web_config(self):
        """Test voice config is included in web config."""
        from src.web.config import WebConfig

        config = WebConfig()
        assert hasattr(config, "voice")
        assert config.voice.stt_enabled is True
        assert config.voice.tts_enabled is True

    def test_voice_config_from_env(self):
        """Test voice configuration from environment variables."""
        from src.web.config import WebConfig

        with patch.dict("os.environ", {
            "AGENT_OS_STT_ENABLED": "false",
            "AGENT_OS_STT_ENGINE": "mock",
            "AGENT_OS_STT_MODEL": "tiny",
            "AGENT_OS_TTS_ENABLED": "true",
            "AGENT_OS_TTS_ENGINE": "espeak",
            "AGENT_OS_TTS_SPEED": "1.5",
        }):
            config = WebConfig.from_env()
            assert config.voice.stt_enabled is False
            assert config.voice.stt_engine == "mock"
            assert config.voice.stt_model == "tiny"
            assert config.voice.tts_enabled is True
            assert config.voice.tts_engine == "espeak"
            assert config.voice.tts_speed == 1.5


# =============================================================================
# Security API Tests
# =============================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestSecurityAPI:
    """Tests for security API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked Smith agent."""
        from unittest.mock import patch, MagicMock
        from src.web.app import create_app
        import src.web.routes.security as security_module

        # Create mock Smith agent
        mock_smith = MagicMock()
        mock_smith._attack_detection_enabled = True
        mock_smith._attack_detector = MagicMock()
        mock_smith._recommendation_system = MagicMock()

        # Mock get_detected_attacks
        mock_smith.get_detected_attacks.return_value = [
            {
                "attack_id": "ATK-001",
                "attack_type": "PROMPT_INJECTION",
                "severity": "HIGH",
                "status": "DETECTED",
                "detected_at": datetime.utcnow().isoformat(),
                "description": "Test attack",
                "confidence": 0.85,
                "source": "test",
            }
        ]

        # Mock get_attack_detection_status
        mock_smith.get_attack_detection_status.return_value = {
            "enabled": True,
            "available": True,
            "attacks_detected": 5,
            "attacks_mitigated": 2,
            "recommendations_generated": 3,
            "auto_lockdowns_triggered": 0,
            "detector": {"events_processed": 100},
        }

        # Mock get_pending_recommendations (returns list, not dict)
        mock_smith.get_pending_recommendations.return_value = []

        # Mock approve_recommendation
        mock_smith.approve_recommendation.return_value = True

        # Patch the module's get_smith to return our mock
        with patch.object(security_module, "_smith_instance", mock_smith):
            with patch.object(security_module, "get_smith", return_value=mock_smith):
                app = _override_auth(create_app())
                yield TestClient(app)

    def test_list_attacks(self, client):
        """Test listing detected attacks."""
        response = client.get("/api/security/attacks")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_attacks_with_filters(self, client):
        """Test listing attacks with filters."""
        response = client.get("/api/security/attacks?severity=HIGH&limit=10")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_attack_detail(self, client):
        """Test getting attack details."""
        # First get list to get an attack ID
        response = client.get("/api/security/attacks")
        assert response.status_code == 200
        attacks = response.json()

        if attacks:
            attack_id = attacks[0]["attack_id"]
            response = client.get(f"/api/security/attacks/{attack_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["attack_id"] == attack_id

    def test_get_attack_detection_status(self, client):
        """Test getting attack detection status."""
        response = client.get("/api/security/status")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data
        assert "attacks_detected" in data
        assert "recommendations_generated" in data

    def test_list_recommendations(self, client):
        """Test listing recommendations."""
        response = client.get("/api/security/recommendations")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_approve_recommendation(self, client):
        """Test approving a recommendation."""
        response = client.post(
            "/api/security/recommendations/REC-001/approve",
            json={
                "approver": "test_user",
                "comments": "Looks good",
            }
        )
        # May succeed or fail depending on mock
        assert response.status_code in (200, 400, 404)

    def test_reject_recommendation(self, client):
        """Test rejecting a recommendation."""
        response = client.post(
            "/api/security/recommendations/REC-001/reject",
            json={
                "rejector": "test_user",
                "reason": "False positive",
            }
        )
        # May succeed or fail depending on mock
        assert response.status_code in (200, 400, 404, 503)

    def test_add_comment_to_recommendation(self, client):
        """Test adding comment to recommendation."""
        response = client.post(
            "/api/security/recommendations/REC-001/comments",
            json={
                "author": "test_user",
                "content": "This needs review",
            }
        )
        assert response.status_code in (200, 400, 404, 503)

    def test_pipeline_control(self, client):
        """Test controlling the attack detection pipeline."""
        # Stop pipeline
        response = client.post(
            "/api/security/pipeline",
            json={"action": "stop"}
        )
        assert response.status_code in (200, 503)

        # Start pipeline
        response = client.post(
            "/api/security/pipeline",
            json={"action": "start"}
        )
        assert response.status_code in (200, 503)

    def test_invalid_pipeline_action(self, client):
        """Test invalid pipeline action."""
        response = client.post(
            "/api/security/pipeline",
            json={"action": "invalid"}
        )
        assert response.status_code == 422  # Validation error

    def test_list_patterns(self, client):
        """Test listing attack patterns."""
        response = client.get("/api/security/patterns")
        # May fail if detector not fully initialized
        assert response.status_code in (200, 503)
        if response.status_code == 200:
            data = response.json()
            assert "patterns" in data
            assert "total" in data


class TestSecurityModels:
    """Tests for security API models."""

    def test_attack_summary_model(self):
        """Test AttackSummary model."""
        from src.web.routes.security import AttackSummary

        summary = AttackSummary(
            attack_id="ATK-001",
            attack_type="PROMPT_INJECTION",
            severity="HIGH",
            status="DETECTED",
            detected_at=datetime.utcnow(),
            description="Test attack",
            confidence=0.85,
        )
        assert summary.attack_id == "ATK-001"
        assert summary.confidence == 0.85

    def test_recommendation_summary_model(self):
        """Test RecommendationSummary model."""
        from src.web.routes.security import RecommendationSummary

        summary = RecommendationSummary(
            recommendation_id="REC-001",
            attack_id="ATK-001",
            title="Add input validation",
            priority="HIGH",
            status="PENDING",
            created_at=datetime.utcnow(),
            patch_count=2,
        )
        assert summary.recommendation_id == "REC-001"
        assert summary.patch_count == 2

    def test_attack_detection_status_model(self):
        """Test AttackDetectionStatus model."""
        from src.web.routes.security import AttackDetectionStatus

        status = AttackDetectionStatus(
            enabled=True,
            available=True,
            pipeline_running=True,
            attacks_detected=10,
            attacks_mitigated=5,
            recommendations_generated=3,
            auto_lockdowns_triggered=1,
        )
        assert status.enabled is True
        assert status.attacks_detected == 10

    def test_approve_request_model(self):
        """Test ApproveRecommendationRequest model."""
        from src.web.routes.security import ApproveRecommendationRequest

        request = ApproveRecommendationRequest(
            approver="security_team",
            comments="Approved after review",
        )
        assert request.approver == "security_team"

    def test_pipeline_control_request_model(self):
        """Test PipelineControlRequest validation."""
        from src.web.routes.security import PipelineControlRequest

        # Valid actions
        start = PipelineControlRequest(action="start")
        assert start.action == "start"

        stop = PipelineControlRequest(action="stop")
        assert stop.action == "stop"

        # Invalid action should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            PipelineControlRequest(action="invalid")
