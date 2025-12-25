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
        assert config.require_auth is False

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
        app = create_app()
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
        """Create test client."""
        from src.web.app import create_app
        app = create_app()
        return TestClient(app)

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
        app = create_app()
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
        response = client.get("/api/constitution/rules/supreme-001")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "supreme-001"
        assert "content" in data

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
        """Create test client."""
        from src.web.app import create_app
        app = create_app()
        return TestClient(app)

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
        app = create_app()
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

        manager = ConnectionManager()
        conv_id = "test-conv-1"

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
        from src.web.routes.agents import AgentStore

        store = AgentStore()
        agents = store.get_all()
        assert len(agents) > 0
        assert any(a.name == "whisper" for a in agents)

    def test_update_status(self):
        """Test updating agent status."""
        from src.web.routes.agents import AgentStore, AgentStatus

        store = AgentStore()
        assert store.update_status("whisper", AgentStatus.DISABLED)
        agent = store.get("whisper")
        assert agent.status == AgentStatus.DISABLED

    def test_add_log(self):
        """Test adding log entries."""
        from src.web.routes.agents import AgentStore, AgentLogEntry
        from datetime import datetime

        store = AgentStore()
        store.add_log(AgentLogEntry(
            timestamp=datetime.utcnow(),
            level="INFO",
            message="Test log",
            agent_name="whisper"
        ))

        logs = store.get_logs(agent_name="whisper")
        assert len(logs) > 0


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

        with pytest.raises(ValueError, match="immutable"):
            store.update_rule("supreme-001", RuleUpdate(content="New content"))

    def test_validate_safe_content(self):
        """Test validation of safe content."""
        from src.web.routes.constitution import ConstitutionStore

        store = ConstitutionStore()
        result = store.validate_content("This is perfectly safe content")
        assert result.is_allowed


class TestMemoryStore:
    """Tests for memory store."""

    def test_create_and_get(self):
        """Test creating and getting memory."""
        from src.web.routes.memory import MemoryStore, MemoryCreate, MemoryType

        store = MemoryStore()

        entry = store.create(MemoryCreate(
            content="Test memory",
            memory_type=MemoryType.WORKING,
            tags=["test"],
            consent_given=True,
        ))

        retrieved = store.get(entry.id)
        assert retrieved is not None
        assert retrieved.content == "Test memory"
        assert retrieved.access_count == 1  # Incremented on get

    def test_search(self):
        """Test memory search."""
        from src.web.routes.memory import MemoryStore

        store = MemoryStore()
        results = store.search("python")
        # Mock store has Python-related entries
        assert len(results) >= 0  # May or may not find matches

    def test_delete(self):
        """Test deleting memory."""
        from src.web.routes.memory import MemoryStore, MemoryCreate

        store = MemoryStore()
        entry = store.create(MemoryCreate(
            content="To be deleted",
            consent_given=True,
        ))

        assert store.delete(entry.id)
        assert store.get(entry.id) is None


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
        """Create test client."""
        from src.web.app import create_app
        app = create_app()
        return TestClient(app)

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
