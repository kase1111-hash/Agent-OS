# API Reference

Agent-OS provides a comprehensive REST API for interacting with the system. All endpoints are available at `http://localhost:8080` by default.

## Interactive Documentation

The API includes interactive Swagger documentation:

- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc
- **OpenAPI JSON**: http://localhost:8080/openapi.json

## Base Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/health` | GET | Health check |

## Authentication API (`/api/auth`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/register` | POST | Register a new user |
| `/login` | POST | Authenticate and get session token |
| `/logout` | POST | End current session |
| `/me` | GET | Get current user info |
| `/status` | GET | Get authentication status |
| `/profile` | PUT | Update user profile |
| `/change-password` | POST | Change password |
| `/sessions` | GET | List active sessions |
| `/sessions/{id}` | DELETE | Revoke a specific session |
| `/logout-all` | POST | Revoke all sessions |
| `/users/count` | GET | Get total user count |

## Chat API (`/api/chat`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/send` | POST | Send a message to the system |
| `/conversations` | GET | List all conversations |
| `/conversations/{id}` | GET | Get conversation messages |
| `/conversations/{id}` | DELETE | Delete a conversation |
| `/conversations/{id}/export` | GET | Export conversation |
| `/conversations/{id}/archive` | POST | Archive conversation |
| `/conversations/{id}/unarchive` | POST | Unarchive conversation |
| `/status` | GET | Get chat system status |
| `/models` | GET | List available LLM models |
| `/models/current` | GET | Get current model |
| `/models/switch` | POST | Switch active model |
| `/export` | GET | Export all conversations |
| `/import` | POST | Import conversations |
| `/storage/stats` | GET | Get storage statistics |
| `/search` | GET | Search conversations |

### WebSocket

- `/ws/chat` - Real-time chat communication

## Agents API (`/api/agents`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | List all agents |
| `/{name}` | GET | Get agent details |
| `/{name}/metrics` | GET | Get agent metrics |
| `/{name}/start` | POST | Start an agent |
| `/{name}/stop` | POST | Stop an agent |
| `/{name}/restart` | POST | Restart an agent |
| `/{name}/logs` | GET | Get agent logs |
| `/logs/all` | GET | Get all agent logs |
| `/stats/overview` | GET | Get agents overview |

## Memory API (`/api/memory`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | List memory entries |
| `/` | POST | Create memory entry |
| `/` | DELETE | Clear all memories |
| `/{id}` | GET | Get specific memory |
| `/{id}` | PUT | Update memory entry |
| `/{id}` | DELETE | Delete memory entry |
| `/stats` | GET | Get memory statistics |
| `/search` | GET | Search memories |
| `/export` | GET | Export all memories |

## Constitution API (`/api/constitution`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/overview` | GET | Get constitution overview |
| `/sections` | GET | List constitution sections |
| `/rules` | GET | List all rules |
| `/rules` | POST | Create a new rule |
| `/rules/{id}` | GET | Get specific rule |
| `/rules/{id}` | PUT | Update a rule |
| `/rules/{id}` | DELETE | Delete a rule |
| `/validate` | POST | Validate action against constitution |
| `/search` | GET | Search rules |

## Contracts API (`/api/contracts`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | List all contracts |
| `/` | POST | Create a contract |
| `/{id}` | GET | Get contract details |
| `/{id}` | DELETE | Delete a contract |
| `/{id}/revoke` | POST | Revoke a contract |
| `/stats` | GET | Get contracts statistics |
| `/types` | GET | List contract types |
| `/templates` | GET | List contract templates |
| `/templates/{id}` | GET | Get template details |
| `/from-template` | POST | Create contract from template |

## Security API (`/api/security`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/attacks` | GET | List detected attacks |
| `/attacks/{id}` | GET | Get attack details |
| `/attacks/{id}/false-positive` | POST | Mark as false positive |
| `/recommendations` | GET | List fix recommendations |
| `/recommendations/{id}` | GET | Get recommendation details |
| `/recommendations/{id}/markdown` | GET | Get markdown format |
| `/recommendations/{id}/approve` | POST | Approve recommendation |
| `/recommendations/{id}/reject` | POST | Reject recommendation |
| `/recommendations/{id}/comments` | POST | Add review comment |
| `/recommendations/{id}/assign` | POST | Assign reviewers |
| `/status` | GET | Get detection system status |
| `/pipeline` | POST | Start/stop detection pipeline |
| `/patterns` | GET | List detection patterns |
| `/patterns/{id}/enable` | POST | Enable a pattern |
| `/patterns/{id}/disable` | POST | Disable a pattern |

## System API (`/api/system`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/info` | GET | Get system information |
| `/status` | GET | Get system status |
| `/health` | GET | Get component health |
| `/settings` | GET | List all settings |
| `/settings/{key}` | GET | Get specific setting |
| `/settings/{key}` | PUT | Update setting |
| `/logs` | GET | Get system logs |
| `/shutdown` | POST | Shutdown the system |
| `/restart` | POST | Restart the system |
| `/version` | GET | Get version info |
| `/dreaming` | GET | Get dreaming status |
| `/metrics` | GET | Get system metrics |

## Voice API (`/api/voice`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Get voice system status |
| `/transcribe` | POST | Transcribe audio (base64) |
| `/transcribe/file` | POST | Transcribe audio file |
| `/synthesize` | POST | Text-to-speech synthesis |
| `/synthesize/stream` | POST | Streaming TTS |
| `/voices` | GET | List available voices |
| `/config/stt` | PUT | Configure speech-to-text |
| `/config/tts` | PUT | Configure text-to-speech |

## Images API (`/api/images`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/models` | GET | List image models |
| `/models/{id}` | GET | Get model details |
| `/generate` | POST | Generate an image |
| `/generate/{job_id}` | GET | Get generation status |
| `/jobs` | GET | List generation jobs |
| `/gallery` | GET | List gallery images |
| `/gallery/{id}` | DELETE | Delete gallery image |
| `/image/{id}` | GET | Get image file |
| `/stats` | GET | Get image statistics |

## Observability API (`/api/observability`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Detailed health check |
| `/health/{check}` | GET | Specific health check |
| `/health/checks/list` | GET | List available checks |
| `/metrics` | GET | Prometheus metrics |
| `/metrics/json` | GET | JSON metrics |
| `/traces` | GET | List traces |
| `/traces` | DELETE | Clear traces |
| `/status` | GET | Observability status |

## Intent Log API (`/api/intent-log`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | List intent log entries |
| `/` | DELETE | Clear intent log |
| `/{id}` | GET | Get specific entry |
| `/stats` | GET | Get intent log statistics |
| `/types` | GET | List intent types |
| `/recent` | GET | Get recent entries |
| `/session/{id}` | GET | Get session entries |
| `/export` | POST | Export intent log |

## Common Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden (constitutional violation) |
| 404 | Not Found |
| 422 | Validation Error |
| 500 | Internal Server Error |

## Authentication

Most endpoints require authentication. Include the session token in requests:

```http
Authorization: Bearer <session_token>
```

Or use session cookies set during login.

## Rate Limiting

Default rate limits (configurable):

- **Chat**: 60 requests/minute
- **Memory**: 100 requests/minute
- **Other**: 200 requests/minute

## Example Usage

### Send a Chat Message

```bash
curl -X POST http://localhost:8080/api/chat/send \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"message": "Hello, what can you help me with?"}'
```

### List Agents

```bash
curl http://localhost:8080/api/agents \
  -H "Authorization: Bearer <token>"
```

### Check System Health

```bash
curl http://localhost:8080/health
```

## WebSocket Protocol

Connect to `/ws/chat` for real-time communication:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/chat');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'message',
    content: 'Hello!'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### WebSocket Message Types

| Type | Direction | Description |
|------|-----------|-------------|
| `message` | Client -> Server | Send a chat message |
| `response` | Server -> Client | Agent response |
| `status` | Server -> Client | Status updates |
| `error` | Server -> Client | Error notifications |
| `heartbeat` | Bidirectional | Keep-alive ping |

---

For more details, see the interactive API documentation at `/docs` when running Agent-OS.
