# Changelog

All notable changes to Agent OS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha] - 2026-01-01

### Added

#### Core Components
- **Whisper Orchestrator**: Intent classification and request routing
- **Agent Smith**: Constitutional validation and security enforcement
- **Seshat Archivist**: Memory and RAG retrieval system with consent-based storage
- **Quill Refiner**: Document formatting and refinement
- **Muse Creative**: Creative content generation with image support
- **Sage Reasoner**: Long-context reasoning and analysis
- **Constitutional Parser**: YAML/Markdown constitution parsing with hot-reload

#### Security Features
- Constitutional validation for all requests
- Comprehensive audit logging with hash-chained integrity
- Redis-backed rate limiting with multiple strategies
- AES-256-GCM encryption at rest
- Attack detection system with pattern matching and SIEM integration
- Post-quantum cryptography support (experimental, requires liboqs)
- Boundary daemon integration for trust policy enforcement

#### Web Interface
- Full-featured web UI with chat, agents, memory, and constitution views
- WebSocket support for real-time chat streaming
- Image generation gallery with multiple model support
- Voice interface (STT via Whisper, TTS via Piper)
- System status dashboard with "dreaming" activity indicator
- Dark/light theme support

#### Integrations
- Ollama local LLM backend
- Prometheus metrics export
- Grafana dashboard templates
- Boundary SIEM integration for enterprise security monitoring
- Boundary Daemon integration for external policy enforcement

#### Developer Features
- Comprehensive test suite (1498+ passing tests)
- SDK for building custom agents
- CLI tools for development and debugging
- Docker and docker-compose deployment
- Windows build support (experimental)

### Security
- Fixed all Bandit high-severity security findings
- MD5 usage marked with `usedforsecurity=False`
- Safe XML parsing with defusedxml
- Safe tarfile extraction with path traversal protection
- SQL injection protection with parameterized queries

### Documentation
- Complete API documentation via OpenAPI/Swagger
- Integration guides for Boundary SIEM and Boundary Daemon
- Architecture documentation
- Security policy and responsible disclosure guidelines
- Alpha release notes with known limitations

### Known Issues
- Multi-user features not yet implemented
- Mobile apps planned for Phase 2
- Post-quantum certificate upgrade not implemented
- Some tests require optional dependencies (cryptography, numpy)

### Breaking Changes Expected
This is an alpha release. The following may change before 1.0:
- Configuration format
- API endpoints and response formats
- Database schemas
- Agent prompt templates
- Plugin API

---

## [Unreleased]

### Planned for 0.2.0
- Improved multi-user support
- Enhanced tool sandboxing
- Federation protocol implementation
- Performance optimizations
- Additional agent templates

---

*For detailed release notes, see [ALPHA_RELEASE_NOTES.md](ALPHA_RELEASE_NOTES.md)*
