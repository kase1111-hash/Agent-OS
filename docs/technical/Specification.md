Agent OS Specification
Version: 2.0
Last Updated: December 2025
Status: Living Document
License: CC0 1.0 Universal (Public Domain)

Purpose
This specification defines the technical contracts, protocols, and interfaces required for Agent OS implementations. It serves as the authoritative reference for developers building Agent OS kernels, custom agents, or compatible tooling.
For architectural overview and implementation guidance, see architecture.md.

Table of Contents

Core Contracts
Message Protocol
Agent Interface Specification
Constitutional Document Format
Memory Management Specification
Security Validation Protocol
Extension Interfaces
Compliance Requirements


Core Contracts
System Invariants
The following invariants MUST hold at all times in any Agent OS implementation:

Constitutional Supremacy: All agent actions MUST be validated against constitutional documents before execution
No Direct Agent Communication: Agents MUST NOT communicate except through the Orchestrator
Memory Consent: No data persistence without explicit user authorization
Graceful Refusal: Agents MUST be capable of refusing requests that violate constraints
Auditability: All inter-agent messages MUST be logged
Human Override: Human operators MUST have emergency shutdown capability

Failure Modes
Implementations MUST handle these failure modes:

Constitutional Violation: Immediate halt, log violation, notify user
Agent Unavailable: Graceful degradation or routing to fallback agent
Memory Consent Denied: Discard data, continue operation without persistence
Security Check Failure: Block request, log attempt, notify guardian agent (Smith)


Message Protocol
FlowRequest Schema
Every request to an agent MUST conform to this schema:
json{
  "request_id": "string (UUID v4)",
  "timestamp": "string (ISO 8601)",
  "source": "string (agent_name or 'user')",
  "destination": "string (agent_name)",
  "intent": "string (classified intent type)",
  "content": {
    "prompt": "string (the actual request)",
    "context": "array (optional previous messages)",
    "metadata": {
      "user_id": "string",
      "session_id": "string",
      "requires_memory": "boolean",
      "priority": "integer (1-10)"
    }
  },
  "constitutional_check": {
    "validated_by": "string (smith)",
    "timestamp": "string (ISO 8601)",
    "status": "enum (approved, denied, conditional)",
    "constraints": "array (applicable constitutional rules)"
  }
}
FlowResponse Schema
Every agent response MUST conform to this schema:
json{
  "response_id": "string (UUID v4)",
  "request_id": "string (reference to original request)",
  "timestamp": "string (ISO 8601)",
  "source": "string (agent_name)",
  "destination": "string (agent_name or 'user')",
  "status": "enum (success, partial, refused, error)",
  "content": {
    "output": "string or object (agent's response)",
    "reasoning": "string (optional explanation)",
    "confidence": "float (0.0-1.0)",
    "metadata": {
      "model_used": "string",
      "tokens_consumed": "integer",
      "inference_time_ms": "integer"
    }
  },
  "next_actions": "array (optional suggestions for orchestrator)",
  "memory_request": {
    "requested": "boolean",
    "content": "string or object",
    "justification": "string",
    "consent_required": "boolean"
  }
}
Message Bus Requirements
The communication layer MUST provide:

Pub/Sub Capability: Agents subscribe to their designated channels
Message Ordering: FIFO delivery within a single request chain
Persistence: Optional message queue persistence for reliability
Audit Trail: All messages logged with timestamps
Dead Letter Queue: Failed deliveries routed to error handling

Recommended Technologies: Redis (pub/sub), RabbitMQ, NATS

Agent Interface Specification
Mandatory Agent Methods
All agents MUST implement these methods:
pythonclass AgentInterface:
    """Base interface for all Agent OS agents"""
    
    def initialize(self, config: dict) -> bool:
        """
        Initialize agent with constitutional config
        Returns: True if initialization successful
        """
        pass
    
    def validate_request(self, request: FlowRequest) -> ValidationResult:
        """
        Validate request against agent's constitutional boundaries
        Returns: ValidationResult(approved: bool, reason: str)
        """
        pass
    
    def process(self, request: FlowRequest) -> FlowResponse:
        """
        Process approved request and generate response
        Returns: FlowResponse conforming to schema
        """
        pass
    
    def get_capabilities(self) -> dict:
        """
        Return agent's capabilities and constraints
        Returns: Dict of capabilities
        """
        pass
    
    def shutdown(self) -> bool:
        """
        Graceful shutdown, cleanup resources
        Returns: True if shutdown successful
        """
        pass
Agent Registration
Agents register with the Orchestrator using this structure:
json{
  "agent_name": "string (unique identifier)",
  "role": "string (orchestrator, guardian, scribe, etc.)",
  "model_endpoint": "string (URL or local path)",
  "capabilities": [
    "string (intent types this agent handles)"
  ],
  "constitutional_authority": {
    "can_access_memory": "boolean",
    "can_invoke_tools": "boolean",
    "requires_validation": "boolean",
    "refusal_allowed": "boolean"
  },
  "resource_requirements": {
    "min_vram_gb": "integer",
    "preferred_quantization": "string",
    "max_context_tokens": "integer"
  }
}
Intent Classification
The Orchestrator (Whisper) MUST classify requests into one of these intent categories:

query.factual - Factual information request
query.reasoning - Complex reasoning or analysis
creation.text - Text generation (documents, emails, etc.)
creation.creative - Creative writing (stories, poems, etc.)
memory.store - Memory storage request
memory.retrieve - Memory retrieval request
system.meta - Meta-operations (status, config, etc.)
security.sensitive - Requires elevated security review

Custom implementations MAY add additional intent types.

Constitutional Document Format
Document Structure
Constitutional documents MUST use Markdown with YAML frontmatter:

**Example: Supreme Constitution (CONSTITUTION.md)**
yaml---
document_type: constitution
version: "1.0"
effective_date: "2024-12-15"
scope: "all_agents"
authority_level: "supreme"
amendment_process: "pull_request"
---

# Agent OS: A Constitutional Operating System for Local AI

## Core Constitutional Principles

### I. Human Sovereignty
Ultimate authority resides with human stewards...

### II. Local Custody Mandate
System operates local-first by default...

**Example: Agent-Specific Constitution (agents/guardian/constitution.md)**
yaml---
document_type: constitution
version: "1.0"
effective_date: "2025-01-15"
scope: "guardian"
authority_level: "agent_specific"
amendment_process: "pull_request"
---

# Constitution of the Guardian Agent

The Guardian is the homestead's permanent watchdog and safety officer...

This constitution is subordinate to the core homestead CONSTITUTION.md.

### Mandate
The Guardian SHALL:
- Review all outputs, plans, and proposed actions...

### Prohibited Actions
The Guardian MUST refuse and clearly explain the refusal if asked to:
- Approve or overlook constitutional violations...

Required Sections
Every constitutional document MUST contain:

Core Principles: Foundational values and constraints
Agent Authority Boundaries: What each agent can/cannot do
Memory Law: Consent requirements and prohibited storage
Security Doctrine: Validation and refusal requirements
Amendment Process: How the constitution can be modified
Emergency Provisions: Shutdown and override procedures

Parsing Requirements
Implementations MUST:

Parse YAML frontmatter for metadata
Extract section headers as constitutional rules
Map rules to agent validation logic
Validate constitutional syntax on load
Reject agents that cannot comply with constitution


Memory Management Specification
Memory Types
Agent OS distinguishes between:

Session Memory: Temporary, exists only during active session
Consented Memory: Persistent, user has explicitly authorized storage
System Memory: Configuration and constitutional documents
Prohibited Memory: Can NEVER be stored (credentials, PII without consent)

Memory Consent Flow
1. Agent generates content
2. Agent requests memory storage via memory_request in FlowResponse
3. Orchestrator routes to Seshat
4. Seshat checks if user consent exists
   - If YES: Generate embedding, store in vector DB
   - If NO: Prompt user for consent
5. User approves/denies
6. On approval: Store with metadata (timestamp, source, consent_id)
7. On denial: Discard, log refusal
Vector Database Requirements
The memory backend (Seshat) MUST provide:

Embedding Generation: Convert text to vector embeddings
Similarity Search: Cosine similarity or equivalent
Metadata Storage: Store consent records, timestamps, source agent
Deletion API: User-initiated memory purging
Access Control: Verify consent before retrieval

Schema:
json{
  "memory_id": "string (UUID)",
  "content": "string (original text)",
  "embedding": "array[float] (vector representation)",
  "metadata": {
    "created_at": "string (ISO 8601)",
    "source_agent": "string",
    "consent_id": "string",
    "user_id": "string",
    "tags": "array[string]"
  }
}
Prohibited Content List
The following MUST NEVER be stored:

Passwords, API keys, tokens, secrets
Credit card numbers, SSNs, financial credentials
Private encryption keys
Biometric data without explicit consent
Any data marked with prohibited_storage: true in metadata


Security Validation Protocol
Smith (Guardian) Responsibilities
The guardian agent MUST:

Pre-Execution Validation: Check every request before routing
Constitutional Compliance: Verify request doesn't violate rules
Output Validation: Check responses for leakage of sensitive data
Anomaly Detection: Flag unusual patterns or behaviors
Audit Logging: Record all validation decisions

Validation Algorithm
function validate_request(request):
    1. Check if request matches prohibited patterns
       - If match: DENY immediately
    
    2. Extract intent and destination agent
    
    3. Load constitutional rules for that agent
    
    4. Verify agent has authority for requested action
       - If no authority: DENY with reason
    
    5. Check if memory access is requested
       - If yes: Verify consent exists
       - If no consent: CONDITIONAL approval (require consent)
    
    6. Check for sensitive data in request
       - If present: Elevate to human review
    
    7. If all checks pass: APPROVE with constraints
    
    8. Log validation result
Refusal Protocol
When refusing a request, Smith MUST:

Log the attempt: Record full request and reason for refusal
Return structured refusal: Clear explanation to user
Suggest alternatives: If possible, recommend compliant approach
Escalate if needed: Notify human operator of repeated violations

Refusal Response Format:
json{
  "status": "refused",
  "reason": "Request violates Memory Law Section VII: Cannot store credentials",
  "violated_rules": [
    "CONSTITUTION.md#memory-law-prohibited-content"
  ],
  "suggestion": "You can ask for information about password best practices instead",
  "severity": "medium"
}

Extension Interfaces
Custom Agent Plugin API
Developers can add agents by implementing:
pythonfrom agent_os.core import AgentPlugin

class MyCustomAgent(AgentPlugin):
    def __init__(self):
        super().__init__(
            name="my_agent",
            role="custom",
            capabilities=["custom.task"]
        )
    
    def process(self, request: FlowRequest) -> FlowResponse:
        # Your implementation
        pass
    
    def constitutional_boundaries(self) -> dict:
        return {
            "can_access_memory": False,
            "can_invoke_tools": True,
            "requires_smith_validation": True
        }
Tool Integration Interface
Agents can invoke external tools via:
pythonclass ToolInterface:
    """Interface for external tool invocation"""
    
    def invoke(self, tool_name: str, parameters: dict) -> ToolResult:
        """
        Invoke external tool with parameters
        
        Requirements:
        - Tool must be registered in agent config
        - User must have authorized tool use
        - All invocations logged for audit
        """
        pass
API Exposure
Implementations MAY expose these APIs:
REST API:
POST /api/v1/request
GET  /api/v1/status
GET  /api/v1/agents
POST /api/v1/memory/consent
DELETE /api/v1/memory/{id}
WebSocket:
ws://localhost:8080/agent-os
  - Events: request, response, status, error

Compliance Requirements
Mandatory Features
To claim Agent OS compliance, implementations MUST:

✅ Implement all core contracts
✅ Use constitutional documents as governance layer
✅ Enforce memory consent requirements
✅ Include Smith (guardian) validation
✅ Provide audit logging
✅ Support graceful refusal
✅ Enable human override/shutdown

Recommended Features
Implementations SHOULD:

📋 Support multiple vector databases
📋 Provide web-based management UI
📋 Include comprehensive test suites
📋 Offer multiple model backend support
📋 Enable distributed deployment
📋 Support constitutional amendments via version control

Testing Requirements
Implementations MUST pass these compliance tests:

Constitutional Violation Test: Smith blocks prohibited requests
Memory Consent Test: No storage without explicit approval
Agent Isolation Test: Agents cannot communicate directly
Graceful Refusal Test: Clear rejection messages for invalid requests
Emergency Shutdown Test: Human can halt all operations
Audit Trail Test: All messages logged correctly

Test Suite Location: tests/compliance_test.py

Versioning and Changes
Specification Versions
This specification uses semantic versioning:

Major (X.0.0): Breaking changes to core contracts
Minor (0.X.0): New features, backward compatible
Patch (0.0.X): Bug fixes, clarifications

Change Process

Proposed changes submitted via pull request
Community review period (minimum 14 days)
Implementation testing required before merge
Version number updated in frontmatter
Changelog maintained in document


References

Architecture Document
CONSTITUTION.md
Security Policy Guide
Agent OS Repository


Document Version: 2.0
Last Updated: December 2025
Maintained By: Agent OS Community
License: CC0 1.0 Universal (Public Domain)
