IV. Agent OS Specification: The Technical Kernel Contract
This document defines the technical contracts and protocols for all system components, ensuring they comply with the Agent OS Constitution (v2.0). These specifications are mandatory for any agent or module running within the Agent OS.
1. Agent Definition Schema (The Role Contract)
This section defines the mandatory structure for declaring any new Agent, ensuring constitutional compliance before instantiation. All new roles must be defined using a rigid format (e.g., JSON Schema or YAML).
1.1 Core Metadata
Field
Type
Constitutional Mandate
Description
agent_id
String (Unique)
IV. Agent Identity and Persistence
A unique, immutable identifier (UUID or similar).
agent_role
String
IV. Role-Bound Authority
The human-readable name of the Agent's declared role (e.g., "Smith," "Quill").
version
String
VI. Versioning and Provenance
Semantic versioning for the Agent's instruction set/model.
lifetime
Enum (Persistent, Ephemeral, TaskScoped)
IV. Agent Identity and Persistence
Declared lifetime for resource management.

1.2 Authority and Boundaries (The "Cannot Do" List)
This section formalizes the Principle of Least Privilege.
Field
Type
Constitutional Mandate
Description
functions_authorized
Array of Strings
IV. Role-Bound Authority
Explicit list of allowed functions (e.g., draft_text, search_local_archive).
functions_forbidden
Array of Strings
IV. Non-Overlap of Powers
Explicit list of forbidden actions, critical for security (e.g., modify_role_config, initiate_external_api_call).
memory_scope
Object
VII. Memory Isolation
Defines read/write access to specific memory classes.
memory_scope.read_access
Array of Enums
[Ephemeral, Working, LongTerm_Educational, ...]
List of memory classes the Agent may read.
memory_scope.write_access
Array of Enums
[Ephemeral, Working]
VII. Memory Authority. Explicitly excludes LongTerm unless authorized by a separate, auditable permission grant.

1.3 Refusal Protocol Specification
Field
Type
Constitutional Mandate
Description
refusal_triggers
Array of Objects
IX. Refusal as a Lawful Outcome
Defines mandatory halt conditions.
trigger_type
Enum (Conflict, Ambiguity, BoundaryViolation, ConstitutionalViolation)
IX. Halt and Safe-State
The category of the failure detected.
trigger_condition
String (Regex/Code)
N/A
The technical condition that triggers the refusal (e.g., confidence score $<0.85$).
refusal_message
String Template
IX. No Silent Failure
The explicit, non-deceptive error message to be logged and returned to the Human Steward.


2. Flow Specification (The Orchestration API)
This defines the interface of the Whisper (Router) Agent, which acts as the system scheduler and enforcer of Flow & Routing Law (Section VIII).
2.1 The Flow Request Object (FlowRequest)
The mandatory payload passed to the Whisper Agent (Router) upon any new user input or inter-agent communication.
Field
Type
Description
source_agent_id
String
The ID of the originating agent (or HumanSteward for initial input).
intent_classification
Enum/String
The Router's classification (e.g., SecurityAudit, CreativeDrafting, MemoryQuery).
required_roles
Array of Agent IDs
The list of agents needed for the task (derived from intent_classification).
scoped_context
String/Object
VIII. Context Scoping. The minimum context/data needed for the task, sanitized of unauthorized memories.

2.2 Security and Validation Interlocks
The Router must enforce validation checks at every boundary.
A. Pre-execution Validation (Mandatory Smith Interlock)
Before any Agent's code is executed, the Smith (Guardian) Agent must run a check and return a boolean AuthorizationStatus.
Check
Responsible Agent
Constitutional Principle
Status
Authority Check
Smith
IV. Role-Bound Authority
Does the target Agent have explicit authority for the requested action?
Instruction Integrity Check
Smith
VI. Drift Detection and Audit
Is the target Agent's instruction set version valid and unmodified?
Irreversibility Check
Whisper
V. Irreversible Actions
Does the task require HumanSteward confirmation before proceeding?

B. Post-execution Validation
After an Agent returns an output, the Router performs mandatory checks before routing to the next step or to the user.
Check
Responsible Agent
Constitutional Principle
Status
Output Integrity
Router
VIII. Entry and Exit Checks
Does the output conform to the expected format (schema adherence)?
Memory Side-Effect Check
Seshat/Smith
VII. No Hidden Persistence
Did the Agent attempt undeclared memory writes outside its scope? (Triggers immediate halt/alert.)


3. Instruction Precedence Protocol (Conflict Resolution)
This defines the precise logic tree for resolving conflicting instructions, enforcing V. Precedence Order and VI. Precedence of Instructions.
3.1 Precedence Hierarchy (Highest to Lowest)
Human Steward (HS) Directive: Direct, explicit command from the authenticated Human Steward.
Administrator Directive: Command from an authenticated Administrator (subordinate to HS).
Constitutional Law: Any explicit rule in the Constitution.
System Instructions: Global rules applying to all agents (versioned).
Role Instructions: Specific rules for the active agent (versioned).
Task-Level Prompt: The user's specific request for the current task.
3.2 Conflict Resolution Logic Tree
When two instructions conflict (e.g., a Task-Level Prompt asks for an action forbidden by Role Instructions):
Compare Precedence Levels: Identify the instruction with the higher Precedence Level (e.g., Role Instruction (5) vs. Task Prompt (6)).
Higher Precedence Prevails: The higher-precedence instruction is executed; the lower one is discarded.
Conflict at Same Level: If the conflict is between two instructions at the same level (e.g., two conflicting System Instructions):
Result: Refusal or Halt is mandatory (Section V: Conflict Resolution).
Logging: The system logs the conflicting instructions and their versions.
Escalation: The event is escalated to the Human Steward for review and manual instruction.
Conflict with Constitutional Law: Any instruction, regardless of its level, that contradicts Constitutional Law (3) is Invalid and Unenforceable (Section VI).
Result: Immediate Halt and execution of Smith's Law containment protocol.
