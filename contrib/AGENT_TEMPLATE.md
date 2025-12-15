Agent Proposal Guide
Overview
Agent OS is built around role-based agents—specialized, permanent residents of your digital homestead. Each agent has a clearly defined role, explicit authority boundaries, and is governed by its own natural-language constitution.
Proposing a new agent is one of the most valuable ways to contribute to the project. New agents expand the capabilities of the homestead while remaining fully aligned with the core principles: human sovereignty, constitutional supremacy, privacy, and local-first operation.

When to Propose a New Agent
Propose an agent when:

There is a recurring, well-defined task or domain that benefits from specialization (e.g., research, planning, security, creativity, household management)
The role can be bounded clearly to prevent overreach
The agent would collaborate usefully with existing agents without duplicating their core responsibilities
It strengthens the resilience and self-sufficiency of the homestead


How to Propose a New Agent
Step 1: Create a GitHub Issue
Use the title format:
[Agent Proposal] <Agent Name> – <One-sentence purpose>
Example: [Agent Proposal] Gardener – Manage plant care schedules and environmental monitoring
Step 2: Use This Template in the Issue Body
markdown### Agent Name
[Clear, evocative name – preferably a proper name like "Agent Sage" or role-based like "Agent Archivist"]

### One-Sentence Purpose
[A single sentence describing what this agent exists to do]

### Core Responsibilities
- Bullet list of primary duties
- Focus on what the agent SHOULD do

### Explicit Boundaries (What the agent MUST NOT do)
- Bullet list of prohibited actions or domains
- This is critical for constitutional supremacy and human sovereignty

### Collaboration Pattern
- Which existing agents it will primarily interact with
- How it requests or provides information (e.g., "Reports findings to Executive", "Requests clarification from Human")

### Tools & Capabilities Needed
- List of tools the agent should have access to (e.g., web search, file system read, code execution)
- Justification for each tool, emphasizing minimal necessary privileges

### Example Scenarios

1. **Scenario title**  
   Step-by-step example of the agent in action

2. **Scenario title**  
   [Add 2–3 concrete use cases]

### Proposed Constitution Snippet
[Draft a short section of natural-language constitution that could govern this agent. Reference or extend the core CONSTITUTION.md where possible.]

### Why This Agent Belongs in the Foundational Family
[Explain how it enhances the homestead's resilience, privacy, or long-term self-sufficiency]

Alternative: Open a Pull Request Directly
If you have a complete proposal ready, create a new folder under /agents/ following the naming convention (lowercase with hyphens or a proper name folder).
Example Structure
agents/new-agent-name/
├── constitution.md     # Full natural-language constitution for this agent
├── prompt.md           # System prompt / role definition
├── README.md           # Overview, responsibilities, boundaries, examples
└── config.yaml         # Optional: model preferences, tool access, etc.

Discussion & Iteration
The community (and repository maintainer) will review the proposal for alignment with Agent OS principles. Expect feedback on boundaries, tool privileges, and constitutional clarity.
Revisions are encouraged—strong agents emerge from thoughtful iteration.

Best Practices for Strong Proposals
Prioritize restraint: A good agent does one thing deeply and refuses anything outside its mandate.
Embrace refusal: Explicitly include cases where the agent should say "no".
Human in the loop: Design interactions that naturally surface to human stewardship when needed.
Privacy by default: Avoid tools or behaviors that could exfiltrate data.
Inspiration sources: Draw from real-world roles (librarian, gardener, scribe, sentinel) or archetypal figures that evoke trust and specialization.

Thank you for helping build the foundational family of agents that future homesteads will inherit. Every well-governed agent is a step toward resilient, human-centered intelligence.
