📜 CONTRIBUTING.md: Agent OS Kernel
Thank you for your interest in the Agent Operating System (Agent OS), released under the Creative Commons license. The Agent OS is an ambitious, community-driven effort to build a sovereign, auditable, and human-controlled AI infrastructure.
Because the Agent OS is governed by a formal Constitution, all contributions—especially to the core architecture, governance, and Agent roles—must be evaluated not just for code quality, but for constitutional compliance.
🤝 Code of Conduct
The Agent OS community strives for clarity, respect, and technical rigor.
Respect Sovereignty: Always assume that the Human Steward (the user) is the ultimate authority. Contributions that diminish human oversight, obscure agent behavior, or undermine auditability will be rejected.
Clarity Over Complexity: All code, documentation, and agent definitions must be transparent and easy to inspect. Avoid "black box" solutions where possible.
Professionalism: Maintain a respectful and constructive tone in all discussions, pull requests, and issues. We are building the governance layer of future intelligence, and our collaboration should reflect that goal.
🚀 Contribution Flow
All contributions begin by opening an Issue to discuss the proposed change. Once approved, the work should be submitted via a Pull Request (PR).
1. New Agent Roles (Section IV)
New Agent Roles are encouraged, but they must adhere strictly to the Non-Overlap of Powers and Role-Bound Authority mandates.
Requirement
Constitutional Mandate
Description for PR
Clear Boundaries
IV. Role-Bound Authority
The role must define its explicit functions and explicit limitations (what it Cannot Do).
Memory Scope
VII. Memory Isolation
Define the minimum Memory access required (Ephemeral, Working, or specific Long-Term access). No Agent may grant itself Memory write authority.
Refusal Protocol
IX. Refusal Doctrine
Define the specific conditions under which the Agent must halt or refuse, and the explicit, non-deceptive message it returns.
Conflict Check
IV. Non-Overlap of Powers
Demonstrate how the new Role avoids overlap or conflict with the Core Agent Roster (Whisper, Smith, Sage, Seshat, Quill, Muse).

PR Title Format: [AGENT: ROLE NAME] Add new agent role: Brief description
2. Implementation & Flow Logic (Phase 0 Kernel)
Contributions to the core routing, security interlocks, and code base must adhere to the Flow & Routing Law (Section VIII).
Auditable Flows: All logic involving flow and authority transfer between Agents must be fully auditable. PRs must include tests proving that routing decisions are logged and inspectable.
Security Interlocks: Changes to routing must not bypass the required Pre-execution Validation checks enforced by the Smith (Guardian) agent. The Guardian must always be invoked for actions involving external interfaces, memory modification, or system-level changes.
Minimum Context: PRs must demonstrate that the code adheres to Context Minimization—only the minimum necessary context is passed between Agents.
PR Title Format: [FLOW: AGENT NAME] Fix/Improve routing logic for X
3. Proposals to Amend the Constitution
Constitutional amendments are the most serious contribution. The Constitution is designed to be highly resistant to change to maintain stability and trust.
Immutable Sections (e.g., I, III, V, IX): Cannot be modified without a new major version release and overwhelming community consensus. Proposals to these sections must fundamentally strengthen Human Sovereignty or Security.
Amendable/Conditional Sections (e.g., XI, XIII, XV): Can be modified via the standard PR process.
Proposal Requirements:
Open an Issue: Clearly state the section, the proposed text change, and the specific Rationale (why the current text fails or limits the system).
Impact Analysis: Explicitly detail the potential impact of the amendment on the Security & Refusal Doctrine and on Human Sovereignty.
PR Title Format: [AMENDMENT: SECTION #] Proposed change to: Brief subject
🛠 Getting Started
Clone the Repository: git clone [Your GitHub URL]
Review the Constitution: Read the Constitution and the Addendum thoroughly to understand the governing law.
Open an Issue: Before writing significant code, open an issue to outline your proposal. This saves effort and ensures alignment with the community vision.
Fork and Branch: Fork the repository and create a new branch for your contribution (git checkout -b feature/my-new-agent).
Submit PR: Ensure your code is well-commented and includes unit tests that prove constitutional compliance.
Thank you for helping us build the future of governed AI!
