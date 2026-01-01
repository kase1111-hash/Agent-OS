Contributing to Agent OS Kernel
Thank you for your interest in the Agent Operating System (Agent OS), released under the Creative Commons license. The Agent OS is an ambitious, community-driven effort to build a sovereign, auditable, and human-controlled AI infrastructure.
Because the Agent OS is governed by a formal Constitution, all contributions—especially to the core architecture, governance, and Agent roles—must be evaluated not just for code quality, but for constitutional compliance.

Code of Conduct
The Agent OS community strives for clarity, respect, and technical rigor:
Respect Sovereignty: Always assume that the Human Steward (the user) is the ultimate authority. Contributions that diminish human oversight, obscure agent behavior, or undermine auditability will be rejected.
Clarity Over Complexity: All code, documentation, and agent definitions must be transparent and easy to inspect. Avoid "black box" solutions where possible.
Professionalism: Maintain a respectful and constructive tone in all discussions, pull requests, and issues. We are building the governance layer of future intelligence, and our collaboration should reflect that goal.

Contribution Flow
All contributions begin by opening an Issue to discuss the proposed change. Once approved, the work should be submitted via a Pull Request (PR).

1. New Agent Roles (Section IV)
New Agent Roles are encouraged, but they must adhere strictly to the Non-Overlap of Powers and Role-Bound Authority mandates.
Requirements for New Agent PRs
RequirementConstitutional MandateDescriptionClear BoundariesIV. Role-Bound AuthorityDefine explicit functions and limitations (what it Cannot Do)Memory ScopeVII. Memory IsolationDefine minimum Memory access required (Ephemeral, Working, or specific Long-Term access). No Agent may grant itself Memory write authorityRefusal ProtocolIX. Refusal DoctrineDefine specific conditions under which the Agent must halt or refuse, and the explicit, non-deceptive message it returnsConflict CheckIV. Non-Overlap of PowersDemonstrate how the new Role avoids overlap or conflict with the Core Agent Roster (Whisper, Smith, Sage, Seshat, Quill, Muse)
PR Title Format: [AGENT: ROLE NAME] Add new agent role: Brief description

2. Implementation & Flow Logic (Phase 0 Kernel)
Contributions to the core routing, security interlocks, and code base must adhere to the Flow & Routing Law (Section VIII).
Auditable Flows: All logic involving flow and authority transfer between Agents must be fully auditable. PRs must include tests proving that routing decisions are logged and inspectable.
Security Interlocks: Changes to routing must not bypass the required Pre-execution Validation checks enforced by the Smith (Guardian) agent. The Guardian must always be invoked for actions involving external interfaces, memory modification, or system-level changes.
Minimum Context: PRs must demonstrate that the code adheres to Context Minimization—only the minimum necessary context is passed between Agents.
PR Title Format: [FLOW: AGENT NAME] Fix/Improve routing logic for X

3. Proposals to Amend the Constitution
Constitutional amendments are the most serious contribution. The Constitution is designed to be highly resistant to change to maintain stability and trust.
Amendment Categories
Immutable Sections (e.g., I, III, V, IX): Cannot be modified without a new major version release and overwhelming community consensus. Proposals to these sections must fundamentally strengthen Human Sovereignty or Security.
Amendable/Conditional Sections (e.g., XI, XIII, XV): Can be modified via the standard PR process.
Proposal Requirements

Open an Issue: Clearly state the section, the proposed text change, and the specific Rationale (why the current text fails or limits the system)
Impact Analysis: Explicitly detail the potential impact of the amendment on the Security & Refusal Doctrine and on Human Sovereignty

PR Title Format: [AMENDMENT: SECTION #] Proposed change to: Brief subject

Getting Started

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/kase1111-hash/Agent-OS.git
   cd Agent-OS
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"  # Development tools
   ```

3. **Review the Constitution:**
   Read [CONSTITUTION.md](../CONSTITUTION.md) and the Addendum thoroughly to understand the governing law.

4. **Open an Issue:**
   Before writing significant code, open an issue to outline your proposal. This saves effort and ensures alignment with the community vision.

5. **Fork and Branch:**
   Fork the repository and create a new branch for your contribution:
   ```bash
   git checkout -b feature/my-new-agent
   ```

6. **Run Tests:**
   Ensure all tests pass before submitting:
   ```bash
   pytest tests/
   ```

7. **Code Quality:**
   Format your code and run type checks:
   ```bash
   black src/ tests/
   isort src/ tests/
   mypy src/
   ```

8. **Submit PR:**
   Ensure your code is well-commented and includes unit tests that prove constitutional compliance.

---

## Development Environment

### Required Tools

| Tool | Purpose | Installation |
|------|---------|--------------|
| Python 3.10+ | Runtime | [python.org](https://python.org) |
| pip | Package manager | Included with Python |
| pytest | Testing | `pip install pytest` |
| black | Formatting | `pip install black` |
| mypy | Type checking | `pip install mypy` |

### Running the Application

```bash
python -m uvicorn src.web.app:get_app --factory --host 0.0.0.0 --port 8080 --reload
```

### Running Tests

```bash
# All tests
pytest tests/

# With coverage
pytest --cov=src tests/

# Specific test file
pytest tests/test_kernel.py -v
```

---

## Project Structure

```
Agent-OS/
├── src/                    # Source code
│   ├── agents/             # Agent implementations
│   ├── core/               # Constitutional kernel
│   ├── kernel/             # Conversational kernel
│   ├── memory/             # Memory vault
│   ├── messaging/          # Message bus
│   ├── boundary/           # Security daemon
│   └── web/                # Web interface
├── tests/                  # Test suite
├── docs/                   # Documentation
├── agents/                 # Agent constitutions
└── examples/               # Usage examples
```

---

## Resources

- [README.md](../README.md) - Project overview
- [docs/README.md](../docs/README.md) - Documentation index
- [ROADMAP.md](../ROADMAP.md) - Development timeline
- [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) - Community standards

---

Thank you for helping us build the future of governed AI!

---

*Last Updated: January 2026*
*License: CC0 1.0 Universal (Public Domain)*
