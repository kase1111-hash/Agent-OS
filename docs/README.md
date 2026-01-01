# Agent-OS Documentation

Welcome to the Agent-OS documentation. This guide will help you navigate the comprehensive documentation for this language-native, constitutionally-governed AI operating system.

## Quick Start

**New to Agent-OS?** Start here:
1. [README.md](../README.md) - Project overview and vision
2. [START_HERE.md](../START_HERE.md) - Windows quick start guide (beginner-friendly)
3. [FAQ.md](./FAQ.md) - Frequently asked questions (all levels)
4. [RUNNING_AND_COMPILING.md](./RUNNING_AND_COMPILING.md) - Detailed installation and running guide
5. [examples/basic-homestead-setup.md](../examples/basic-homestead-setup.md) - Step-by-step setup guide

## Core Documents

### Constitutional Framework
- [CONSTITUTION.md](../CONSTITUTION.md) - Supreme law of Agent-OS, defining governance and agent authorities
- [glossary.md](./glossary.md) - Key terminology and concepts

### Project Planning
- [ROADMAP.md](../ROADMAP.md) - Development timeline and milestones (Phase 0-4 through 2028)
- [CONTRIBUTING.md](../CONTRIBUTING.md) - How to contribute to the project
- [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) - Community standards and guidelines

## Understanding Agent-OS

### Philosophy & Vision

**What Agent-OS Is (and Isn't)**
- [lni-manifesto.md](./lni-manifesto.md) - Core principles of Language-Native Intelligence
- [Not-an-OS.md](./Not-an-OS.md) - Clarification: This is NOT an AI OS, but a language-native OS
- [Product-Story.md](./Product-Story.md) - Product vision and modular ecosystem approach

**Conceptual Framework**
- [The_Document_Revolution.md](./The_Document_Revolution.md) - Vision of documents as executable specifications
- [Conversational-Kernel.md](./Conversational-Kernel.md) - Natural language kernel architecture (4-layer system)
- [NLOS-Layers.md](./NLOS-Layers.md) - 14-layer Natural Language OS architecture (brain analogue mapping)

### User Experience
- [the-first-24-hours.md](./the-first-24-hours.md) - Narrative walkthrough of living with Agent-OS
- [Bring-Home-Ceremony-Bootstrapping-Flow.md](./Bring-Home-Ceremony-Bootstrapping-Flow.md) - Ceremonial setup and trust establishment process

### Synthetic Mind Architecture
- [Synthetic-Cognitive-Architecture.md](./Synthetic-Cognitive-Architecture.md) - Brain analogue mapping for agent cognition
- [Synthetic-Mind-Stack.md](./Synthetic-Mind-Stack.md) - Interconnected psychological modules
- [Personality-Prism.md](./Personality-Prism.md) - Agent personality framework (6 psychological facets)

## Technical Documentation

### Architecture
- [technical/architecture.md](./technical/architecture.md) - Complete system architecture and reference implementation
- [technical/Specification.md](./technical/Specification.md) - Detailed technical specifications
- [technical/whitepaper.md](./technical/whitepaper.md) - Comprehensive technical whitepaper
- [technical/TECHNICAL_ADDENDUM.md](./technical/TECHNICAL_ADDENDUM.md) - Additional implementation details

### Advanced Topics
- [technical/LNI-testable-theory.md](./technical/LNI-testable-theory.md) - Language-Native Intelligence testable hypotheses
- [technical/Self_Healing_Workflow.md](./technical/Self_Healing_Workflow.md) - Error recovery and self-healing mechanisms

### Reference Guides
- [repository-structure.md](./repository-structure.md) - Complete directory structure with explanations
- [Spec-to-Repo-Mapping.md](./Spec-to-Repo-Mapping.md) - Maps specification documents to repository files

## Governance & Security

### Security
- [governance/security.md](./governance/security.md) - **Comprehensive security policy**: vulnerability reporting, architecture, threat models, best practices
- [governance/agent-os-security-policy-making-guide.md](./governance/agent-os-security-policy-making-guide.md) - User guide for creating document-based security policies
- [technical/red-team.md](./technical/red-team.md) - Security testing guidelines and adversarial scenarios

### Attack Detection & Auto-Remediation (NEW)
Agent Smith now includes a comprehensive attack detection and auto-remediation system:
- **Real-time Detection**: Monitor boundary daemon events and SIEM feeds for attack indicators
- **SIEM Integration**: Connect to Splunk, Elasticsearch, Microsoft Sentinel, and Syslog
- **LLM-Powered Analysis**: Deep attack analysis using Sage agent with MITRE ATT&CK mapping
- **Auto-Remediation**: Generate patches to fix vulnerabilities with sandbox testing
- **Git Integration**: Automatically create PRs for security fixes
- **Multi-Channel Notifications**: Alerts via Slack, Email, PagerDuty, Teams, and webhooks
- **YAML Configuration**: Flexible configuration with environment variable substitution
- **Security API**: RESTful endpoints for attack management and recommendations

### Policy & Licensing
- [governance/policy-brief.md](./governance/policy-brief.md) - High-level policy framework summary
- [governance/license-strategy.md](./governance/license-strategy.md) - CC0 licensing approach and rationale
- [LICENSE](../LICENSE) - CC0 1.0 Universal Public Domain Dedication

## Agents

### Core Agent Families

**Foundational Agents** (each has `constitution.md` and `prompt.md`)
- [agents/executive/](../agents/executive/) - Orion & Aurora (coordination and execution)
- [agents/researcher/](../agents/researcher/) - Information gathering
- [agents/planner/](../agents/planner/) - Strategic planning
- [agents/guardian/](../agents/guardian/) - Security and governance enforcement
- [agents/sage/](../agents/sage/) - Knowledge and wisdom
- [agents/seshat/](../agents/seshat/) - Memory and record-keeping
- [agents/quill/](../agents/quill/) - Writing and documentation
- [agents/muse/](../agents/muse/) - Creative generation

**Specialized Agents**
- [agents/Librarian.md](../agents/Librarian.md) - Library and knowledge management
- [agents/MENTOR.md](../agents/MENTOR.md) - Guidance and teaching
- [agents/GYM-Trainer.md](../agents/GYM-Trainer.md) - Training and evaluation
- [agents/Mirror-Mirror.md](../agents/Mirror-Mirror.md) - Self-reflection
- [agents/OBSERVER/](../agents/OBSERVER/) - System monitoring

### Agent Development
- [agents/Custom-Personalities.md](../agents/Custom-Personalities.md) - Guide for creating custom agents
- [agents/template/](../agents/template/) - Templates for new agents (constitution, prompt, rules)
- [agents/agent-smith-ruleset.md](../agents/agent-smith-ruleset.md) - Smith (watchdog) agent rules
- [agents/agent-whisper-ruleset.md](../agents/agent-whisper-ruleset.md) - Whisper (orchestrator) agent rules

## Examples & Guides

### Getting Started
- [examples/basic-homestead-setup.md](../examples/basic-homestead-setup.md) - Minimal local-first deployment guide
- [examples/constitution-examples.md](../examples/constitution-examples.md) - Example constitutions for different use cases
- [examples/multi-agent-task.md](../examples/multi-agent-task.md) - Multi-agent collaboration examples

### Contributing
- [contrib/Contributing.md](../contrib/Contributing.md) - Detailed contribution guidelines
- [contrib/AGENT_TEMPLATE.md](../contrib/AGENT_TEMPLATE.md) - Template for proposing new agents
- [contrib/ISSUES.md](../contrib/ISSUES.md) - Issue reporting guidelines

## Marketing & Communication

- [LANDING.md](./LANDING.md) - Marketing and feature highlights
- [PRESS_RELEASE.md](./PRESS_RELEASE.md) - Public announcement document
- [LF1M.md](./LF1M.md) - Developer recruitment ("Looking For Group")
- [Why is Agent OS So Small.md](./Why is Agent OS So Small.md) - Minimal design philosophy explanation

## Document Organization

### By Audience

**For Families & New Users**
- Start with [FAQ.md](./FAQ.md) (Kids/Teens section)
- Read [the-first-24-hours.md](./the-first-24-hours.md) for experience overview
- Follow [examples/basic-homestead-setup.md](../examples/basic-homestead-setup.md)

**For Developers & Contributors**
- Read [technical/architecture.md](./technical/architecture.md)
- Study [CONSTITUTION.md](../CONSTITUTION.md)
- Review [CONTRIBUTING.md](../CONTRIBUTING.md)
- Check [ROADMAP.md](../ROADMAP.md)

**For Researchers & Security Professionals**
- Study [technical/whitepaper.md](./technical/whitepaper.md)
- Review [governance/security.md](./governance/security.md)
- Examine [technical/LNI-testable-theory.md](./technical/LNI-testable-theory.md)
- Read [technical/red-team.md](./technical/red-team.md)

### By Topic

**Understanding the Vision**
1. [lni-manifesto.md](./lni-manifesto.md) - Why language-native intelligence?
2. [Not-an-OS.md](./Not-an-OS.md) - What makes this different?
3. [Product-Story.md](./Product-Story.md) - The complete product vision

**Technical Deep Dive**
1. [technical/architecture.md](./technical/architecture.md) - System design
2. [technical/Specification.md](./technical/Specification.md) - Detailed specs
3. [technical/whitepaper.md](./technical/whitepaper.md) - Complete technical overview

**Security & Governance**
1. [CONSTITUTION.md](../CONSTITUTION.md) - Supreme law
2. [governance/security.md](./governance/security.md) - Security policy
3. [agents/guardian/constitution.md](../agents/guardian/constitution.md) - Guardian agent authority

## Status & Versions

**Current Phase:** Phase 0 complete, Phase 1 in progress (Q1 2026)
**Project Status:** Pre-alpha development (~90% implementation complete)
**License:** CC0 1.0 Universal (Public Domain)
**Codebase:** ~64,000 lines of Python across 244 files

See [ROADMAP.md](../ROADMAP.md) for detailed development timeline.

## Recent Updates

- **January 2026**: Added comprehensive attack detection & auto-remediation system to Agent Smith
- **January 2026**: Added SIEM integration (Splunk, Elasticsearch, Sentinel, Syslog)
- **January 2026**: Added multi-channel notification system (Slack, Email, PagerDuty, Teams)
- **January 2026**: Added YAML-based configuration for attack detection
- **January 2026**: Added Git integration for automatic security fix PRs
- **January 2026**: Added LLM-powered attack analysis with MITRE ATT&CK mapping
- **January 2026**: Added Security API endpoints for attack management
- **January 2026**: Added custom exception classes for improved error handling
- **December 2025**: Added Windows build scripts (`build.bat`, `start.bat`)
- **December 2025**: Added beginner-friendly quick start guide (`START_HERE.md`)

---

## Quick Reference

| I want to... | Go to... |
|--------------|----------|
| **Get started on Windows** | [START_HERE.md](../START_HERE.md) |
| **Get started quickly** | [RUNNING_AND_COMPILING.md](./RUNNING_AND_COMPILING.md) |
| **Understand the philosophy** | [lni-manifesto.md](./lni-manifesto.md) |
| **Read technical specs** | [technical/architecture.md](./technical/architecture.md) |
| **Report a security issue** | [governance/security.md](./governance/security.md#reporting-vulnerabilities) |
| **Configure attack detection** | [governance/security.md](./governance/security.md#attack-detection) |
| **Contribute code** | [CONTRIBUTING.md](../CONTRIBUTING.md) |
| **Create a new agent** | [agents/template/](../agents/template/) |
| **Understand agent roles** | [CONSTITUTION.md](../CONSTITUTION.md) |
| **Learn about security** | [governance/security.md](./governance/security.md) |
| **See the roadmap** | [ROADMAP.md](../ROADMAP.md) |
| **Run tests** | [RUNNING_AND_COMPILING.md](./RUNNING_AND_COMPILING.md#testing) |
| **Deploy with Docker** | [RUNNING_AND_COMPILING.md](./RUNNING_AND_COMPILING.md#running-with-docker-compose) |

---

*This documentation is released under CC0 1.0 Universal (Public Domain). For questions or suggestions, see [CONTRIBUTING.md](../CONTRIBUTING.md).*
