# Security Policy

## Reporting a Vulnerability

**Please DO NOT report security vulnerabilities through public GitHub issues.**

We take security seriously. If you discover a security vulnerability in Agent-OS, please report it through one of these secure channels:

### Option 1: GitHub Security Advisories (Preferred)

1. Navigate to the [Security tab](https://github.com/kase1111-hash/Agent-OS/security)
2. Click "Report a vulnerability"
3. Fill out the advisory form with details
4. Submit privately

### Option 2: Private Contact

For vulnerabilities that don't require encrypted communication:
- Open a GitHub issue requesting private security discussion
- Include `[SECURITY]` in your message subject
- A maintainer will provide a secure contact method

## What to Include

- **Vulnerability Type**: Constitutional Bypass, Memory Leak, Injection Attack, etc.
- **Affected Components**: Whisper Orchestrator, Smith Watchdog, Seshat Memory, etc.
- **Severity Assessment**: Critical/High/Medium/Low with reasoning
- **Steps to Reproduce**: Detailed reproduction steps
- **Proof of Concept**: Code, screenshots, or logs
- **Potential Impact**: What could an attacker accomplish?
- **Suggested Fix** (optional): Your ideas for addressing the issue

## Response Timeline

| Severity | Initial Response | Resolution Target |
|----------|-----------------|-------------------|
| Critical | 48 hours | 7-14 days |
| High | 48 hours | 14-30 days |
| Medium | 7 days | 30-60 days |
| Low | 7 days | 60-90 days |

## Supported Versions

| Version | Status | Security Updates |
|---------|--------|-----------------|
| 1.x (Future) | Stable | Full support |
| 0.9.x (Beta) | Beta | Active development |
| 0.5.x (Alpha) | Alpha | Best effort |
| < 0.5 | Experimental | No support |

**Current Status**: Pre-alpha development (Phase 0 complete, Phase 1 starting Q1 2026)

## Security Architecture

Agent-OS is built on foundational security principles:

- **Constitutional Supremacy**: All agent behavior governed by natural language constitutions
- **Defense in Depth**: Multiple security layers from Human Steward to Audit Logging
- **Principle of Least Privilege**: Agents have minimum authority needed for their role
- **Zero Trust Architecture**: Every request validated regardless of source
- **Privacy by Design**: Local-first computation, explicit consent for memory
- **Fail Secure**: System defaults to secure state on failure

## Safe Harbor

We operate under a Good Faith Security Research safe harbor policy. We will not pursue legal action against individuals who:

- Report vulnerabilities through appropriate channels
- Do not exploit vulnerabilities beyond proof of concept
- Do not access, modify, or delete user data
- Make good faith efforts to avoid service disruption
- Allow reasonable time for remediation before disclosure

## Full Security Documentation

For comprehensive security information including:

- Detailed threat model
- Attack detection system configuration
- SIEM integration guides
- Security best practices
- Known limitations

See: [docs/governance/security.md](./docs/governance/security.md)

---

**Document Version**: 1.0
**Last Updated**: January 2026
**License**: CC0 1.0 Universal (Public Domain)
