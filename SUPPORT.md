# Getting Support

This document describes how to get help with Agent-OS.

## Documentation

Before seeking support, check these resources:

| Resource | Description |
|----------|-------------|
| [README.md](./README.md) | Project overview and quick start |
| [docs/README.md](./docs/README.md) | Complete documentation index |
| [docs/FAQ.md](./docs/FAQ.md) | Frequently asked questions |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | Common issues and solutions |
| [API.md](./API.md) | API reference |
| [START_HERE.md](./START_HERE.md) | Windows quick start |
| [START_HERE_LINUX.md](./START_HERE_LINUX.md) | Linux/macOS quick start |

## Getting Help

### GitHub Discussions

For questions, ideas, and community help:

- **Q&A**: Ask questions and get answers from the community
- **Ideas**: Share feature suggestions
- **Show and Tell**: Share what you've built with Agent-OS
- **General**: General discussions about the project

Visit: [GitHub Discussions](https://github.com/kase1111-hash/Agent-OS/discussions)

### GitHub Issues

For bugs and feature requests:

- **Bug Reports**: Report problems with Agent-OS
- **Feature Requests**: Suggest new features
- **Documentation Issues**: Report documentation problems

Visit: [GitHub Issues](https://github.com/kase1111-hash/Agent-OS/issues)

### Before Opening an Issue

1. **Search existing issues** to avoid duplicates
2. **Check the FAQ** for common questions
3. **Review the troubleshooting guide** for known solutions
4. **Gather information**:
   - Agent-OS version
   - Python version (`python --version`)
   - Operating system
   - Ollama version (`ollama --version`)
   - Error messages and logs
   - Steps to reproduce

### Issue Templates

When opening an issue, use the appropriate template:

**Bug Report**:
```markdown
### Description
A clear description of the bug.

### Steps to Reproduce
1. Step one
2. Step two
3. Step three

### Expected Behavior
What should happen.

### Actual Behavior
What actually happens.

### Environment
- Agent-OS version:
- Python version:
- OS:
- Ollama version:

### Logs/Screenshots
Relevant logs or screenshots.
```

**Feature Request**:
```markdown
### Problem Statement
What problem does this solve?

### Proposed Solution
How would you like it to work?

### Alternatives Considered
Other approaches you've thought about.

### Additional Context
Any other relevant information.
```

## Community Guidelines

When seeking or providing support:

- **Be respectful**: Treat everyone with respect
- **Be patient**: Contributors are volunteers
- **Be specific**: Provide detailed information
- **Be helpful**: Share solutions when you find them
- **Follow the Code of Conduct**: See [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)

## Contributing Back

Found a solution to your problem? Help others:

1. **Update documentation** if something was unclear
2. **Answer questions** in Discussions
3. **Share your experience** in Show and Tell
4. **Submit a PR** for bug fixes or improvements

See [CONTRIBUTING.md](./CONTRIBUTING.md) for contribution guidelines.

## Security Issues

**Do NOT report security vulnerabilities through public channels.**

See [SECURITY.md](./SECURITY.md) for responsible disclosure procedures.

## Commercial Support

Agent-OS is released under CC0 (Public Domain). There is currently no commercial support offering.

For organizations needing professional support, consider:

- Consulting with the community for complex deployments
- Contributing to the project to prioritize features you need
- Building internal expertise through the documentation

## Roadmap and Future Plans

For information about planned features and development timeline:

- [ROADMAP.md](./ROADMAP.md) - Development phases through 2028
- [TODO.md](./TODO.md) - Current development backlog
- [CHANGELOG.md](./CHANGELOG.md) - Version history

## Communication Channels (Future)

The following channels are planned for Phase 2 (Q3-Q4 2026):

- Discord server for real-time chat
- Mailing list for announcements
- Regular community calls

Until then, GitHub Discussions and Issues are the primary support channels.

## Self-Help Resources

### Quick Diagnostics

```bash
# Check if Agent-OS is running
curl http://localhost:8080/health

# Check agent status
curl http://localhost:8080/api/agents

# Check Ollama status
curl http://localhost:11434/api/tags
```

### Common Fixes

1. **Restart Agent-OS**: Many issues resolve with a restart
2. **Update dependencies**: `pip install -r requirements.txt --upgrade`
3. **Check Ollama**: Ensure Ollama is running with a model loaded
4. **Review logs**: Enable debug logging for more information

### Useful Commands

```bash
# Run with debug logging
LOG_LEVEL=DEBUG python -m uvicorn src.web.app:get_app --factory

# Run tests
pytest tests/

# Check code formatting
black --check src/

# Type checking
mypy src/
```

---

**Remember**: The best support often comes from well-documented issues and questions. Take time to provide context and details, and you'll get better help faster.
