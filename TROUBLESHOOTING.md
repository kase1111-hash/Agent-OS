# Troubleshooting Guide

This guide covers common issues and solutions when running Agent-OS.

## Quick Diagnostics

Run these commands to diagnose issues:

```bash
# Check system health
curl http://localhost:8080/health

# Check agent status
curl http://localhost:8080/api/agents

# View recent logs
python -m uvicorn src.web.app:get_app --factory --log-level debug
```

## Installation Issues

### Python Version Error

**Error**: `Python version 3.x is not supported`

**Solution**: Agent-OS requires Python 3.10+. Check your version:
```bash
python --version
# or
python3 --version
```

Install Python 3.10+ from [python.org](https://www.python.org/downloads/).

### Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'xyz'`

**Solution**: Install dependencies:
```bash
pip install -r requirements.txt

# Or for development:
pip install -e ".[dev]"
```

### Permission Denied

**Error**: `Permission denied` when running scripts

**Solution**:
```bash
# Linux/macOS
chmod +x build.sh start.sh

# Windows: Run as Administrator or check antivirus settings
```

## Ollama Issues

### Ollama Not Running

**Error**: `Connection refused` or `Cannot connect to Ollama`

**Solution**:
1. Ensure Ollama is installed: [ollama.com/download](https://ollama.com/download)
2. Start Ollama:
   ```bash
   ollama serve
   ```
3. Verify it's running:
   ```bash
   curl http://localhost:11434/api/tags
   ```

### No Models Available

**Error**: `No models found` or empty model list

**Solution**: Pull a model:
```bash
ollama pull mistral
# or
ollama pull llama2
```

### Model Too Large for Memory

**Error**: `Out of memory` or system freezes

**Solution**:
- Use a smaller model: `ollama pull phi` or `ollama pull tinyllama`
- Close other applications
- Check minimum requirements: 8GB RAM, 16GB+ recommended

## Startup Issues

### Port Already in Use

**Error**: `Address already in use: 8080`

**Solution**:
```bash
# Find and kill the process using the port
# Linux/macOS:
lsof -i :8080
kill -9 <PID>

# Windows:
netstat -ano | findstr :8080
taskkill /PID <PID> /F

# Or use a different port:
python -m uvicorn src.web.app:get_app --factory --port 8081
```

### Database Locked

**Error**: `Database is locked` or `sqlite3.OperationalError`

**Solution**:
1. Stop all Agent-OS processes
2. Remove lock file if present:
   ```bash
   rm -f data/*.db-journal
   ```
3. Restart Agent-OS

### Constitution Parse Error

**Error**: `Failed to parse constitution` or `Invalid YAML`

**Solution**:
1. Validate your constitution file:
   ```bash
   python -c "import yaml; yaml.safe_load(open('CONSTITUTION.md'))"
   ```
2. Check for YAML syntax errors (indentation, colons, quotes)
3. Reset to default constitution if needed

## Runtime Issues

### Agents Not Responding

**Symptom**: Messages sent but no response

**Causes and Solutions**:
1. **Ollama not running**: Start Ollama service
2. **Model not loaded**: Check model availability
3. **Agent crashed**: Check logs for errors
4. **Constitutional violation**: Check Smith agent logs

### Slow Response Times

**Symptom**: Long delays before responses

**Solutions**:
- Use a smaller/faster model
- Increase system resources
- Check for memory pressure: `top` or Task Manager
- Reduce concurrent operations

### Memory Issues

**Error**: `MemoryError` or system becoming unresponsive

**Solutions**:
1. Reduce model size
2. Clear conversation history:
   ```bash
   curl -X DELETE http://localhost:8080/api/memory
   ```
3. Restart Agent-OS to free memory
4. Increase system swap space

### WebSocket Disconnections

**Symptom**: Chat interface disconnects frequently

**Solutions**:
1. Check network stability
2. Increase timeout in configuration
3. Check for proxy/firewall issues
4. Monitor server logs for errors

## Security Issues

### Smith Blocking Requests

**Symptom**: `Constitutional violation` errors

**Explanation**: Smith (Guardian agent) is blocking requests that violate the constitution.

**Solutions**:
1. Review the request against constitutional rules
2. Check `/api/security/attacks` for details
3. Modify request to comply with rules
4. If legitimate, review and update constitution

### Authentication Failures

**Error**: `401 Unauthorized`

**Solutions**:
1. Re-login to get a fresh token
2. Clear browser cookies
3. Check session hasn't expired
4. Verify user exists: `/api/auth/users/count`

## Docker Issues

### Container Won't Start

**Error**: Container exits immediately

**Solution**:
```bash
# Check logs
docker compose logs agent-os

# Ensure .env exists
cp .env.example .env

# Rebuild containers
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Volume Permission Issues

**Error**: Permission denied accessing data volumes

**Solution**:
```bash
# Fix ownership (Linux)
sudo chown -R $(id -u):$(id -g) ./data

# Or run as root (not recommended for production)
docker compose run --user root agent-os
```

### Ollama in Docker Can't Use GPU

**Solution**: Use NVIDIA Container Toolkit:
```bash
# Install nvidia-container-toolkit
sudo apt-get install nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker

# Run with GPU access
docker compose -f docker-compose.gpu.yml up -d
```

## Network Issues

### Can't Access from Other Devices

**Problem**: Can only access from localhost

**Solution**: Bind to all interfaces:
```bash
python -m uvicorn src.web.app:get_app --factory --host 0.0.0.0 --port 8080
```

Check firewall allows port 8080.

### Proxy/Corporate Network Issues

**Problem**: Connections fail behind proxy

**Solutions**:
1. Set proxy environment variables:
   ```bash
   export HTTP_PROXY=http://proxy:port
   export HTTPS_PROXY=http://proxy:port
   ```
2. Configure Ollama proxy settings
3. Whitelist required domains

## Performance Tuning

### Improve Response Speed

1. Use faster models (Mistral, Phi)
2. Enable GPU acceleration in Ollama
3. Increase worker count:
   ```bash
   uvicorn src.web.app:get_app --factory --workers 4
   ```
4. Use SSD storage for data directory

### Reduce Memory Usage

1. Use quantized models (Q4, Q5)
2. Limit conversation history length
3. Configure memory cleanup intervals
4. Reduce concurrent agent count

## Logs and Debugging

### Enable Debug Logging

```bash
# Environment variable
export LOG_LEVEL=DEBUG

# Or command line
python -m uvicorn src.web.app:get_app --factory --log-level debug
```

### View Agent Logs

```bash
# Via API
curl http://localhost:8080/api/agents/whisper/logs

# All agents
curl http://localhost:8080/api/agents/logs/all
```

### Check Constitutional Violations

```bash
curl http://localhost:8080/api/security/attacks
```

## Getting More Help

If these solutions don't resolve your issue:

1. **Check existing issues**: [GitHub Issues](https://github.com/kase1111-hash/Agent-OS/issues)
2. **Search discussions**: [GitHub Discussions](https://github.com/kase1111-hash/Agent-OS/discussions)
3. **Open a new issue** with:
   - Agent-OS version
   - Python version
   - Operating system
   - Error messages/logs
   - Steps to reproduce

See [SUPPORT.md](./SUPPORT.md) for more support options.

---

**Last Updated**: January 2026
