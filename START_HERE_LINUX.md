# Agent-OS Quick Start Guide (Linux & macOS)

Welcome! This guide will help you get Agent-OS running on your Linux or macOS computer in just a few minutes.

---

## What You'll Need

Before starting, make sure you have:

1. **Python 3.10 or newer**
2. **Ollama** (for AI chat) - [Download here](https://ollama.com/download)

### Installing Python

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
```

**Fedora/RHEL:**
```bash
sudo dnf install python3 python3-pip
```

**macOS (with Homebrew):**
```bash
brew install python
```

**Verify installation:**
```bash
python3 --version
```

---

## Step 1: Install Agent-OS

1. **Clone or download** this repository:
   ```bash
   git clone https://github.com/kase1111-hash/Agent-OS.git
   cd Agent-OS
   ```

2. **Run the build script:**
   ```bash
   ./build.sh
   ```

That's it! The script will automatically:
- Create a virtual environment
- Install all required packages
- Set up configuration files

> **Note:** First-time setup downloads ~4GB of AI libraries. This may take 10-20 minutes depending on your internet speed.

---

## Step 2: Set Up Ollama

### Linux

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a language model
ollama pull mistral

# Start Ollama (if not already running)
ollama serve
```

### macOS

```bash
# Install via Homebrew
brew install ollama

# Or download from: https://ollama.com/download

# Pull a language model
ollama pull mistral

# Start Ollama (if not already running)
ollama serve
```

> **Tip:** Ollama usually starts automatically after installation.

---

## Step 3: Run Agent-OS

1. **Run the start script:**
   ```bash
   ./start.sh
   ```

2. Wait for the message: `Uvicorn running on http://0.0.0.0:8080`

3. **Open your web browser** and go to:
   ```
   http://localhost:8080
   ```

**You're done!** Agent-OS is now running.

---

## Daily Usage

After the first-time setup, you only need to:

1. Navigate to the Agent-OS directory
2. Run `./start.sh`
3. Open http://localhost:8080 in your browser

---

## Running as a Service (Optional)

### Linux (systemd)

Create a service file:
```bash
sudo nano /etc/systemd/system/agentos.service
```

Add this content (adjust paths as needed):
```ini
[Unit]
Description=Agent-OS AI Assistant
After=network.target ollama.service

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/Agent-OS
ExecStart=/path/to/Agent-OS/venv/bin/python -m uvicorn src.web.app:app --host 0.0.0.0 --port 8080
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable agentos
sudo systemctl start agentos
```

### macOS (launchd)

Create a plist file:
```bash
nano ~/Library/LaunchAgents/com.agentos.plist
```

Add this content (adjust paths as needed):
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.agentos</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/Agent-OS/venv/bin/python</string>
        <string>-m</string>
        <string>uvicorn</string>
        <string>src.web.app:app</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>8080</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/Agent-OS</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

Load the service:
```bash
launchctl load ~/Library/LaunchAgents/com.agentos.plist
```

---

## Troubleshooting

### "Permission denied" when running scripts
```bash
chmod +x build.sh start.sh
```

### "Python is not installed" or wrong version
- Check your Python version: `python3 --version`
- Install Python 3.10+ using your package manager (see above)

### "Ollama not detected"
- Make sure Ollama is installed: https://ollama.com
- Start Ollama: `ollama serve`
- Pull a model: `ollama pull mistral`

### "Network access restricted" in chat
- Make sure Ollama is running: `ollama serve`
- Check available models: `ollama list`

### Build fails with pip errors
```bash
# Upgrade pip and try again
pip install --upgrade pip
pip install -r requirements.txt
```

### Port 8080 already in use
Edit `.env` and change the port:
```bash
AGENT_OS_WEB_PORT=8081
```

Or use a different port directly:
```bash
python -m uvicorn src.web.app:app --host 0.0.0.0 --port 8081
```

### SSL/Certificate errors during pip install
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

---

## Docker Alternative

If you prefer Docker:

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f agentos

# Stop
docker-compose down
```

> **Note:** Set `GRAFANA_ADMIN_PASSWORD` in your environment before running Docker Compose.

---

## Stopping Agent-OS

- Press `Ctrl+C` in the terminal where Agent-OS is running
- Or close the terminal window
- If running as a service:
  - Linux: `sudo systemctl stop agentos`
  - macOS: `launchctl unload ~/Library/LaunchAgents/com.agentos.plist`

---

## Features

Once running, you can:

- **Chat** with the AI assistant
- **Generate images** (requires GPU for best performance)
- **Voice input** (if microphone is available)
- **Manage conversations** and export them

---

## Getting Help

- Check the [full documentation](docs/README.md)
- Report issues at: https://github.com/kase1111-hash/Agent-OS/issues

---

Happy chatting!
