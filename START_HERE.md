# Agent-OS Quick Start Guide (Windows)

Welcome! This guide will help you get Agent-OS running on your Windows computer in just a few minutes.

---

## What You'll Need

Before starting, make sure you have:

1. **Python 3.10 or newer** - [Download here](https://www.python.org/downloads/)
   - During installation, check ✅ "Add Python to PATH"

2. **Ollama** (for AI chat) - [Download here](https://ollama.com/download)

---

## Step 1: Install Agent-OS

1. **Download** or clone this repository to your computer
2. **Open the folder** where you saved it
3. **Double-click `build.bat`**

That's it! The script will automatically:
- Create a virtual environment
- Install all required packages
- Set up configuration files

⏳ **Note:** First-time setup downloads ~4GB of AI libraries. This may take 10-20 minutes depending on your internet speed.

---

## Step 2: Set Up Ollama

1. **Open a new terminal** (Command Prompt or PowerShell)
2. **Pull a language model:**
   ```
   ollama pull mistral
   ```
3. **Start Ollama** (if not already running):
   ```
   ollama serve
   ```

💡 **Tip:** Ollama usually starts automatically after installation.

---

## Step 3: Run Agent-OS

1. **Double-click `start.bat`**
2. Wait for the message: `Uvicorn running on http://0.0.0.0:8080`
3. **Open your web browser** and go to:
   ```
   http://localhost:8080
   ```

🎉 **You're done!** Agent-OS is now running.

---

## Daily Usage

After the first-time setup, you only need to:

1. Double-click `start.bat`
2. Open http://localhost:8080 in your browser

---

## Troubleshooting

### "Python is not installed"
- Download Python from https://python.org
- **Important:** Check "Add Python to PATH" during installation
- Restart your computer after installing

### "Ollama not detected"
- Make sure Ollama is installed: https://ollama.com
- Open a terminal and run: `ollama serve`
- Pull a model: `ollama pull mistral`

### "Network access restricted" in chat
- Make sure Ollama is running (`ollama serve`)
- Check that you pulled a model (`ollama list`)

### Build fails or freezes
- Check your internet connection
- Try running `build.bat` again
- If it still fails, open Command Prompt and run:
  ```
  pip install -r requirements.txt
  ```

### Port 8080 already in use
- Another program is using port 8080
- Close that program, or edit `start.bat` to use a different port

---

## Stopping Agent-OS

- Press `Ctrl+C` in the terminal window where Agent-OS is running
- Or simply close the terminal window

---

## Features

Once running, you can:

- 💬 **Chat** with the AI assistant
- 🖼️ **Generate images** (requires GPU for best performance)
- 🎤 **Voice input** (if microphone is available)
- 📁 **Manage conversations** and export them

---

## Getting Help

- Check the [full documentation](docs/README.md)
- Report issues at: https://github.com/kase1111-hash/Agent-OS/issues

---

Happy chatting! 🚀
