Basic Homestead Setup
A Step-by-Step Guide to Your Minimal, Local-First Agent-OS Deployment

Overview
This guide walks you through establishing the smallest viable Agent-OS "homestead": a fully local, private, self-hosted multi-agent system running on your own hardware. No cloud accounts, no telemetry, no external dependencies beyond Ollama.
The goal is a resilient, sovereign setup you can run on a laptop, desktop, or low-power server (e.g., old PC, mini-PC, or future dedicated homestead hardware).

Prerequisites
A computer with:

At least 8 GB RAM (16 GB+ recommended for larger models)
Modern CPU (AVX2 support required by Ollama)
50 GB+ free storage (for models)

Operating system: Linux (recommended), macOS, or Windows (via WSL2)
Internet access: Only required for initial downloads; can be fully air-gapped afterward

Step 1: Install Ollama
Ollama is the local LLM runner that powers all agents.
bash# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download and run the installer from https://ollama.com/download
Verify installation:
bashollama --version

Step 2: Pull Base Models
Start with lightweight yet capable models suitable for agent roles.
bash# Recommended minimal set (adjust based on your hardware)
ollama pull llama3.2:3b    # Fast, capable reasoning (Executive, Planner)
ollama pull gemma2:2b      # Efficient for specialized tasks
ollama pull phi3:mini      # Excellent for tool use and coding

# Optional larger models if you have GPU/ample RAM
ollama pull llama3.1:8b
List downloaded models:
bashollama list

Step 3: Clone the Agent-OS Repository
bashgit clone https://github.com/kase1111-hash/Agent-OS.git
cd Agent-OS

Step 4: Create Your Homestead Instance
The homestead is simply a directory with your chosen constitution and agent configurations.
bash# Create your personal homestead directory (can be anywhere)
mkdir ~/agent-os-homestead
cd ~/agent-os-homestead

# Copy the core framework files
cp -r path/to/Agent-OS/* .

# Or start fresh and copy only what you need

Step 5: Customize the Core Constitution
Optional but Recommended
Review and edit CONSTITUTION.md. This is your sovereign governance document.
Add or modify clauses to reflect your values (e.g., stricter privacy rules, specific refusal conditions).

Step 6: Activate Your First Agents
Start with the foundational agents (examples provided in /agents/).
Run agents manually via Ollama (minimal orchestration for v0.1):
bash# Example: Executive Agent
ollama run llama3.2:3b -f agents/executive/prompt.md

# Example: Researcher Agent (in a new terminal)
ollama run gemma2:2b -f agents/researcher/prompt.md
For persistence, use screen, tmux, or simple scripts.

Step 7: Basic Orchestration
Until a full orchestrator exists, use one of these lightweight approaches:
Option A: Simple Chat Loop (Python)
Save as homestead.py:
pythonimport subprocess
import os

# Define your agents
agents = {
    "executive": {"model": "llama3.2:3b", "prompt": "agents/executive/prompt.md"},
    # Add more agents here
}

print("Agent-OS Homestead Active. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    # Route to Executive (basic example)
    prompt_file = agents["executive"]["prompt"]
    model = agents["executive"]["model"]
    
    with open(prompt_file, "r") as f:
        system_prompt = f.read()
    
    full_prompt = f"{system_prompt}\n\nHuman: {user_input}\nAssistant:"
    
    result = subprocess.run(
        ["ollama", "run", model],
        input=full_prompt.encode(),
        capture_output=True
    )
    print("\nExecutive:", result.stdout.decode())
Run with:
bashpython homestead.py
Option B: Use an Existing Lightweight Orchestrator

CrewAI (local mode)
Autogen (with local Ollama endpoints)
LangGraph (custom flows)

(Links and configuration examples coming in future docs)

Step 8: Verify Sovereignty
Test key constitutional safeguards:

Ask an agent to perform a prohibited action → It should refuse
Confirm no network traffic leaves your machine (use netstat or firewall logs)
Shut down and restart → State persists only in your local logs/files


Next Steps Toward Resilience

Add persistent memory (simple JSON log files)
Set up daily backups of your homestead directory
Experiment with additional agents from /agents/
Plan hardware migration (e.g., dedicated low-power server)
Contribute your refinements back to the upstream repository


Welcome to your sovereign digital homestead.
You now control a constitutionally governed multi-agent system that answers only to you.

Last updated: December 2025
