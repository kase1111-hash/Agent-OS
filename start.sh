#!/bin/bash
# Agent-OS Start Script for Linux/macOS
# This script starts the Agent-OS web server

set -e

echo "=========================================="
echo "  Agent-OS Start Script (Linux/macOS)"
echo "=========================================="
echo ""

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found."
    echo "Please run ./build.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate
echo "[OK] Virtual environment activated"

# Check for Ollama
if command -v ollama &> /dev/null; then
    echo "[OK] Ollama detected"

    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "[OK] Ollama is running"
    else
        echo "[WARN] Ollama is installed but not running."
        echo "      Starting Ollama in background..."
        ollama serve > /dev/null 2>&1 &
        sleep 2
    fi
else
    echo "[WARN] Ollama not detected. Install from: https://ollama.com"
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Set defaults
HOST="${AGENT_OS_WEB_HOST:-0.0.0.0}"
PORT="${AGENT_OS_WEB_PORT:-8080}"

echo ""
echo "Starting Agent-OS..."
echo "  URL: http://localhost:$PORT"
echo "  Press Ctrl+C to stop"
echo ""

# Start the server
python -m uvicorn src.web.app:app --host "$HOST" --port "$PORT"
