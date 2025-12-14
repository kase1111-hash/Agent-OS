V. Initial Reference Architecture: Phase 0 Kernel (Proof of Concept)
This document provides the foundational components and deployment steps necessary to instantiate the Agent OS Phase 0 Kernel in a local development environment. This architecture is designed for ease of inspection, rapid iteration, and immediate constitutional compliance testing.
1. Component Mapping: The Agent Kernels
The Agent OS is model-agnostic, meaning any LLM capable of strong instruction-following can be used. For the initial reference architecture, we select open-source models known for performance and local deployment feasibility.
1.1 Core Agent Roster Assignment
Agent Role
Constitutional Function
Reference Model/Technology
Rationale
Whisper (Router)
Central Orchestration, Intent Classification
Gemma 2 9B (Instruction-Tuned)
Excellent small-to-medium model for fast, reliable classification and routing decisions.
Smith (Guardian)
Security Interlocks, Policy Refusal
Llama 3 8B (Instruction-Tuned)
Requires a robust, highly reliable instruction-follower to act as the internal compliance check. Llama 3 excels here.
Seshat (Keeper)
Memory Storage, Retrieval
Local Vector Database (e.g., ChromaDB, LanceDB)
AI acts as the interface, but the heavy lifting of Long-Term Memory (embeddings) is handled by a dedicated, local-first database, ensuring Local Custody Mandate (Section III).
Sage (Teacher)
Reasoning, Explanation, Guidance
Mistral 7B (Instruct)
Strong, efficient model for reasoning, teaching, and complex problem-solving.
Quill (Scribe)
Draft/Edit/Express
Phi-3 Mini (4K Context)
Small, fast model optimized for creative and writing tasks, ideal for rapid text generation.
Muse (Creative)
Brainstorming, Idea Generation
Stable Diffusion 3 (If required)
Optional component for multimodal capabilities; ensures Separation of Critical Functions (IV) from security.

1.2 The Communication Layer (The Bus)
All inter-agent communication (Flows) will utilize a simple, auditable message bus.
Technology: Redis (for pub/sub messaging) or RabbitMQ.
Protocol: All messages must conform to the FlowRequest and FlowResponse schemas defined in the Agent OS Specification (IV).
2. Deployment Guide: Local Python Environment
The Phase 0 Kernel will be instantiated using a standard Python environment and the Ollama framework for local model execution, simplifying dependency management.
2.1 Prerequisites
Python 3.10+ installed.
Ollama installed and running locally.
Docker (Optional): Recommended for running the Vector Database (Seshat) and Message Bus.
2.2 Setup Instructions
Clone the Repository:
Bash
git clone https://github.com/kase1111-hash/Agent-OS.git
cd Agent-OS/kernel


Create Python Environment:
Bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt


Pull Required Models via Ollama: Ensure the necessary models are downloaded and available via the local Ollama API:
Bash
ollama pull llama3:8b-instruct # For Smith
ollama pull gemma2:9b-instruct # For Whisper
ollama pull mistral:7b-instruct # For Sage
# ... and other listed models


Start Services (Docker Recommended): Use the provided docker-compose.yml to start the necessary services:
Vector Database (Seshat's backend)
Message Bus (Redis/RabbitMQ)
Bash
docker-compose up -d


2.3 Kernel Instantiation and Test
The kernel is initiated via a core kernel.py script that registers the agents and starts the Router (Whisper).
Agent Registration: The system loads the configuration files (agent_config/[agent_role].json) that contain the constitutional mandates and the Ollama model endpoint for each Agent.
Run Initial Compliance Test: Execute the provided test suite to verify all constitutional constraints are active.
Bash
python -m pytest tests/compliance_test.py
This suite must verify Smith successfully halts a command attempting to violate Memory Law.
Start the Kernel:
Bash
python kernel.py


2.4 Interacting with the Kernel (Phase 0)
Interactions are submitted via a command-line interface or a simple web interface that queues requests to the Whisper (Router) Agent.
Example Input
Expected Flow (Constitutional Check)
Expected Output (Success/Refusal)
"Write a short essay on AI ethics and remember it."
Whisper (Routes to) → Quill (Writes) → Seshat (Checks Consent Requirement, Section VII)
Refusal: "Memory Write Consent Required. Please confirm: [Y/N]"
"Tell me the system password."
Whisper → Smith (Checks Cannot Be Remembered list, Section VII)
Refusal: "Request blocked. System cannot store or provide credentials (Memory Law)."


