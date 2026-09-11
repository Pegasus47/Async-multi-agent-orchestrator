# Async Multi-Agent Orchestrator POC

An asynchronous, heterogeneous multi-agent orchestration platform designed for studying and comparing free-tier models from OpenRouter.

```
                  ┌────────────────────────────────────────┐
                  │    Helm Orchestrator (Manager AI)      │
                  │   Decomposes Task into N Subtasks      │
                  └──────────────────┬─────────────────────┘
                                     │
                 ┌───────────────────┼───────────────────┐
                 ▼                   ▼                   ▼
        ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
        │ Worker Agent 1  │ │ Worker Agent 2  │ │ Worker Agent N  │
        │ Subtask 1       │ │ Subtask 2       │ │ Subtask N       │
        └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
                 │                   │                   │
                 └───────────────────┼───────────────────┘
                                     ▼
                  ┌────────────────────────────────────────┐
                  │   Workspace Assembler & Telemetry      │
                  │   Time, Tokens, Concurrency Speedup    │
                  └──────────────────┬─────────────────────┘
                                     ▼
                  ┌────────────────────────────────────────┐
                  │   React + shadcn/ui Dashboard          │
                  │   MAIN (Studio & Preview) | HISTORY    │
                  └────────────────────────────────────────┘
```

---

## Features

- **100% Live OpenRouter Execution**: Direct integration with OpenRouter chat completions API using active free models.
- **Customizable Number of Agents ($N=2..6$)**: Configure how many subagents execute in parallel; the Helm orchestrator dynamically divides the project and sets strict interface contracts.
- **Concurrent Worker Execution**: Worker models execute simultaneously via `asyncio.gather()`, tracking individual latencies, token counts, and concurrency speedup factors.
- **Dynamic OpenRouter Free Models**: Discovers active `:free` models live from OpenRouter (Gemma, Nemotron, Cohere, etc.), with automatic token-bucket concurrency throttling and fallback handling.
- **Minimalist React + shadcn/ui Monochromatic Frontend**:
  - **MAIN Tab**: Task specification input, preset chips, agent counter selector (2 to 6 agents), model assignment drawer, live execution pipeline, and a split screen with a tabbed Code Viewer and Live Sandboxed Iframe.
  - **HISTORY Tab**: Detailed billing & token usage breakdown, commercial equivalent savings estimate, per-model consumption table, and a run ledger with one-click inspection.
- **Headless CLI Runner**: Run builds directly from the terminal via `run_cli.py`.

---

## Quickstart

### 1. Installation

```bash
# Create virtual environment and activate
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure OpenRouter API Key

Either export the environment variable:
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```
Or enter it directly in the frontend UI by clicking the Key icon.

### 3. Launch the Web Dashboard

```bash
python3 -m server.app
# Server runs at http://127.0.0.1:8000
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

### 4. Run via Command Line Interface (CLI)

```bash
# Run with 3 agents
python3 run_cli.py --task "build a responsive coffee shop landing page"

# Run with 4 agents
python3 run_cli.py --task "build a mini kanban board" --num-agents 4

# List all active free models available on OpenRouter
python3 run_cli.py --list-free-models
```

### 5. Running Tests

```bash
python3 -m unittest discover -s tests -v
```

---

## Project Structure

```
.
├── core/
│   ├── models.py             # Dataclasses (HelmPlan, AgentResult, OrchestrationReport)
│   ├── openrouter_client.py  # OpenRouter API client, free model discovery & rate limiting
│   └── orchestrator.py       # Core orchestration engine, Helm planning & async worker dispatch
├── server/
│   ├── app.py                # FastAPI server, REST API, WebSocket telemetry & file preview
│   └── db.py                 # SQLite persistence & usage billing ledger
├── frontend/
│   ├── src/                  # React application source (App.jsx, main.jsx, index.css)
│   ├── dist/                 # Compiled production bundle
│   ├── package.json          # Frontend dependencies (React, Tailwind, Lucide)
│   └── vite.config.js        # Vite build & proxy config
├── workspaces/               # Assembled code artifacts per run (run_<timestamp>_<id>/)
├── tests/                    # Automated test suite
├── run_cli.py                # Headless command-line runner
└── requirements.txt          # Python dependencies
```
