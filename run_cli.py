#!/usr/bin/env python3
"""
CLI Runner for Async Multi-Agent Orchestrator POC.
Runs orchestrator pipeline headless, writes output files to workspace,
and prints formatted telemetry and concurrency speedup statistics.
"""

import argparse
import asyncio
import json
import os
import sys

from core.openrouter_client import fetch_free_models
from core.ollama_client import OLLAMA_MODELS, check_ollama_health
from core.provider_router import fetch_all_models
from core.orchestrator import run_orchestration, DEFAULT_HELM_MODEL, DEFAULT_WORKER_MODELS
from server.db import save_run, init_db

def main():
    parser = argparse.ArgumentParser(description="Async Multi-Agent Orchestrator POC")
    parser.add_argument("--task", default="build a responsive coffee shop landing page with dark theme", help="Objective prompt for the multi-agent system")
    parser.add_argument("--num-agents", type=int, default=3, help="Number of concurrent subagents to dispatch (2 to 8)")
    parser.add_argument("--api-key", default=None, help="OpenRouter API key (defaults to OPENROUTER_API_KEY env var)")
    parser.add_argument("--helm-model", default=DEFAULT_HELM_MODEL, help="Model playing Helm orchestrator")
    parser.add_argument("--list-free-models", action="store_true", help="List active free models discovered from OpenRouter and exit")
    parser.add_argument("--list-ollama-models", action="store_true", help="List locally available Ollama models and exit")
    parser.add_argument("--list-all-models", action="store_true", help="List all models from all providers (OpenRouter + Ollama) and exit")
    args = parser.parse_args()

    init_db()

    if args.list_free_models:
        print("Discovering active free models from OpenRouter...")
        models = asyncio.run(fetch_free_models(api_key=args.api_key))
        print(f"\nFound {len(models)} Free Models:")
        for m in models:
            print(f"  - {m['id']} ({m['name']}) [Context: {m.get('context_length', 'N/A')}]")
        return

    if args.list_ollama_models:
        print("Checking local Ollama server...")
        is_online = asyncio.run(check_ollama_health())
        status = "ONLINE" if is_online else "OFFLINE"
        print(f"Ollama Server: {status}\n")
        print("Configured Ollama Models (~2B parameters):")
        for m in OLLAMA_MODELS:
            print(f"  - {m['id']} ({m['name']}) [Context: {m.get('context_length', 'N/A')}]")
            print(f"    {m['description']}")
        return

    if args.list_all_models:
        print("Discovering models from all providers...")
        models = asyncio.run(fetch_all_models(api_key=args.api_key))
        ollama_models = [m for m in models if m.get("provider") == "ollama"]
        openrouter_models = [m for m in models if m.get("provider") == "openrouter"]
        print(f"\n--- Local Ollama Models ({len(ollama_models)}) ---")
        for m in ollama_models:
            print(f"  - {m['id']} ({m['name']}) [{m.get('status', 'unknown')}]")
        print(f"\n--- OpenRouter Free Models ({len(openrouter_models)}) ---")
        for m in openrouter_models:
            print(f"  - {m['id']} ({m['name']}) [Context: {m.get('context_length', 'N/A')}]")
        return

    key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("ERROR: OpenRouter API key is required. Set OPENROUTER_API_KEY or pass --api-key.", file=sys.stderr)
        sys.exit(1)

    print("=" * 65)
    print(" ASYNC MULTI-AGENT ORCHESTRATOR POC")
    print("=" * 65)
    print(f" Task        : {args.task}")
    print(f" Agents      : {args.num_agents}")
    print(f" Helm Model  : {args.helm_model}")
    print(f" Workers     : {list(DEFAULT_WORKER_MODELS.values())[:args.num_agents]}")
    print("-" * 65)

    def on_update(event: str, data: dict):
        if event == "helm_planning_start":
            print(f"[*] Helm Orchestrator planning decomposition...")
        elif event == "helm_planning_complete":
            print(f"[+] Plan Summary: {data.get('plan_summary')}")
        elif event == "dispatch_workers_start":
            print(f"[*] Dispatched {data.get('count')} subagents concurrently...")
        elif event == "worker_complete":
            print(f"    -> {data.get('agent_id')} finished {data.get('filename')} in {data.get('duration_s')}s")
        elif event == "worker_error":
            print(f"    [!] {data.get('agent_id')} error: {data.get('error')}")

    report = asyncio.run(
        run_orchestration(
            task=args.task,
            num_agents=args.num_agents,
            helm_model=args.helm_model,
            api_key=key,
            on_update=on_update,
        )
    )

    report_dict = report.to_dict()
    save_run(report_dict)

    print("-" * 65)
    print(" TELEMETRY & CONCURRENCY REPORT")
    print("-" * 65)
    print(f" Wall-Clock Duration    : {report.wall_clock_s}s")
    print(f" Sum of Agent Times     : {report.sum_of_agent_times_s}s")
    print(f" Concurrency Speedup    : {report.concurrency_speedup}x")
    print(f" Total Tokens (In/Out)  : {report.total_tokens_in} / {report.total_tokens_out}")
    print(f" Estimated Cost         : ${report.total_cost_est:.4f}")
    print(f" Workspace Output Dir   : {report.output_dir}")
    print(f" Files Assembled        : {', '.join(report.files_written)}")
    print("=" * 65)

if __name__ == "__main__":
    main()
