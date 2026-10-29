#!/usr/bin/env python3
"""
CLI for the Async Multi-Agent Orchestrator.

On a Sharanga GPU node (inside srun/sbatch) it uses the local Ollama models,
starting `ollama serve` itself if needed. Results go to workspaces/<run_id>/.

  python3 run_cli.py                                   # interactive (needs srun --pty)
  python3 run_cli.py --task "build a kanban board"     # defaults: largest model plans, 3 workers
  python3 run_cli.py --task "..." --planner qwen2.5:1.5b --workers gemma2:2b,smollm:1.7b
  python3 run_cli.py --list-models
  python3 run_cli.py --pull <user>@hpc.bits-hyderabad.ac.in:<repo path>   # on your laptop
"""

import argparse
import asyncio
import getpass
import glob
import json
import os
import subprocess
import sys
import threading
import time
from typing import List, Optional

from core import local_ollama as ol
from core.openrouter_client import OLLAMA_PREFIX, fetch_free_models
from core.orchestrator import run_orchestration
from server.db import get_run_by_id, init_db, save_run

HPC_LOGIN_HOST = "hpc.bits-hyderabad.ac.in"
MIN_AGENTS, MAX_AGENTS = 2, 8
LINE = "-" * 65
HEARTBEAT_S = 30  # print "still waiting" after this many silent seconds


def fail(msg: str) -> None:
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ----------------------------------------------------------------------------
# Ollama session
# ----------------------------------------------------------------------------

class LocalModels:
    """Ensures Ollama is running, and knows which models are pulled."""

    def __init__(self) -> None:
        self.url = ol.server_url()
        self.proc = None
        self.log_path = os.path.join("logs", f"ollama_serve.{os.environ.get('SLURM_JOB_ID', 'local')}.log")
        self.models: List[dict] = []

    def start(self) -> None:
        os.environ["OLLAMA_BASE_URL"] = self.url  # read by the model client
        if ol.is_up(self.url):
            print(f"Ollama: already running at {self.url}")
        else:
            if ol.on_login_node():
                fail("This is the login node, where models must not run.\n"
                     "Get a GPU node first, e.g.:\n"
                     "  srun -p gpu_rtx_pro_6000_6_csis_hyd -c 8 --mem 64G "
                     "--gres=gpu:nvidia_rtx_pro_6000:1 -t 0-4:00 --pty bash -i")
            print(f"Ollama: starting at {self.url} (GPU discovery takes ~25 s) ...", flush=True)
            self.proc = ol.start_server(self.url, self.log_path)
            device = ol.compute_device(self.log_path)
            if device and device.lower() == "cpu":
                print(f"  WARNING: Ollama is using the CPU, so replies will be slow. See {self.log_path}")
            elif device:
                print(f"  Using {device}. Log: {self.log_path}")

        self.models = ol.list_models(self.url)
        if not self.models:
            fail("Ollama has no models. Check OLLAMA_MODELS points at the folder holding blobs/ and "
                 "manifests/ (your notes use ~/openSourceModels/models), or pull one: ollama pull qwen2.5:1.5b")

    def stop(self) -> None:
        if self.proc:
            print("Stopping the Ollama server this CLI started.")
            ol.stop_server(self.proc)
            self.proc = None

    @property
    def names(self) -> List[str]:
        return [m["name"] for m in self.models]

    def resolve(self, name: str) -> str:
        """User-typed name or list number -> model ID for the orchestrator."""
        name = name.strip()
        if name.isdigit():
            if 1 <= int(name) <= len(self.models):
                return OLLAMA_PREFIX + self.models[int(name) - 1]["name"]
            raise ValueError(f"Pick a number from 1 to {len(self.models)}.")
        name = name[len(OLLAMA_PREFIX):] if name.startswith(OLLAMA_PREFIX) else name
        if name in self.names:
            return OLLAMA_PREFIX + name
        matches = [n for n in self.names if n.split(":")[0] == name]  # "gemma2" -> "gemma2:2b"
        if len(matches) == 1:
            return OLLAMA_PREFIX + matches[0]
        raise ValueError(f"'{name}' is not a pulled model. Available: {', '.join(self.names)}. "
                         f"To add one: ollama pull {name}")

    def print_table(self) -> None:
        for i, m in enumerate(self.models, 1):
            print(f"  [{i}] {m['name']:<28} {m['params']:>6}  {m['size_gb']:>5} GB")


# ----------------------------------------------------------------------------
# Choosing the task and models
# ----------------------------------------------------------------------------

def default_workers(local: LocalModels, n: int) -> List[str]:
    return [OLLAMA_PREFIX + local.names[i % len(local.names)] for i in range(n)]


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer or default


def ask_interactively(local: LocalModels):
    print(LINE)
    task = ""
    while not task:
        task = ask("Task (what should the agents build?)")

    print("\nAvailable models:")
    local.print_table()

    while True:
        try:
            planner = local.resolve(ask("\nPlanner model (number or name)", "1"))
            break
        except ValueError as e:
            print(f"  {e}")

    while True:
        try:
            n = int(ask(f"Number of worker agents ({MIN_AGENTS}-{MAX_AGENTS})", "3"))
            if MIN_AGENTS <= n <= MAX_AGENTS:
                break
        except ValueError:
            pass
        print(f"  Enter a number from {MIN_AGENTS} to {MAX_AGENTS}.")

    default = ",".join(str(local.names.index(w[len(OLLAMA_PREFIX):]) + 1) for w in default_workers(local, n))
    while True:
        raw = ask(f"Worker models, {n} comma-separated (numbers or names)", default)
        try:
            workers = [local.resolve(x) for x in raw.split(",") if x.strip()]
            if len(workers) == n:
                break
            print(f"  Give exactly {n} models.")
        except ValueError as e:
            print(f"  {e}")
    return task, planner, workers


def from_flags(args, local: Optional[LocalModels]):
    """Planner and workers from flags. Without local models, names are OpenRouter IDs."""
    resolve = local.resolve if local else (lambda x: x.strip())
    try:
        planner = resolve(args.planner) if args.planner else (OLLAMA_PREFIX + local.names[0] if local else None)
        if args.workers:
            workers = [resolve(w) for w in args.workers.split(",") if w.strip()]
        elif local:
            workers = default_workers(local, args.num_agents)
        else:
            fail("--workers is required with --no-local.")
    except ValueError as e:
        fail(str(e))
    if not planner:
        fail("--planner is required with --no-local.")
    if not MIN_AGENTS <= len(workers) <= MAX_AGENTS:
        fail(f"Use {MIN_AGENTS} to {MAX_AGENTS} worker models (got {len(workers)}).")
    return planner, workers


# ----------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------

def short(model_id: str) -> str:
    return model_id[len(OLLAMA_PREFIX):] if model_id.startswith(OLLAMA_PREFIX) else model_id


def warm_up(local: LocalModels, model_ids: List[str]) -> None:
    """Load each model once, so load time isn't counted as agent time."""
    for mid in dict.fromkeys(m for m in model_ids if m.startswith(OLLAMA_PREFIX)):
        print(f"Loading {short(mid)} into memory ...", end=" ", flush=True)
        print(f"{ol.load_model(local.url, short(mid)):.1f}s")


def run_once(task: str, planner: str, workers: List[str], api_key: Optional[str]) -> None:
    started = time.time()
    state = {"last_print": started, "waiting_for": "the planner", "pending": set()}

    def log(msg: str) -> None:
        state["last_print"] = time.time()
        print(f"[{time.time() - started:6.1f}s] {msg}", flush=True)

    def heartbeat(stop: threading.Event) -> None:
        while not stop.wait(5):
            if time.time() - state["last_print"] >= HEARTBEAT_S:
                pending = ", ".join(sorted(state["pending"])) or state["waiting_for"]
                log(f"still waiting for {pending} ...")

    def on_update(event: str, data: dict) -> None:
        if event == "worker_start":
            state["pending"].add(data["agent_id"])
        elif event in ("worker_complete", "worker_error"):
            state["pending"].discard(data["agent_id"])
        if event == "helm_planning_start":
            log(f"Planning with {short(planner)} ...")
        elif event == "helm_planning_complete":
            log(f"Plan: {data.get('plan_summary')}")
            for aid, sub in data.get("subtasks", {}).items():
                print(f"           {aid}: {sub['role']} -> {sub['filename']}")
        elif event == "packaging_start":
            log("Planner is reviewing actual source and packaging requirements ...")
        elif event == "packaging_complete":
            log(f"Packaging: {data['status']}" + (f" ({data['error']})" if data.get('error') else ""))
        elif event == "worker_start":
            log(f"{data['agent_id']} ({short(data['model'])}) writing {data['filename']} ...")
        elif event == "worker_complete":
            log(f"{data['agent_id']} done: {data['filename']} in {data['duration_s']}s "
                f"({data.get('tokens_out', 0)} tokens)")
        elif event == "worker_error":
            log(f"{data['agent_id']} FAILED: {data['error']}")

    print(LINE)
    print(f" Task    : {task}")
    print(f" Planner : {short(planner)}")
    print(f" Workers : {', '.join(short(w) for w in workers)}")
    print(LINE)

    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(stop,), daemon=True).start()
    try:
        report = asyncio.run(run_orchestration(
            task=task,
            num_agents=len(workers),
            helm_model=planner,
            worker_models={f"agent_{i + 1}": w for i, w in enumerate(workers)},
            api_key=api_key,
            on_update=on_update,
        ))
    except ValueError as e:  # the planner's reply wasn't valid JSON
        print(f"\nPlanning failed: {e}")
        print("Small models often can't produce the JSON plan. Try a larger planner model.")
        return
    except Exception as e:
        print(f"\nRun failed: {e}")
        return
    finally:
        stop.set()

    record = report.to_dict()
    save_run(record)
    # run_orchestration atomically writes run.json for both CLI and dashboard runs.

    failed = [a["agent_id"] for a in record["agents"] if a.get("error")]
    print(LINE)
    print(f" Wall clock      : {report.wall_clock_s}s   (sum of agent times {report.sum_of_agent_times_s}s, "
          f"speedup {report.concurrency_speedup}x)")
    print(f" Tokens in / out : {report.total_tokens_in} / {report.total_tokens_out}")
    print(f" Files           : {', '.join(report.files_written) or 'none'}")
    if failed:
        print(f" Failed agents   : {', '.join(failed)}")
    print(f" Packaging       : {report.packaging.get('status', 'unknown')}")
    if report.packaging.get('error'):
        print(f" Packaging error : {report.packaging['error']}")
    print(f" Runtime / entry : {report.project.get('runtime', '?')} / {report.project.get('entry_point', '?')}")
    print(" Startup / tests : not run (review SETUP.md and scripts on the laptop)")
    print(f" Output folder   : {report.output_dir}")
    print(LINE)
    print(" To view it in the website, run this on your laptop (in the repo folder):")
    print(f"   python3 run_cli.py --pull {getpass.getuser()}@{HPC_LOGIN_HOST}:{os.getcwd()}")
    print(LINE)


# ----------------------------------------------------------------------------
# Laptop side: fetch runs from the HPC
# ----------------------------------------------------------------------------

def pull(source: str) -> None:
    """rsync workspaces/ from the HPC repo, then add new runs to the local History."""
    init_db()
    remote = source.rstrip("/") + "/workspaces/"
    print(f"Fetching {remote} ...")
    os.makedirs("workspaces", exist_ok=True)
    if subprocess.call(["rsync", "-az", remote, "workspaces/"]) != 0:
        fail("rsync failed (see the message above). Check the user, host and repo path.")

    added = present = 0
    for path in sorted(glob.glob(os.path.join("workspaces", "*", "run.json"))):
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        if get_run_by_id(record["run_id"]):
            present += 1
        else:
            save_run(record)
            added += 1
    print(f"Added {added} new run(s) to History ({present} already there).")
    print("Start the website with: python3 -m server.app")


# ----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Run the multi-agent orchestrator with local Ollama models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run with no arguments for interactive mode.",
    )
    p.add_argument("--task", help="what the agents should build")
    p.add_argument("--planner", "--helm-model", dest="planner",
                   help="planner model (default: the largest pulled model)")
    p.add_argument("--workers", help="comma-separated worker models, one per agent")
    p.add_argument("--num-agents", type=int, default=3,
                   help=f"number of workers when --workers isn't given ({MIN_AGENTS}-{MAX_AGENTS}, default 3)")
    p.add_argument("--max-tokens", type=int,
                   help="cap tokens per model reply (useful on CPU, e.g. 800)")
    p.add_argument("--timeout", type=int, default=600,
                   help="seconds to wait for one local model reply (default 600)")
    p.add_argument("--list-models", action="store_true", help="list the pulled Ollama models and exit")
    p.add_argument("--pull", metavar="USER@HOST:REPO_PATH",
                   help="(laptop) fetch HPC runs into workspaces/ and History")
    p.add_argument("--no-local", action="store_true", help="don't use Ollama; model names are OpenRouter IDs")
    p.add_argument("--api-key", help="OpenRouter key, only needed for OpenRouter models")
    p.add_argument("--list-free-models", action="store_true", help="list OpenRouter free models and exit")
    args = p.parse_args()

    if args.pull:
        return pull(args.pull)
    if args.list_free_models:
        for m in asyncio.run(fetch_free_models(api_key=args.api_key)):
            print(f"  {m['id']}  ({m['name']})")
        return

    os.environ["OLLAMA_TIMEOUT_S"] = str(args.timeout)
    if args.max_tokens:
        os.environ["LLM_MAX_TOKENS"] = str(args.max_tokens)

    interactive = not args.task and not args.list_models
    if interactive and not sys.stdin.isatty():
        fail('No --task given and no keyboard (batch job). Use e.g. --task "build a kanban board".')

    init_db()
    local = None if args.no_local else LocalModels()
    try:
        if local:
            local.start()
            print(f"Models: {', '.join(local.names)}")
        if args.list_models:
            local.print_table() if local else fail("--list-models needs Ollama (drop --no-local).")
            return

        while True:
            if interactive:
                task, planner, workers = ask_interactively(local)
            else:
                task = args.task
                planner, workers = from_flags(args, local)

            api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
            if not api_key and not all(m.startswith(OLLAMA_PREFIX) for m in [planner, *workers]):
                fail("An OpenRouter model was chosen but no key is set (OPENROUTER_API_KEY or --api-key).")

            if local:
                warm_up(local, [planner, *workers])
            run_once(task, planner, workers, api_key)

            if not interactive or ask("\nRun another task? (y/n)", "n").lower() != "y":
                break
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
    except ol.OllamaError as e:
        fail(str(e))
    finally:
        if local:
            local.stop()


if __name__ == "__main__":
    main()
