"""
Helpers for an Ollama server running on the same machine as the CLI
(a Sharanga GPU node inside a Slurm job). Standard library only.

    url = server_url()                 # where Ollama is (or will be)
    proc = start_server(url, log_path) # only if is_up(url) is False
    models = list_models(url)          # [{"name", "size_gb", "params"}, ...]
    load_model(url, name)              # warm a model into GPU memory
    stop_server(proc)
"""

import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

LOGIN_NODES = {"hpc01"}  # never run models here; processes are killed


class OllamaError(RuntimeError):
    """A problem the user can fix; the message says how."""


def on_login_node() -> bool:
    return socket.gethostname().split(".")[0] in LOGIN_NODES


def server_url() -> str:
    """OLLAMA_BASE_URL if set, else a per-job port so jobs on a shared node don't clash."""
    if os.environ.get("OLLAMA_BASE_URL"):
        return os.environ["OLLAMA_BASE_URL"].rstrip("/")
    job_id = os.environ.get("SLURM_JOB_ID")
    port = 11500 + int(job_id) % 1000 if job_id else 11434
    return f"http://127.0.0.1:{port}"


def _get(url: str, timeout: float) -> Dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def is_up(url: str) -> bool:
    try:
        _get(url + "/api/version", timeout=2)
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def list_models(url: str) -> List[Dict]:
    """Pulled models, largest first."""
    data = _get(url + "/api/tags", timeout=10)
    models = [
        {
            "name": m["name"],
            "size_gb": round(m.get("size", 0) / 1e9, 1),
            "params": m.get("details", {}).get("parameter_size", "?"),
        }
        for m in data.get("models", [])
    ]
    return sorted(models, key=lambda m: m["size_gb"], reverse=True)


def find_binary() -> str:
    path = shutil.which("ollama") or os.path.expanduser("~/ollama/bin/ollama")
    if not os.path.exists(path):
        raise OllamaError(
            "Ollama binary not found (looked on PATH and in ~/ollama/bin).\n"
            "Install it as in ollama-on-sharanga.md, section 1, keeping bin/ and lib/ side by side."
        )
    return path


def start_server(url: str, log_path: str, wait_s: int = 120) -> subprocess.Popen:
    """Start `ollama serve` on the URL's port and wait until it answers."""
    host_port = url.split("://", 1)[-1]
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)  # Slurm's value makes GPU discovery time out (Issue #2)
    env["OLLAMA_HOST"] = host_port
    env.setdefault("OLLAMA_KEEP_ALIVE", "30m")
    env.setdefault("OLLAMA_NUM_PARALLEL", "4")

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log = open(log_path, "a")
    proc = subprocess.Popen(
        [find_binary(), "serve"], env=env, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,  # Ctrl-C in the CLI doesn't kill it mid-cleanup
    )

    deadline = time.time() + wait_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise OllamaError(f"`ollama serve` exited with code {proc.returncode}. See {log_path}")
        if is_up(url):
            return proc
        time.sleep(1)
    stop_server(proc)
    raise OllamaError(f"Ollama did not answer within {wait_s}s. See {log_path}")


def compute_device(log_path: str) -> Optional[str]:
    """'CUDA' / 'cpu' from the server log's 'inference compute' line, if present yet."""
    try:
        with open(log_path) as f:
            lines = [l for l in f if "inference compute" in l]
    except OSError:
        return None
    if not lines:
        return None
    for part in lines[-1].split():
        if part.startswith("library="):
            return part.split("=", 1)[1]
    return None


def load_model(url: str, name: str, timeout_s: int = 600) -> float:
    """Load a model into memory (an empty generate request). Returns seconds taken."""
    started = time.time()
    req = urllib.request.Request(
        url + "/api/generate",
        data=json.dumps({"model": name}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        raise OllamaError(f"Could not load {name}: {e.read().decode(errors='replace')}")
    except (urllib.error.URLError, OSError) as e:
        raise OllamaError(f"Could not load {name}: {e}. Is the Ollama server still running?")
    return time.time() - started


def stop_server(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
