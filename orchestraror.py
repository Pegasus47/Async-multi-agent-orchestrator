"""
POC v2: LLM Orchestrator dynamically delegating to 3 subagents.

Difference from v1 (orchestrator.py):
  v1 -> orchestrator used a hardcoded split_task() function.
  v2 -> orchestrator is ITSELF a model call. You give it the overall task,
        it decides what each of the 3 subagents should build, in what role,
        and with what instructions. Then subagents run concurrently, and
        their outputs get assembled into real files on disk.

Example task: "build a static site for a coffee shop landing page"
  Orchestrator might decide:
    agent_1 -> role: html_structure   -> writes index.html
    agent_2 -> role: css_styling      -> writes style.css
    agent_3 -> role: js_interactivity -> writes script.js
  (but it's the orchestrator model deciding this breakdown, not us)

Modes:
  MOCK mode (default): orchestrator planning + subagent code generation are
  both simulated with canned-but-realistic outputs. Proves the pipeline
  (plan -> dispatch -> concurrent execution -> assemble -> write files)
  works with zero API cost.

  LIVE mode (--live, needs OPENROUTER_API_KEY): orchestrator and subagents
  are real model calls. Swap ORCHESTRATOR_MODEL / MODEL_MAP to change which
  models play which role -- that's your heterogeneous-agent knob.

Run:
  python orchestrator_llm.py --task "build a static site for a coffee shop landing page"
  OPENROUTER_API_KEY=sk-... python orchestrator_llm.py --live --task "..."
"""

import asyncio
import json
import os
import re
import time
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# The model playing "manager" -- decides the plan
ORCHESTRATOR_MODEL = "google/gemini-2.0-flash-exp:free"

# The models playing "workers" -- pick freely, heterogeneous by design
MODEL_MAP = {
    "agent_1": "meta-llama/llama-3.3-70b-instruct:free",
    "agent_2": "qwen/qwen-2.5-72b-instruct:free",
    "agent_3": "google/gemini-2.0-flash-exp:free",
}

ORCHESTRATOR_SYSTEM_PROMPT = """You are a project manager AI coordinating exactly 3 subagents to complete a small build task.

Break the task into exactly 3 subtasks, one per agent. Each subtask should be a concrete, independently-completable piece of work with a clear output filename.

Respond with ONLY valid JSON in this exact shape, no markdown fences, no commentary:
{
  "plan_summary": "<one sentence describing your overall approach>",
  "agent_1": {"role": "<short role name>", "filename": "<output filename>", "instructions": "<precise instructions for this agent, including what the other agents are building so outputs are compatible>"},
  "agent_2": {"role": "<short role name>", "filename": "<output filename>", "instructions": "<...>"},
  "agent_3": {"role": "<short role name>", "filename": "<output filename>", "instructions": "<...>"}
}
"""

SUBAGENT_SYSTEM_PROMPT = """You are a specialist subagent. You will be given precise instructions for one piece of a larger build.
Output ONLY the raw file content requested -- no markdown fences, no explanation, no commentary before or after."""


@dataclass
class AgentResult:
    agent_id: str
    model: str
    role: str
    filename: str
    instructions: str
    output: str
    started_at: float
    finished_at: float
    duration_s: float = field(init=False)
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    error: Optional[str] = None

    def __post_init__(self):
        self.duration_s = round(self.finished_at - self.started_at, 3)


# ---------------------------------------------------------------------------
# Shared: call a model (real or mocked)
# ---------------------------------------------------------------------------
async def call_model(model: str, system: str, user: str, session=None) -> tuple[str, dict]:
    """Returns (content, usage_dict). session=None => mock mode."""
    if session is None:
        await asyncio.sleep(0.3)  # simulate latency
        return "", {}

    async with session.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 1200,
        },
        timeout=90,
    ) as resp:
        data = await resp.json()
        if "error" in data:
            raise RuntimeError(str(data["error"]))
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return content, usage


# ---------------------------------------------------------------------------
# Orchestrator planning step
# ---------------------------------------------------------------------------
def mock_plan(task: str) -> dict:
    """Canned-but-realistic plan for mock mode, shaped like a real LLM would answer."""
    return {
        "plan_summary": f"Build a 3-file static site for: {task}",
        "agent_1": {
            "role": "html_structure",
            "filename": "index.html",
            "instructions": f"Write the HTML structure for: {task}. Link to style.css and script.js. Keep it semantic.",
        },
        "agent_2": {
            "role": "css_styling",
            "filename": "style.css",
            "instructions": f"Write CSS styling to match the theme of: {task}. Clean, modern, responsive.",
        },
        "agent_3": {
            "role": "js_interactivity",
            "filename": "script.js",
            "instructions": f"Write small vanilla JS for basic interactivity fitting: {task} (e.g. a button or nav toggle).",
        },
    }


async def get_plan(task: str, live: bool, session=None) -> dict:
    if not live:
        return mock_plan(task)

    content, _ = await call_model(ORCHESTRATOR_MODEL, ORCHESTRATOR_SYSTEM_PROMPT, task, session)
    # strip stray markdown fences if the model adds them anyway
    cleaned = re.sub(r"^```(json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Mock subagent output generators (realistic-looking, not just placeholder text)
# ---------------------------------------------------------------------------
def mock_subagent_output(role: str, task: str) -> str:
    if "html" in role:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{task.title()}</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header><h1>{task.title()}</h1></header>
  <main><p>Welcome! This page was scaffolded by a mock subagent.</p>
    <button id="cta">Learn more</button>
  </main>
  <script src="script.js"></script>
</body>
</html>"""
    if "css" in role:
        return """body { font-family: system-ui, sans-serif; margin: 0; background: #faf7f2; color: #2b2320; }
header { padding: 2rem; text-align: center; background: #6f4e37; color: #fff; }
main { max-width: 700px; margin: 2rem auto; padding: 0 1rem; }
button { padding: 0.6rem 1.2rem; border: none; border-radius: 6px; background: #6f4e37; color: #fff; cursor: pointer; }"""
    return """document.getElementById('cta')?.addEventListener('click', () => {
  alert('Thanks for checking us out!');
});"""


# ---------------------------------------------------------------------------
# Subagent execution
# ---------------------------------------------------------------------------
async def run_agent(agent_id: str, plan_entry: dict, task: str, live: bool, session=None) -> AgentResult:
    model = MODEL_MAP[agent_id]
    role = plan_entry["role"]
    filename = plan_entry["filename"]
    instructions = plan_entry["instructions"]
    started = time.time()

    try:
        if live:
            content, usage = await call_model(model, SUBAGENT_SYSTEM_PROMPT, instructions, session)
            tokens_in = usage.get("prompt_tokens")
            tokens_out = usage.get("completion_tokens")
        else:
            await asyncio.sleep(__import__("random").uniform(0.5, 1.8))
            content = mock_subagent_output(role, task)
            tokens_in = len(instructions.split())
            tokens_out = len(content.split())

        finished = time.time()
        return AgentResult(
            agent_id=agent_id, model=model + ("" if live else " (mocked)"),
            role=role, filename=filename, instructions=instructions,
            output=content, started_at=started, finished_at=finished,
            tokens_in=tokens_in, tokens_out=tokens_out,
        )
    except Exception as e:
        finished = time.time()
        return AgentResult(
            agent_id=agent_id, model=model, role=role, filename=filename,
            instructions=instructions, output="", started_at=started,
            finished_at=finished, error=str(e),
        )


# ---------------------------------------------------------------------------
# Orchestration end to end
# ---------------------------------------------------------------------------
async def orchestrate(task: str, live: bool, out_dir: str) -> dict:
    wall_start = time.time()

    if live:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            plan = await get_plan(task, live, session)
            results = await asyncio.gather(*[
                run_agent(aid, plan[aid], task, live, session)
                for aid in ["agent_1", "agent_2", "agent_3"]
            ])
    else:
        plan = await get_plan(task, live)
        results = await asyncio.gather(*[
            run_agent(aid, plan[aid], task, live)
            for aid in ["agent_1", "agent_2", "agent_3"]
        ])

    wall_end = time.time()

    # assemble: write each agent's output to its planned filename
    os.makedirs(out_dir, exist_ok=True)
    written_files = []
    for r in results:
        if r.error:
            continue
        path = os.path.join(out_dir, r.filename)
        with open(path, "w") as f:
            f.write(r.output)
        written_files.append(path)

    total_agent_time = sum(r.duration_s for r in results)
    report = {
        "task": task,
        "mode": "live" if live else "mock",
        "plan_summary": plan.get("plan_summary"),
        "wall_clock_s": round(wall_end - wall_start, 3),
        "sum_of_agent_times_s": round(total_agent_time, 3),
        "concurrency_speedup": round(total_agent_time / max(wall_end - wall_start, 1e-6), 2),
        "output_dir": out_dir,
        "files_written": written_files,
        "agents": [asdict(r) for r in results],
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--task", default="build a static site for a coffee shop landing page")
    parser.add_argument("--out", default="site_output")
    args = parser.parse_args()

    if args.live and "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit("Set OPENROUTER_API_KEY env var to run in --live mode")

    report = asyncio.run(orchestrate(args.task, args.live, args.out))

    print(json.dumps({k: v for k, v in report.items() if k != "agents"}, indent=2))
    print("\n--- per-agent summary ---")
    for a in report["agents"]:
        status = "ERROR: " + a["error"] if a["error"] else f'{a["role"]} -> {a["filename"]} ({a["duration_s"]}s)'
        print(f'  {a["agent_id"]}: {status}')

    with open("last_run_report_v2.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
