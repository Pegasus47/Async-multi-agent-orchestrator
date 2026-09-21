import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Dict, Any, List, Optional, Callable

from core.models import SubTaskSpec, HelmPlan, AgentResult, OrchestrationReport
from core.openrouter_client import call_openrouter
from core.provider_router import call_model

logger = logging.getLogger("orchestrator")

DEFAULT_HELM_MODEL = "google/gemma-4-31b-it:free"
DEFAULT_WORKER_MODELS = {
    "agent_1": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "agent_2": "cohere/north-mini-code:free",
    "agent_3": "nvidia/nemotron-3-super-120b-a12b:free",
    "agent_4": "google/gemma-4-26b-a4b-it:free",
    "agent_5": "liquid/lfm-2.5-2.6b:free",
    "agent_6": "openrouter/free",
    # Local Ollama models (~2B parameters)
    "agent_7": "ollama/gemma2:2b",
    "agent_8": "ollama/qwen2.5:1.5b",
    "agent_9": "ollama/smollm:1.7b",
}

def get_helm_system_prompt(num_agents: int) -> str:
    agent_entries = []
    for i in range(num_agents):
        agent_entries.append(
            f'  "agent_{i+1}": {{"role": "<role name>", "filename": "<target filename>", "instructions": "<precise requirements and interface contracts>"}}'
        )
    entries_str = ",\n".join(agent_entries)

    return f"""You are a master engineering orchestrator (Helm AI).
Your mission is to decompose a user's software build task into exactly {num_agents} modular, concurrently executable subtasks.

Requirements:
1. Divide the project into exactly {num_agents} distinct, complementary roles (agent_1 through agent_{num_agents}).
2. Provide exact target filenames for each agent.
3. Define strict interface contracts in the instructions so all agents produce interoperable code (shared IDs, classes, function signatures).
4. Output ONLY valid JSON matching this exact structure without markdown fences or preamble:
{{
  "plan_summary": "<concise sentence explaining architectural decomposition>",
{entries_str}
}}
"""

WORKER_SYSTEM_PROMPT = """You are an expert autonomous software engineer.
You are given a targeted subtask as part of a multi-agent build pipeline.
Provide ONLY the raw source code for your designated file.
DO NOT wrap your code in markdown fences (no ```).
DO NOT include conversational greetings, explanations, or sign-offs.
Output raw, production-quality, functional code only."""


def extract_json_from_text(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
    try:
        return json.loads(cleaned.strip())
    except Exception:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
    raise ValueError(f"Could not parse valid JSON from model response: {text[:200]}...")


async def plan_task(
    task: str,
    helm_model: str,
    num_agents: int,
    api_key: Optional[str] = None,
    on_update: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> HelmPlan:
    if on_update:
        on_update("helm_planning_start", {"helm_model": helm_model, "task": task, "num_agents": num_agents})

    user_prompt = f"Decompose this build task into {num_agents} subagent specifications:\n\nTask: {task}"
    content, usage = await call_model(
        model=helm_model,
        system_prompt=get_helm_system_prompt(num_agents),
        user_prompt=user_prompt,
        api_key=api_key,
        temperature=0.1,
    )
    raw_plan = extract_json_from_text(content)

    subtasks = {}
    for i in range(num_agents):
        aid = f"agent_{i+1}"
        if aid in raw_plan:
            entry = raw_plan[aid]
            subtasks[aid] = SubTaskSpec(
                role=entry.get("role", f"worker_{i+1}"),
                filename=entry.get("filename", f"file_{i+1}.txt"),
                instructions=entry.get("instructions", ""),
            )
        else:
            subtasks[aid] = SubTaskSpec(
                role=f"worker_{i+1}",
                filename=f"file_{i+1}.txt",
                instructions=f"Complete part {i+1} of task: {task}",
            )

    plan = HelmPlan(
        plan_summary=raw_plan.get("plan_summary", f"Decomposition of '{task}' into {num_agents} subtasks"),
        subtasks=subtasks,
    )

    if on_update:
        on_update("helm_planning_complete", {
            "plan_summary": plan.plan_summary,
            "subtasks": {k: {"role": v.role, "filename": v.filename} for k, v in plan.subtasks.items()}
        })

    return plan


async def execute_worker(
    agent_id: str,
    spec: SubTaskSpec,
    model: str,
    task: str,
    api_key: Optional[str] = None,
    on_update: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> AgentResult:
    started = time.time()
    if on_update:
        on_update("worker_start", {"agent_id": agent_id, "role": spec.role, "model": model, "filename": spec.filename})

    try:
        user_msg = (
            f"Overall Project: {task}\n"
            f"Your Role: {spec.role}\n"
            f"Target Filename: {spec.filename}\n"
            f"Detailed Instructions:\n{spec.instructions}\n\n"
            "Remember: Return ONLY the exact, raw file content for your designated file."
        )
        content, usage = await call_model(
            model=model,
            system_prompt=WORKER_SYSTEM_PROMPT,
            user_prompt=user_msg,
            api_key=api_key,
            temperature=0.2,
            max_tokens=3000,
        )
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)
        cost = 0.0  # Free tier

        content = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", content)
        content = re.sub(r"\n```$", "", content)

        finished = time.time()
        res = AgentResult(
            agent_id=agent_id,
            model=model,
            role=spec.role,
            filename=spec.filename,
            instructions=spec.instructions,
            output=content,
            started_at=started,
            finished_at=finished,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_est=cost,
        )

        if on_update:
            on_update("worker_complete", {
                "agent_id": agent_id,
                "duration_s": res.duration_s,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "filename": spec.filename
            })

        return res

    except Exception as e:
        finished = time.time()
        logger.error(f"Error executing {agent_id} ({model}): {e}")
        res = AgentResult(
            agent_id=agent_id,
            model=model,
            role=spec.role,
            filename=spec.filename,
            instructions=spec.instructions,
            output="",
            started_at=started,
            finished_at=finished,
            error=str(e),
        )
        if on_update:
            on_update("worker_error", {"agent_id": agent_id, "error": str(e)})
        return res


async def run_orchestration(
    task: str,
    num_agents: int = 3,
    helm_model: Optional[str] = None,
    worker_models: Optional[Dict[str, str]] = None,
    api_key: Optional[str] = None,
    on_update: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> OrchestrationReport:
    num_agents = max(2, min(num_agents, 8))
    run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    helm = helm_model or DEFAULT_HELM_MODEL
    workers = worker_models or {}

    wall_start = time.time()
    if on_update:
        on_update("run_init", {"run_id": run_id, "task": task, "helm_model": helm, "num_agents": num_agents})

    # 1. Helm Planning Phase
    plan = await plan_task(task, helm, num_agents, api_key, on_update)

    # 2. Parallel Worker Dispatch Phase
    if on_update:
        on_update("dispatch_workers_start", {"count": len(plan.subtasks)})

    worker_coros = []
    for i in range(num_agents):
        aid = f"agent_{i+1}"
        spec = plan.subtasks.get(aid, SubTaskSpec(role=f"worker_{i+1}", filename=f"file_{i+1}.txt", instructions=""))
        fallback_model = list(DEFAULT_WORKER_MODELS.values())[i % len(DEFAULT_WORKER_MODELS)]
        w_model = workers.get(aid, fallback_model)
        worker_coros.append(
            execute_worker(aid, spec, w_model, task, api_key, on_update)
        )

    results: List[AgentResult] = await asyncio.gather(*worker_coros)

    wall_end = time.time()
    wall_clock_s = round(max(wall_end - wall_start, 0.001), 3)

    # 3. Assemble Files into Workspace
    workspace_dir = os.path.join(os.getcwd(), "workspaces", run_id)
    os.makedirs(workspace_dir, exist_ok=True)
    written_files = []

    for r in results:
        if r.output and not r.error:
            filepath = os.path.join(workspace_dir, r.filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(r.output)
            written_files.append(r.filename)

    total_agent_time = sum(r.duration_s for r in results)
    concurrency_speedup = round(total_agent_time / max(wall_clock_s, 0.001), 2)
    total_tokens_in = sum(r.tokens_in or 0 for r in results)
    total_tokens_out = sum(r.tokens_out or 0 for r in results)
    total_cost = sum(r.cost_est or 0.0 for r in results)

    report = OrchestrationReport(
        run_id=run_id,
        task=task,
        helm_model=helm,
        num_agents=num_agents,
        plan_summary=plan.plan_summary,
        wall_clock_s=wall_clock_s,
        sum_of_agent_times_s=round(total_agent_time, 3),
        concurrency_speedup=concurrency_speedup,
        total_tokens_in=total_tokens_in,
        total_tokens_out=total_tokens_out,
        total_cost_est=round(total_cost, 4),
        output_dir=workspace_dir,
        files_written=written_files,
        agents=[
            {
                "agent_id": r.agent_id,
                "role": r.role,
                "model": r.model,
                "filename": r.filename,
                "instructions": r.instructions,
                "duration_s": r.duration_s,
                "tokens_in": r.tokens_in,
                "tokens_out": r.tokens_out,
                "cost_est": r.cost_est,
                "error": r.error,
                "content_preview": (r.output[:120] + "...") if r.output else "",
            }
            for r in results
        ],
    )

    if on_update:
        on_update("run_finished", {
            "run_id": run_id,
            "wall_clock_s": wall_clock_s,
            "speedup": concurrency_speedup,
            "files": written_files,
            "total_tokens": total_tokens_in + total_tokens_out
        })

    return report
