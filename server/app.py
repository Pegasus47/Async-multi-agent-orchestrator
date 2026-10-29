import asyncio
import json
import logging
import os
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal
import aiohttp

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.openrouter_client import fetch_free_models
from core.orchestrator import run_orchestration, DEFAULT_HELM_MODEL
from core.local_ollama import server_url, on_login_node
from server.db import init_db, save_run, get_all_runs, get_run_by_id, get_usage_billing_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("server")

init_db()

app = FastAPI(title="Async Multi-Agent Orchestrator POC", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000", "http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_connections: List[WebSocket] = []

async def broadcast_telemetry(event: str, data: Dict[str, Any]):
    message = json.dumps({"event": event, "data": data})
    disconnected = []
    for ws in active_connections:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in active_connections:
            active_connections.remove(ws)


class OrchestrateRequest(BaseModel):
    task: str
    num_agents: int = Field(default=3, ge=2, le=8)
    helm_model: Optional[str] = None
    worker_models: Optional[Dict[str, str]] = None
    api_key: Optional[str] = None
    provider: Literal['ollama', 'openrouter'] = 'ollama'


@app.get("/api/models/free")
async def get_free_models(api_key: Optional[str] = None):
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    models = await fetch_free_models(key)
    return {"models": models}


@app.get("/api/models")
async def get_models(provider: Literal['ollama', 'openrouter'] = 'ollama'):
    if provider == 'openrouter':
        return {"models": await fetch_free_models(), "provider": provider}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=4)) as session:
            async with session.get(server_url() + '/api/tags') as response:
                response.raise_for_status()
                data = await response.json()
        models = [{"id": "ollama/" + item['name'], "name": item['name']}
                  for item in data.get('models', [])]
        return {"models": models, "provider": provider, "endpoint": server_url()}
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=503, detail="Cannot reach configured Ollama server. Start it on your local machine or authorized compute node; no server was started automatically.") from exc


@app.post("/api/orchestrate")
async def orchestrate_endpoint(req: OrchestrateRequest):
    if not req.task.strip():
        raise HTTPException(status_code=400, detail="Task description cannot be empty.")
    key = None
    helm = req.helm_model
    workers = req.worker_models or {}
    selected = [helm] + [workers.get(f'agent_{i+1}') for i in range(req.num_agents)]
    if any(not model for model in selected):
        raise HTTPException(status_code=400, detail="Select a planner and every worker model explicitly.")
    if req.provider == 'ollama':
        if on_login_node():
            raise HTTPException(status_code=403, detail="Model execution is forbidden on an HPC login node. Use a Slurm compute allocation.")
        if any(not model.startswith('ollama/') for model in selected):
            raise HTTPException(status_code=400, detail="Ollama mode requires Ollama models for every role; no provider fallback is allowed.")
        available = {model['id'] for model in (await get_models('ollama'))['models']}
        missing = set(selected) - available
        if missing:
            raise HTTPException(status_code=400, detail="Ollama models not installed: " + ', '.join(sorted(missing)))
    else:
        if any(model.startswith('ollama/') for model in selected):
            raise HTTPException(status_code=400, detail="OpenRouter mode cannot use Ollama model IDs.")
        key = req.api_key or os.environ.get('OPENROUTER_API_KEY')
        if not key:
            raise HTTPException(status_code=400, detail="OpenRouter mode needs an API key. Ollama mode does not.")

    loop = asyncio.get_event_loop()
    def on_update(event: str, data: Dict[str, Any]):
        loop.create_task(broadcast_telemetry(event, data))

    try:
        report = await run_orchestration(
            task=req.task,
            num_agents=req.num_agents,
            helm_model=helm,
            worker_models={f"agent_{i+1}": workers[f"agent_{i+1}"] for i in range(req.num_agents)},
            api_key=key,
            on_update=on_update,
        )
        report_dict = report.to_dict()
        save_run(report_dict)
        return report_dict
    except Exception as e:
        logger.exception("Orchestration failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/billing")
async def get_billing():
    """Returns total billing breakdown and token usage for the HISTORY tab."""
    return get_usage_billing_report()


@app.get("/api/runs")
async def list_runs(limit: int = 50):
    return {"runs": get_all_runs(limit=limit)}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    run = get_run_by_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if not run.get('project'):
        try:
            manifest = workspace_path(run_id, 'run.json')
            local = json.loads(manifest.read_text())
            if local.get('run_id') == run_id:
                for field in ('project', 'packaging', 'phase_usage'):
                    if field in local:
                        run[field] = local[field]
        except (OSError, ValueError, HTTPException):
            pass
    return run


def workspace_path(run_id: str, filename: str = "") -> Path:
    root = (Path.cwd() / "workspaces").resolve()
    if not run_id or run_id in (".", "..") or "/" in run_id or "\\" in run_id:
        raise HTTPException(status_code=400, detail="Invalid run ID")
    run_dir = root / run_id
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run workspace directory not found")
    target = run_dir / filename
    # Symlinks can expose files belonging to other runs or outside the workspace.
    if any(part.is_symlink() for part in (target, *target.parents) if part != root):
        raise HTTPException(status_code=403, detail="Workspace symlinks are not supported")
    resolved = target.resolve()
    if not resolved.is_relative_to(run_dir.resolve()):
        raise HTTPException(status_code=403, detail="Path outside run workspace")
    return resolved


@app.get("/api/runs/{run_id}/files")
async def get_run_files(run_id: str):
    run_dir = workspace_path(run_id)
    files = {}
    errors = {}
    for path in sorted(run_dir.rglob("*")):
        relative = path.relative_to(run_dir).as_posix()
        try:
            safe = workspace_path(run_id, relative)
            if not safe.is_file():
                continue
            if safe.stat().st_size > 1024 * 1024:
                errors[relative] = "File exceeds the 1 MiB inspector limit"
                continue
            files[relative] = safe.read_text(encoding="utf-8")
        except (OSError, UnicodeError, HTTPException) as exc:
            errors[relative] = getattr(exc, "detail", str(exc))
    return {"run_id": run_id, "files": files, "errors": errors}


@app.get("/api/preview/{run_id}/{filename:path}")
async def preview_file(run_id: str, filename: str):
    path = workspace_path(run_id, filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Preview file not found")
    mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return FileResponse(path, media_type=mime_type, headers={
        # Also isolate pages opened directly, outside the dashboard iframe.
        "Content-Security-Policy": "sandbox allow-scripts allow-modals allow-forms; frame-ancestors 'self'; connect-src 'none'; form-action 'none'",
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "no-store",
    })


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
    except Exception:
        if websocket in active_connections:
            active_connections.remove(websocket)


# Mount frontend dist or static directory
frontend_dir = str(Path(__file__).resolve().parent.parent / "frontend")
frontend_dist = os.path.join(frontend_dir, "dist")

if os.path.exists(frontend_dist) and os.path.exists(os.path.join(frontend_dist, "index.html")):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")
    
    @app.get("/", response_class=HTMLResponse)
    async def serve_root():
        index_file = os.path.join(frontend_dist, "index.html")
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()

    @app.get("/{full_path:path}", response_class=HTMLResponse)
    async def serve_spa(full_path: str):
        if full_path.startswith("api/") or full_path.startswith("ws/"):
            raise HTTPException(status_code=404, detail="Not Found")
        index_file = os.path.join(frontend_dist, "index.html")
        if os.path.exists(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                return f.read()
        raise HTTPException(status_code=404, detail="Page not found")
else:
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
    @app.get("/", response_class=HTMLResponse)
    async def serve_index():
        index_path = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                return f.read()
        return "<h1>Backend Ready. Building React Frontend...</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)
