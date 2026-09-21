import asyncio
import json
import logging
import os
import mimetypes
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.openrouter_client import fetch_free_models
from core.provider_router import fetch_all_models
from core.orchestrator import run_orchestration, DEFAULT_HELM_MODEL
from server.db import init_db, save_run, get_all_runs, get_run_by_id, get_usage_billing_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("server")

init_db()

app = FastAPI(title="Async Multi-Agent Orchestrator POC", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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


@app.get("/api/models/free")
async def get_free_models(api_key: Optional[str] = None):
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    models = await fetch_free_models(key)
    return {"models": models}


@app.get("/api/models/all")
async def get_all_available_models(api_key: Optional[str] = None):
    """Returns models from all providers (OpenRouter + local Ollama)."""
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    models = await fetch_all_models(key)
    return {"models": models}


@app.post("/api/orchestrate")
async def orchestrate_endpoint(req: OrchestrateRequest):
    if not req.task.strip():
        raise HTTPException(status_code=400, detail="Task description cannot be empty.")

    key = req.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise HTTPException(
            status_code=400,
            detail="OpenRouter API key is required. Please set OPENROUTER_API_KEY in environment or configure it via the Key settings in the UI."
        )

    loop = asyncio.get_event_loop()
    def on_update(event: str, data: Dict[str, Any]):
        loop.create_task(broadcast_telemetry(event, data))

    try:
        report = await run_orchestration(
            task=req.task,
            num_agents=req.num_agents,
            helm_model=req.helm_model,
            worker_models=req.worker_models,
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
    return run


@app.get("/api/runs/{run_id}/files")
async def get_run_files(run_id: str):
    run_dir = os.path.join(os.getcwd(), "workspaces", run_id)
    if not os.path.exists(run_dir):
        raise HTTPException(status_code=404, detail="Run workspace directory not found")

    files = {}
    for fname in os.listdir(run_dir):
        fpath = os.path.join(run_dir, fname)
        if os.path.isfile(fpath):
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    files[fname] = f.read()
            except Exception as e:
                files[fname] = f"Error reading file: {e}"
    return {"run_id": run_id, "files": files}


@app.get("/api/preview/{run_id}/{filename:path}")
async def preview_file(run_id: str, filename: str):
    fpath = os.path.join(os.getcwd(), "workspaces", run_id, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail=f"File {filename} not found in workspace")

    mime_type, _ = mimetypes.guess_type(fpath)
    if not mime_type:
        mime_type = "text/plain"

    return FileResponse(fpath, media_type=mime_type)


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
frontend_dist = os.path.join(os.getcwd(), "frontend", "dist")
frontend_dir = os.path.join(os.getcwd(), "frontend")

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
