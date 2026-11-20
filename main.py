"""
main.py — FastAPI application entry point.

Routes:
    GET  /                          → Renders the web UI
    POST /generate_report           → Kicks off the research workflow in background
    GET  /task_status/{task_id}     → Returns current status of a task
    GET  /task_progress/{task_id}   → Returns progress log for a task
"""

import uuid
import threading

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.planning_agent import run_research_workflow

app = FastAPI(title="Research Agent")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory task store (swap for Postgres in production)
tasks: dict[str, dict] = {}


# ── Routes ──────────────────────────────────────────────────────────

@app.get("/")
async def index(request: Request):
    """Render the main UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate_report")
async def generate_report(request: Request):
    """Start a research task in the background and return a task ID."""
    form = await request.form()
    topic = form.get("topic", "")

    if not topic:
        return JSONResponse({"error": "topic is required"}, status_code=400)

    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "started", "topic": topic, "progress": [], "result": None}

    thread = threading.Thread(
        target=_run_task,
        args=(task_id, str(topic)),
        daemon=True,
    )
    thread.start()

    return {"task_id": task_id, "status": "started"}


@app.get("/task_status/{task_id}")
async def task_status(task_id: str):
    """Check task status."""
    task = tasks.get(task_id)
    if not task:
        return JSONResponse({"error": "task not found"}, status_code=404)
    return {"task_id": task_id, "status": task["status"]}


@app.get("/task_progress/{task_id}")
async def task_progress(task_id: str):
    """Get detailed progress for a task."""
    task = tasks.get(task_id)
    if not task:
        return JSONResponse({"error": "task not found"}, status_code=404)
    return {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "result": task["result"],
    }


# ── Background runner ───────────────────────────────────────────────

def _run_task(task_id: str, topic: str) -> None:
    """Execute the full research workflow and update the task store."""
    task = tasks[task_id]
    try:
        task["progress"].append("Planning research…")
        task["status"] = "running"

        result = run_research_workflow(topic)

        task["result"] = result
        task["status"] = "completed"
        task["progress"].append("Done.")
    except Exception as exc:
        task["status"] = "failed"
        task["progress"].append(f"Error: {exc}")
