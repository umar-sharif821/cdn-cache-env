"""
FastAPI server exposing OpenEnv interface over HTTP.
Endpoints: POST /reset, POST /step, GET /state, GET /health, GET /tasks
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import uvicorn

from env.cache import CDNCacheEnv, TASK_CONFIGS
from env.models import Action, StepResult

app = FastAPI(
    title="CDN Cache Optimizer - OpenEnv",
    description=(
        "RL environment simulating edge CDN cache management. "
        "Agent decides which files to evict when cache is full. "
        "Implements full OpenEnv spec."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global env instance (stateful per session)
_env: Optional[CDNCacheEnv] = None


class ResetRequest(BaseModel):
    task_id: str = "task_easy"
    seed: int = 42


class StepRequest(BaseModel):
    evict_file_id: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "env": "cdn-cache-optimizer"}


@app.get("/tasks")
def list_tasks():
    return {
        task_id: {
            "name": cfg.name,
            "difficulty": cfg.difficulty,
            "description": cfg.description,
            "cache_capacity_mb": cfg.cache_capacity_mb,
            "episode_length": cfg.episode_length,
        }
        for task_id, cfg in TASK_CONFIGS.items()
    }


@app.post("/reset")
def reset(req: ResetRequest):
    global _env
    if req.task_id not in TASK_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{req.task_id}'. Valid: {list(TASK_CONFIGS.keys())}"
        )
    _env = CDNCacheEnv(task_id=req.task_id, seed=req.seed)
    obs = _env.reset()
    return {"observation": obs.dict(), "task": _env.config.dict()}


@app.post("/step")
def step(req: StepRequest):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    if _env._done:
        raise HTTPException(status_code=400, detail="Episode done. Call /reset.")

    action = Action(evict_file_id=req.evict_file_id)
    result: StepResult = _env.step(action)
    return result.dict()


@app.get("/state")
def state():
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return _env.state()


@app.get("/")
def root():
    return {
        "name": "CDN Cache Optimizer",
        "spec": "OpenEnv v1",
        "endpoints": ["/reset", "/step", "/state", "/health", "/tasks"],
        "tasks": list(TASK_CONFIGS.keys()),
    }


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=7860, reload=False)
