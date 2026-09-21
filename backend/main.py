from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from db import get_problem, init_db, list_problems, log_submission, sync_problems_from_filesystem
from debug_runner import handle_debug_session
from runner import run_submission

BASE_DIR = Path(__file__).resolve().parent
PROBLEMS_DIR = BASE_DIR / "problems"

app = FastAPI(title="LeetCode Practice API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SubmissionRequest(BaseModel):
    problem_id: str
    code: str


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    sync_problems_from_filesystem(PROBLEMS_DIR)


@app.get("/problems")
def get_problems() -> list[dict]:
    return list_problems()


@app.get("/problems/{problem_id}")
def get_problem_details(problem_id: str) -> dict:
    problem = get_problem(problem_id)
    if not problem:
        raise HTTPException(status_code=404, detail="Problem not found")

    problem_dir = PROBLEMS_DIR / problem_id
    md_path = problem_dir / "problem.md"
    starter_path = problem_dir / "starter.py"

    if not md_path.exists() or not starter_path.exists():
        raise HTTPException(status_code=500, detail="Problem assets missing")

    return {
        **problem,
        "problem_markdown": md_path.read_text(encoding="utf-8"),
        "starter_code": starter_path.read_text(encoding="utf-8"),
    }


@app.post("/submit")
def submit_solution(payload: SubmissionRequest) -> dict:
    problem = get_problem(payload.problem_id)
    if not problem:
        raise HTTPException(status_code=404, detail="Problem not found")

    tests_path = PROBLEMS_DIR / payload.problem_id / "tests.json"
    if not tests_path.exists():
        raise HTTPException(status_code=500, detail="tests.json missing for problem")

    try:
        tests = json.loads(tests_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid tests.json: {exc}") from exc

    runner_config = {}
    runner_config_path = PROBLEMS_DIR / payload.problem_id / "runner.json"
    if runner_config_path.exists():
        try:
            runner_config = json.loads(runner_config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail=f"Invalid runner.json: {exc}") from exc

    result = run_submission(payload.code, tests, **runner_config)
    log_submission(payload.problem_id, payload.code, result.get("all_passed", False))

    return {
        "problem_id": payload.problem_id,
        **result,
    }


@app.websocket("/ws/debug")
async def debug_websocket(websocket: WebSocket):
    await handle_debug_session(websocket)
