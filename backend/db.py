from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "app.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS problems (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                difficulty TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_id TEXT NOT NULL,
                code TEXT NOT NULL,
                passed INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (problem_id) REFERENCES problems(id)
            )
            """
        )


def _extract_metadata(problem_md: Path, fallback_id: str) -> tuple[str, str]:
    title = fallback_id.replace("_", " ").title()
    difficulty = "Medium"

    if not problem_md.exists():
        return title, difficulty

    lines = problem_md.read_text(encoding="utf-8").splitlines()

    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break

    for line in lines:
        if line.lower().startswith("difficulty:"):
            difficulty = line.split(":", 1)[1].strip() or difficulty
            break

    return title, difficulty


def sync_problems_from_filesystem(problems_dir: Path) -> None:
    with _connect() as conn:
        for problem_dir in sorted(problems_dir.iterdir()):
            if not problem_dir.is_dir():
                continue
            problem_id = problem_dir.name
            title, difficulty = _extract_metadata(problem_dir / "problem.md", problem_id)
            conn.execute(
                """
                INSERT INTO problems (id, title, difficulty)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    difficulty=excluded.difficulty
                """,
                (problem_id, title, difficulty),
            )


def list_problems() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title, difficulty FROM problems ORDER BY title ASC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_problem(problem_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, title, difficulty FROM problems WHERE id = ?",
            (problem_id,),
        ).fetchone()
    return dict(row) if row else None


def log_submission(problem_id: str, code: str, passed: bool) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO submissions (problem_id, code, passed, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (problem_id, code, int(passed), timestamp),
        )
