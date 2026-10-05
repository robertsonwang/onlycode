from __future__ import annotations

from pathlib import Path

PROBLEMS_DIR = Path(__file__).resolve().parent / "problems"

SQL_PLACEHOLDER_TOKEN = '"__SQL_QUERY_PLACEHOLDER__"'


def is_sql_problem(problem_id: str) -> bool:
    return (PROBLEMS_DIR / problem_id / "query_starter.sql").exists()


def build_submission_code(problem_id: str, submitted_code: str) -> str:
    """Return the full Python source to execute for a submission.

    SQL-mode problems only show the user a raw SQL editor, so `submitted_code`
    is just the query text. Splice it into the problem's starter.py harness in
    place of the query placeholder before executing.
    """
    if not is_sql_problem(problem_id):
        return submitted_code

    template_path = PROBLEMS_DIR / problem_id / "starter.py"
    template = template_path.read_text(encoding="utf-8")
    if SQL_PLACEHOLDER_TOKEN not in template:
        return submitted_code
    return template.replace(SQL_PLACEHOLDER_TOKEN, repr(submitted_code), 1)
