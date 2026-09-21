from __future__ import annotations

import json
import resource
import shutil
import subprocess
import sys
from typing import Any

RESULT_MARKER = "__RESULT_JSON__"


def _limit_resources(memory_limit_mb: int) -> None:
    if memory_limit_mb <= 0:
        return
    memory_bytes = memory_limit_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except (ValueError, OSError):
        # RLIMIT_AS is not supported on macOS; skip memory limiting
        pass


def _build_runner_script(code: str, tests: list[dict[str, Any]]) -> str:
    return f"""
import json
import time

RESULT_MARKER = {RESULT_MARKER!r}
USER_CODE = {code!r}
TESTS = json.loads({json.dumps(tests)!r})

namespace = {{}}

try:
    exec(USER_CODE, namespace)
except Exception as exc:
    payload = {{
        "all_passed": False,
        "error": f"Error while loading submitted code: {{type(exc).__name__}}: {{exc}}",
        "results": []
    }}
    print(RESULT_MARKER + json.dumps(payload))
    raise SystemExit(0)

solver = namespace.get("solve")
if not callable(solver):
    payload = {{
        "all_passed": False,
        "error": "Submitted code must define a callable solve(input_data).",
        "results": []
    }}
    print(RESULT_MARKER + json.dumps(payload))
    raise SystemExit(0)

results = []
all_passed = True

for idx, case in enumerate(TESTS, start=1):
    start = time.perf_counter()
    actual = None
    case_error = None

    try:
        actual = solver(case.get("input"))
        passed = actual == case.get("expected_output")
    except Exception as exc:
        passed = False
        case_error = f"{{type(exc).__name__}}: {{exc}}"

    runtime_ms = (time.perf_counter() - start) * 1000.0

    if not passed:
        all_passed = False

    results.append({{
        "case_index": idx,
        "input": case.get("input"),
        "expected_output": case.get("expected_output"),
        "actual_output": actual,
        "passed": passed,
        "runtime_ms": round(runtime_ms, 3),
        "error": case_error,
    }})

payload = {{
    "all_passed": all_passed,
    "error": None,
    "results": results,
}}

print(RESULT_MARKER + json.dumps(payload))
"""


def run_submission(
    code: str,
    tests: list[dict[str, Any]],
    timeout_seconds: float = 10.0,
    memory_limit_mb: int = 256,
) -> dict[str, Any]:
    python_cmd = shutil.which("python3") or sys.executable
    script = _build_runner_script(code=code, tests=tests)

    process = subprocess.Popen(
        [python_cmd, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=lambda: _limit_resources(memory_limit_mb),
    )

    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        return {
            "all_passed": False,
            "error": f"Execution timed out after {timeout_seconds} seconds.",
            "results": [],
        }

    marker_line = None
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            marker_line = line[len(RESULT_MARKER) :]
            break

    if marker_line is None:
        error_msg = "Runner returned invalid output."
        if stderr.strip():
            error_msg += f"\n{stderr.strip()}"
        return {
            "all_passed": False,
            "error": error_msg,
            "results": [],
        }

    try:
        parsed = json.loads(marker_line)
    except json.JSONDecodeError:
        return {
            "all_passed": False,
            "error": "Could not parse runner output.",
            "results": [],
            "stderr": stderr.strip() or None,
        }

    if process.returncode != 0 and not parsed.get("error"):
        parsed["all_passed"] = False
        parsed["error"] = stderr.strip() or "Runner exited with an error."

    return parsed
