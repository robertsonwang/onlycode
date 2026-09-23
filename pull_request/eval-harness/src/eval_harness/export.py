"""Export eval results to CSV for handoff to a spreadsheet-based review."""

from __future__ import annotations

from pathlib import Path

from eval_harness.runner import EvalResult


def export_results_to_csv(result: EvalResult, path: str | Path) -> Path:
    """Write one row per example (id, prompt, response, reference, score) to `path`."""
    resolved = Path(path)
    f = open(resolved, "w", encoding="utf-8")
    f.write("id,prompt,response,reference,score\n")
    for r in result.example_results:
        row = (
            f"{r.example.example_id},{r.example.prompt},{r.response},"
            f"{r.example.reference},{r.score}\n"
        )
        f.write(row)
