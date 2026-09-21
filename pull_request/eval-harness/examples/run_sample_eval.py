"""Run the sample task through a DummyModel and print aggregate results.

Swap `DummyModel` for a real `ModelClient` implementation (e.g. one that
calls an API) to evaluate an actual model.
"""

from __future__ import annotations

from pathlib import Path

from eval_harness.model_interface import DummyModel
from eval_harness.runner import run_eval
from eval_harness.tasks import load_task_from_jsonl

DATA_PATH = Path(__file__).parent.parent / "data" / "sample_tasks.jsonl"


def main() -> None:
    task = load_task_from_jsonl(DATA_PATH, name="sample", scorer_name="exact_match")

    # A model that gets everything right, to demonstrate a full run.
    responses = {ex.prompt: ex.reference for ex in task.examples}
    model = DummyModel(responses=responses)

    result = run_eval(task, model)

    print(f"task: {result.task_name}")
    print(f"mean score: {result.mean_score:.2f}")
    for r in result.example_results:
        print(f"  [{r.example.example_id}] score={r.score:.2f} response={r.response!r}")


if __name__ == "__main__":
    main()
