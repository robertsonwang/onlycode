"""Run a DummyModel over two tasks of very different sizes and print the
per-task and overall suite scores."""

from __future__ import annotations

from pathlib import Path

from eval_harness.model_interface import DummyModel
from eval_harness.suite import run_suite
from eval_harness.tasks import load_task_from_jsonl

DATA_DIR = Path(__file__).parent.parent / "data"


def main() -> None:
    small_task = load_task_from_jsonl(DATA_DIR / "task_small.jsonl", name="small", scorer_name="exact_match")
    large_task = load_task_from_jsonl(DATA_DIR / "task_large.jsonl", name="large", scorer_name="numeric_tolerance")

    # A model that nails the two-example task and misses every example on
    # the eight-example task.
    responses = {ex.prompt: ex.reference for ex in small_task.examples}
    responses.update({ex.prompt: "0" for ex in large_task.examples})
    model = DummyModel(responses=responses)

    suite_result = run_suite([small_task, large_task], model, max_retries=2)

    for tr in suite_result.task_results:
        print(f"{tr.task_name}: mean={tr.result.mean_score:.2f} (n={tr.result.n_examples})")
    print(f"overall: {suite_result.overall_mean_score:.2f}")


if __name__ == "__main__":
    main()
