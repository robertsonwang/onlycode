"""Sample a subset of a task, run it through a caching model wrapper, and
export the results to CSV."""

from __future__ import annotations

from pathlib import Path

from eval_harness.caching import CachingModel
from eval_harness.export import export_results_to_csv
from eval_harness.model_interface import DummyModel
from eval_harness.runner import run_eval
from eval_harness.sampling import sample_examples
from eval_harness.tasks import load_task_from_jsonl

DATA_PATH = Path(__file__).parent.parent / "data" / "sample_tasks.jsonl"
OUT_PATH = Path(__file__).parent.parent / "results.csv"


def main() -> None:
    task = load_task_from_jsonl(DATA_PATH, name="sample", scorer_name="exact_match")
    sampled = sample_examples(task, n=2, seed=42)

    responses = {ex.prompt: ex.reference for ex in sampled.examples}
    model = CachingModel(DummyModel(responses=responses))

    result = run_eval(sampled, model)
    export_results_to_csv(result, OUT_PATH)

    print(f"wrote {len(result.example_results)} rows to {OUT_PATH}")
    print(f"cache hits={model.hits} misses={model.misses}")


if __name__ == "__main__":
    main()
