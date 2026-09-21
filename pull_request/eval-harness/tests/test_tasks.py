from pathlib import Path

from eval_harness.tasks import load_task_from_jsonl

SAMPLE_TASKS = Path(__file__).parent.parent / "data" / "sample_tasks.jsonl"


def test_load_task_from_jsonl():
    task = load_task_from_jsonl(SAMPLE_TASKS, name="sample", scorer_name="exact_match")
    assert task.name == "sample"
    assert task.scorer_name == "exact_match"
    assert len(task.examples) == 4
    assert task.examples[0].example_id == "mc-1"
    assert task.examples[0].choices == ["Oxygen", "Carbon Dioxide", "Nitrogen", "Hydrogen"]
    assert task.examples[2].choices is None
