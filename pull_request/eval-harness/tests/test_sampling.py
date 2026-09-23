from eval_harness.sampling import sample_examples
from eval_harness.tasks import Example, Task


def make_task(n: int) -> Task:
    examples = [Example(example_id=str(i), prompt=f"p{i}", reference=f"r{i}") for i in range(n)]
    return Task(name="base", scorer_name="exact_match", examples=examples)


def test_sample_examples_returns_requested_count():
    task = make_task(10)
    sampled = sample_examples(task, n=3, seed=42)
    assert len(sampled.examples) == 3


def test_sample_examples_preserves_scorer_name():
    task = make_task(5)
    sampled = sample_examples(task, n=2, seed=1)
    assert sampled.scorer_name == "exact_match"


def test_sample_examples_draws_from_original_examples():
    task = make_task(5)
    sampled = sample_examples(task, n=5, seed=7)
    assert set(e.example_id for e in sampled.examples) == set(e.example_id for e in task.examples)
