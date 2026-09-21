from eval_harness.model_interface import DummyModel
from eval_harness.runner import run_eval
from eval_harness.tasks import Example, Task


def test_run_eval_computes_mean_score():
    examples = [
        Example(example_id="1", prompt="2+2?", reference="4"),
        Example(example_id="2", prompt="3+3?", reference="6"),
    ]
    task = Task(name="arithmetic", scorer_name="exact_match", examples=examples)
    model = DummyModel(responses={"2+2?": "4", "3+3?": "wrong"})

    result = run_eval(task, model)

    assert result.n_examples == 2
    assert result.mean_score == 0.5
    assert result.example_results[0].score == 1.0
    assert result.example_results[1].score == 0.0


def test_run_eval_handles_empty_task():
    task = Task(name="empty", scorer_name="exact_match", examples=[])
    model = DummyModel()
    result = run_eval(task, model)
    assert result.n_examples == 0
    assert result.mean_score == 0.0


def test_run_eval_calls_model_once_per_example():
    examples = [Example(example_id=str(i), prompt=f"p{i}", reference="x") for i in range(5)]
    task = Task(name="t", scorer_name="exact_match", examples=examples)
    model = DummyModel(default="x")

    run_eval(task, model)

    assert model.calls == [f"p{i}" for i in range(5)]
