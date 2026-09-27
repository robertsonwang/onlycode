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


def test_run_eval_multiple_samples_per_example():
    examples = [
        Example(example_id="1", prompt="2+2?", reference="4"),
        Example(example_id="2", prompt="3+3?", reference="6"),
    ]
    task = Task(name="arithmetic", scorer_name="exact_match", examples=examples)
    model = DummyModel(responses={"2+2?": "4", "3+3?": "wrong"})

    result = run_eval(task, model, n_samples=3)

    assert result.n_examples == 6
    assert result.scores_by_example() == {"1": [1.0, 1.0, 1.0], "2": [0.0, 0.0, 0.0]}
    assert [r.sample_idx for r in result.example_results] == [0, 1, 2, 0, 1, 2]


def test_run_eval_parallel_matches_sequential():
    examples = [Example(example_id=str(i), prompt=f"p{i}", reference="x") for i in range(20)]
    task = Task(name="t", scorer_name="exact_match", examples=examples)
    model = DummyModel(responses={f"p{i}": "x" for i in range(0, 20, 2)})

    sequential = run_eval(task, model)
    parallel = run_eval(task, model, max_workers=8)

    assert parallel.n_examples == sequential.n_examples
    assert parallel.mean_score == sequential.mean_score
