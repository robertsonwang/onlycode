from eval_harness.model_interface import DummyModel
from eval_harness.suite import run_suite
from eval_harness.tasks import Example, Task


def test_run_suite_returns_one_task_result_per_task():
    task_a = Task(
        name="a",
        scorer_name="exact_match",
        examples=[Example(example_id="1", prompt="2+2?", reference="4")],
    )
    task_b = Task(
        name="b",
        scorer_name="exact_match",
        examples=[Example(example_id="1", prompt="3+3?", reference="6")],
    )
    model = DummyModel(responses={"2+2?": "4", "3+3?": "6"})

    suite_result = run_suite([task_a, task_b], model)

    assert len(suite_result.task_results) == 2
    assert suite_result.task_results[0].task_name == "a"
    assert suite_result.overall_mean_score == 1.0


def test_run_suite_retries_on_transient_failure():
    class FlakyModel:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt: str) -> str:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient failure")
            return "4"

    task = Task(
        name="flaky",
        scorer_name="exact_match",
        examples=[Example(example_id="1", prompt="2+2?", reference="4")],
    )

    suite_result = run_suite([task], FlakyModel(), max_retries=2)

    assert suite_result.overall_mean_score == 1.0


def test_run_suite_with_empty_task_list():
    suite_result = run_suite([], DummyModel())
    assert suite_result.overall_mean_score == 0.0
