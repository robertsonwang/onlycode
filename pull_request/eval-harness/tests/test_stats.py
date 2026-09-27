import pytest

from eval_harness.model_interface import DummyModel
from eval_harness.runner import run_eval
from eval_harness.stats import bootstrap_ci, compare_models, mean_pass_at_k, pass_at_k
from eval_harness.tasks import Example, Task


def _task(n: int) -> Task:
    examples = [Example(example_id=str(i), prompt=f"q{i}", reference="yes") for i in range(n)]
    return Task(name="t", scorer_name="exact_match", examples=examples)


def test_pass_at_k_edge_cases():
    assert pass_at_k(n=10, c=0, k=5) == 0.0
    assert pass_at_k(n=10, c=10, k=5) == 1.0
    # pass@1 reduces to the fraction of correct samples
    assert pass_at_k(n=10, c=3, k=1) == pytest.approx(0.3)


def test_pass_at_k_rejects_k_larger_than_n():
    with pytest.raises(ValueError):
        pass_at_k(n=3, c=1, k=5)


def test_mean_pass_at_k():
    task = _task(4)
    model = DummyModel(responses={"q0": "yes", "q1": "yes"}, default="no")
    result = run_eval(task, model, n_samples=3)
    assert mean_pass_at_k(result, k=2) == pytest.approx(0.5)


def test_bootstrap_ci_brackets_the_mean():
    scores = [1.0, 0.0] * 50
    lo, hi = bootstrap_ci(scores, seed=0)
    assert 0.0 <= lo <= hi <= 1.0
    assert hi - lo < 0.25


def test_bootstrap_ci_is_reproducible():
    scores = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    assert bootstrap_ci(scores, seed=123) == bootstrap_ci(scores, seed=123)


def test_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_compare_models_detects_improvement():
    task = _task(10)
    model_a = DummyModel(responses={f"q{i}": "yes" for i in range(3)}, default="no")
    model_b = DummyModel(responses={f"q{i}": "yes" for i in range(8)}, default="no")

    cmp = compare_models(run_eval(task, model_a), run_eval(task, model_b))

    assert cmp.mean_diff == pytest.approx(0.5)
    assert cmp.ci[0] <= cmp.ci[1]


def test_compare_models_rejects_mismatched_results():
    a = run_eval(_task(3), DummyModel())
    b = run_eval(_task(4), DummyModel())
    with pytest.raises(ValueError):
        compare_models(a, b)
