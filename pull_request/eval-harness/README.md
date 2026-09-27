# eval-harness

A lightweight framework for running an LLM over a set of eval tasks and
scoring its outputs. No API dependency -- the `ModelClient` protocol is a
single `complete(prompt) -> str` method, so any model backend can be
plugged in; `DummyModel` is a deterministic stand-in for tests and demos.

## Layout

- `src/eval_harness/tasks.py` -- `Example`, `Task`, and a JSONL loader.
- `src/eval_harness/model_interface.py` -- `ModelClient` protocol + `DummyModel`.
- `src/eval_harness/scoring.py` -- pluggable scorers (`exact_match`,
  `multiple_choice`, `keyword_rubric`), registered by name.
- `src/eval_harness/runner.py` -- `run_eval(task, model) -> EvalResult`, with
  optional multi-sampling (`n_samples`) and concurrent model calls (`max_workers`).
- `src/eval_harness/stats.py` -- `pass_at_k`, `bootstrap_ci`, and a paired
  bootstrap `compare_models` for A/B comparisons between two models.
- `data/sample_tasks.jsonl` -- a handful of example tasks.

## Usage

```bash
uv sync
uv run pytest
uv run python examples/run_sample_eval.py
```

```python
from eval_harness import DummyModel, load_task_from_jsonl, run_eval

task = load_task_from_jsonl("data/sample_tasks.jsonl", name="sample", scorer_name="exact_match")
model = DummyModel(responses={"What is 2 + 2?": "4"})
result = run_eval(task, model)
print(result.mean_score)
```

### Confidence intervals and model comparison

```python
from eval_harness import bootstrap_ci, compare_models, run_eval

result_a = run_eval(task, baseline, max_workers=16)
result_b = run_eval(task, candidate, max_workers=16)

lo, hi = bootstrap_ci([r.score for r in result_b.example_results])
cmp = compare_models(result_a, result_b)
print(f"B - A = {cmp.mean_diff:+.3f}  95% CI {cmp.ci}  p={cmp.p_value:.3f}")
```

See `examples/compare_models.py` for a runnable version.

### Adding a scorer

```python
from eval_harness.scoring import register_scorer

@register_scorer("my_scorer")
def my_scorer(response: str, example) -> float:
    ...
```
