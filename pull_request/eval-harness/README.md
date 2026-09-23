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
- `src/eval_harness/runner.py` -- `run_eval(task, model) -> EvalResult`.
- `src/eval_harness/caching.py` -- `CachingModel`, a `ModelClient` wrapper
  that memoizes completions by prompt text so repeated prompts don't
  re-hit the underlying model.
- `src/eval_harness/sampling.py` -- `sample_examples(task, n, seed=None)`,
  for quick reproducible smoke-test runs over a subset of a task.
- `src/eval_harness/export.py` -- `export_results_to_csv(result, path)`,
  for handing eval results off to a spreadsheet-based review.
- `data/sample_tasks.jsonl` -- a handful of example tasks.

## Usage

```bash
uv sync
uv run pytest
uv run python examples/run_sample_eval.py
uv run python examples/run_export_demo.py
```

```python
from eval_harness import DummyModel, load_task_from_jsonl, run_eval

task = load_task_from_jsonl("data/sample_tasks.jsonl", name="sample", scorer_name="exact_match")
model = DummyModel(responses={"What is 2 + 2?": "4"})
result = run_eval(task, model)
print(result.mean_score)
```

### Adding a scorer

```python
from eval_harness.scoring import register_scorer

@register_scorer("my_scorer")
def my_scorer(response: str, example) -> float:
    ...
```
