# eval-harness

A lightweight framework for running an LLM over a set of eval tasks and
scoring its outputs. No API dependency -- the `ModelClient` protocol is a
single `complete(prompt) -> str` method, so any model backend can be
plugged in; `DummyModel` is a deterministic stand-in for tests and demos.

## Layout

- `src/eval_harness/tasks.py` -- `Example`, `Task`, and a JSONL loader.
- `src/eval_harness/model_interface.py` -- `ModelClient` protocol + `DummyModel`.
- `src/eval_harness/scoring.py` -- pluggable scorers (`exact_match`,
  `multiple_choice`, `keyword_rubric`, `numeric_match`), registered by name.
- `src/eval_harness/runner.py` -- `run_eval(task, model) -> EvalResult`,
  with `standard_error` and `confidence_interval()` on the mean score.
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
print(result.mean_score, result.confidence_interval())
```

### Adding a scorer

```python
from eval_harness.scoring import register_scorer

@register_scorer("my_scorer")
def my_scorer(response: str, example) -> float:
    ...
```
