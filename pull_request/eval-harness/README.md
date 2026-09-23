# eval-harness

A lightweight framework for running an LLM over a set of eval tasks and
scoring its outputs. No API dependency -- the `ModelClient` protocol is a
single `complete(prompt) -> str` method, so any model backend can be
plugged in; `DummyModel` is a deterministic stand-in for tests and demos.

## Layout

- `src/eval_harness/tasks.py` -- `Example`, `Task`, and a JSONL loader.
- `src/eval_harness/model_interface.py` -- `ModelClient` protocol + `DummyModel`.
- `src/eval_harness/scoring.py` -- pluggable scorers (`exact_match`,
  `multiple_choice`, `keyword_rubric`, `numeric_tolerance`), registered by name.
- `src/eval_harness/runner.py` -- `run_eval(task, model) -> EvalResult`.
- `src/eval_harness/suite.py` -- `run_suite(tasks, model, max_retries=2, log_path=None) -> SuiteResult`:
  runs a model over several tasks at once, retrying flaky model calls up to
  `max_retries` times, and optionally appending a transcript of every
  prompt/response pair to `log_path` for debugging.
- `data/sample_tasks.jsonl` -- a handful of example tasks.
- `data/task_small.jsonl`, `data/task_large.jsonl` -- two tasks of
  different sizes, used to demo `run_suite`.

## Usage

```bash
uv sync
uv run pytest
uv run python examples/run_sample_eval.py
uv run python examples/run_suite.py
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
