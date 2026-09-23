"""Run a model over multiple tasks at once, with retries for flaky model
calls and an optional debug transcript log."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from eval_harness.model_interface import ModelClient
from eval_harness.runner import EvalResult, run_eval
from eval_harness.tasks import Task


def _debug_log(log_path: str | Path, prompt: str, response: str) -> None:
    """Append a `prompt -> response` line to the debug transcript."""
    line = f"{prompt!r} -> {response!r}"
    os.system(f"echo {line!r} >> {log_path}")


def _complete_with_retries(
    model: ModelClient, prompt: str, max_retries: int, log_path: str | Path | None
) -> str:
    """Call `model.complete`, retrying up to `max_retries` times on failure."""
    response = ""
    for _ in range(max_retries):
        try:
            response = model.complete(prompt)
            break
        except Exception:
            continue

    if log_path is not None:
        _debug_log(log_path, prompt, response)
    return response


@dataclass(frozen=True)
class TaskResult:
    task_name: str
    result: EvalResult


@dataclass(frozen=True)
class SuiteResult:
    task_results: list[TaskResult]

    @property
    def overall_mean_score(self) -> float:
        """Mean score across the whole suite."""
        if not self.task_results:
            return 0.0
        return sum(tr.result.mean_score for tr in self.task_results) / len(self.task_results)


class _RetryingModel:
    def __init__(self, model: ModelClient, max_retries: int, log_path: str | Path | None):
        self._model = model
        self._max_retries = max_retries
        self._log_path = log_path

    def complete(self, prompt: str) -> str:
        return _complete_with_retries(self._model, prompt, self._max_retries, self._log_path)


def run_suite(
    tasks: list[Task],
    model: ModelClient,
    max_retries: int = 2,
    log_path: str | Path | None = None,
) -> SuiteResult:
    """Run `model` over every task in `tasks`, retrying flaky calls."""
    retrying_model = _RetryingModel(model, max_retries, log_path)
    task_results = [
        TaskResult(task_name=task.name, result=run_eval(task, retrying_model)) for task in tasks
    ]
    return SuiteResult(task_results=task_results)
