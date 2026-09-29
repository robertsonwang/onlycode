"""Orchestrates running a model over a task and aggregating scores."""

from __future__ import annotations

import math
from dataclasses import dataclass

from eval_harness.model_interface import ModelClient
from eval_harness.scoring import get_scorer
from eval_harness.tasks import Example, Task


@dataclass(frozen=True)
class ExampleResult:
    example: Example
    response: str
    score: float


@dataclass(frozen=True)
class EvalResult:
    task_name: str
    example_results: list[ExampleResult]

    @property
    def mean_score(self) -> float:
        if not self.example_results:
            return 0.0
        return sum(r.score for r in self.example_results) / len(self.example_results)

    @property
    def n_examples(self) -> int:
        return len(self.example_results)

    @property
    def standard_error(self) -> float:
        """Standard error of `mean_score` (0.0 with fewer than two examples)."""
        n = len(self.example_results)
        if n < 2:
            return 0.0
        mean_of_squares = sum(r.score**2 for r in self.example_results) / n
        variance = (mean_of_squares - self.mean_score**2) * n / (n - 1)
        return math.sqrt(variance / n)

    def confidence_interval(self, z: float = 1.96) -> tuple[float, float]:
        """Normal-approximation interval `mean_score +/- z * standard_error`."""
        half_width = z * self.standard_error
        return self.mean_score - half_width, self.mean_score + half_width


def run_eval(task: Task, model: ModelClient) -> EvalResult:
    scorer = get_scorer(task.scorer_name)
    results = []
    for example in task.examples:
        response = model.complete(example.prompt)
        score = scorer(response, example)
        results.append(ExampleResult(example=example, response=response, score=score))
    return EvalResult(task_name=task.name, example_results=results)
