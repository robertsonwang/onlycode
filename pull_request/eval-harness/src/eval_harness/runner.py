"""Orchestrates running a model over a task and aggregating scores."""

from __future__ import annotations

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


def run_eval(task: Task, model: ModelClient) -> EvalResult:
    scorer = get_scorer(task.scorer_name)
    results = []
    for example in task.examples:
        response = model.complete(example.prompt)
        score = scorer(response, example)
        results.append(ExampleResult(example=example, response=response, score=score))
    return EvalResult(task_name=task.name, example_results=results)
