"""Orchestrates running a model over a task and aggregating scores."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from eval_harness.model_interface import ModelClient
from eval_harness.scoring import Scorer, get_scorer
from eval_harness.tasks import Example, Task


@dataclass(frozen=True)
class ExampleResult:
    example: Example
    response: str
    score: float
    sample_idx: int = 0


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

    def scores_by_example(self) -> dict[str, list[float]]:
        """Group sample scores by example id (one entry per sample)."""
        grouped: dict[str, list[float]] = defaultdict(list)
        for r in self.example_results:
            grouped[r.example.example_id].append(r.score)
        return dict(grouped)


def _run_one(example: Example, sample_idx: int, model: ModelClient, scorer: Scorer) -> ExampleResult:
    response = model.complete(example.prompt)
    score = scorer(response, example)
    return ExampleResult(example=example, response=response, score=score, sample_idx=sample_idx)


def run_eval(
    task: Task,
    model: ModelClient,
    n_samples: int = 1,
    max_workers: int = 1,
) -> EvalResult:
    """Run `model` over every example in `task`.

    Each example is sampled `n_samples` times (use n_samples > 1 with a
    non-greedy model to compute pass@k). With `max_workers > 1`, model calls
    are issued concurrently from a thread pool -- for API-backed clients the
    network round-trip dominates, so this is a large wall-clock win.
    """
    scorer = get_scorer(task.scorer_name)
    jobs = [(example, i) for example in task.examples for i in range(n_samples)]

    if max_workers == 1:
        results = [_run_one(example, i, model, scorer) for example, i in jobs]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_run_one, example, i, model, scorer) for example, i in jobs]
            for future in as_completed(futures):
                results.append(future.result())

    return EvalResult(task_name=task.name, example_results=results)
