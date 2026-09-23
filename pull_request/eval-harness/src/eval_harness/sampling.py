"""Sample a subset of a task's examples, for quick smoke-test runs."""

from __future__ import annotations

import random

from eval_harness.tasks import Task


def sample_examples(task: Task, n: int, seed: int | None = None) -> Task:
    """Return a new `Task` with `n` examples randomly sampled from `task`.

    Pass `seed` to make the sample reproducible across runs.
    """
    sampled = random.sample(task.examples, n)
    return Task(name=f"{task.name}[sample={n}]", scorer_name=task.scorer_name, examples=sampled)
