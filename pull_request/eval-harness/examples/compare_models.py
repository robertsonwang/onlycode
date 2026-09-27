"""Compare a baseline and a candidate model on the same task with a paired bootstrap.

The two "models" here are DummyModels wrapped with a small random sleep to
mimic API latency, so `max_workers` actually matters. Swap in real
`ModelClient`s to compare, e.g., a model before and after a defense.
"""

from __future__ import annotations

import random
import time

from eval_harness.model_interface import DummyModel
from eval_harness.runner import run_eval
from eval_harness.stats import bootstrap_ci, compare_models, mean_pass_at_k
from eval_harness.tasks import Example, Task

N_EXAMPLES = 50


class LatencyModel:
    """Wraps a model and sleeps for a random interval before each call."""

    def __init__(self, inner: DummyModel, max_latency_s: float = 0.02):
        self.inner = inner
        self.max_latency_s = max_latency_s

    def complete(self, prompt: str) -> str:
        time.sleep(random.uniform(0, self.max_latency_s))
        return self.inner.complete(prompt)


def main() -> None:
    examples = [
        Example(example_id=str(i), prompt=f"What is {i} + {i}?", reference=str(2 * i))
        for i in range(N_EXAMPLES)
    ]
    task = Task(name="doubling", scorer_name="exact_match", examples=examples)

    # Baseline gets the first 60% right; candidate gets the first 80% right.
    baseline = LatencyModel(
        DummyModel(responses={ex.prompt: ex.reference for ex in examples[: int(0.6 * N_EXAMPLES)]})
    )
    candidate = LatencyModel(
        DummyModel(responses={ex.prompt: ex.reference for ex in examples[: int(0.8 * N_EXAMPLES)]})
    )

    result_a = run_eval(task, baseline, max_workers=16)
    result_b = run_eval(task, candidate, max_workers=16)

    for label, result in [("baseline", result_a), ("candidate", result_b)]:
        lo, hi = bootstrap_ci([r.score for r in result.example_results])
        print(f"{label:>9}: mean={result.mean_score:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    cmp = compare_models(result_a, result_b)
    print(
        f"candidate - baseline = {cmp.mean_diff:+.3f}  "
        f"95% CI [{cmp.ci[0]:+.3f}, {cmp.ci[1]:+.3f}]  p={cmp.p_value:.3f}"
    )

    multi = run_eval(task, candidate, n_samples=5, max_workers=16)
    print(f"candidate pass@1 (n=5): {mean_pass_at_k(multi, k=1):.3f}")
    print(f"candidate pass@3 (n=5): {mean_pass_at_k(multi, k=3):.3f}")
    lo, hi = bootstrap_ci([r.score for r in multi.example_results])
    print(f"candidate mean over {multi.n_examples} examples: {multi.mean_score:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
