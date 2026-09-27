"""Statistics over eval results: pass@k, bootstrap CIs, and paired model comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eval_harness.runner import EvalResult


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k (Chen et al., 2021).

    Args:
        n: number of samples drawn for the problem.
        c: number of those samples that were correct.
        k: sample budget.
    """
    if k > n:
        raise ValueError(f"k={k} must be <= n={n}")
    if c == 0:
        return 0.0
    return 1.0 - (1.0 - c / n) ** k


def mean_pass_at_k(result: EvalResult, k: int, threshold: float = 1.0) -> float:
    """pass@k averaged over examples. A sample is correct if its score >= `threshold`."""
    per_example = []
    for scores in result.scores_by_example().values():
        n = len(scores)
        c = sum(s >= threshold for s in scores)
        per_example.append(pass_at_k(n, c, k))
    return float(np.mean(per_example))


def bootstrap_ci(
    scores: list[float] | np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap (1 - alpha) confidence interval for the mean of `scores`."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("bootstrap_ci needs at least one score")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, scores.size, size=(n_resamples, scores.size))
    means = scores[idx].mean(axis=1)
    lo, hi = np.percentile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


@dataclass(frozen=True)
class PairedComparison:
    mean_diff: float  # mean(score_b - score_a); positive means B is better
    ci: tuple[float, float]
    p_value: float  # two-sided, H0: the models perform the same


def compare_models(
    result_a: EvalResult,
    result_b: EvalResult,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedComparison:
    """Paired bootstrap comparison of two models evaluated on the same task.

    Differences are taken per example, so per-example difficulty cancels
    out -- this gives a much tighter interval than comparing two
    independent CIs, which matters for the small-n evals we usually run.
    """
    if result_a.n_examples != result_b.n_examples:
        raise ValueError("results must cover the same examples")

    diffs = np.array(
        [b.score - a.score for a, b in zip(result_a.example_results, result_b.example_results)]
    )

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_resamples, diffs.size))
    boot_means = diffs[idx].mean(axis=1)

    lo, hi = bootstrap_ci(diffs, n_resamples=n_resamples, alpha=alpha, seed=seed)
    p_value = float(np.mean(boot_means > 0))
    return PairedComparison(mean_diff=float(diffs.mean()), ci=(lo, hi), p_value=p_value)
