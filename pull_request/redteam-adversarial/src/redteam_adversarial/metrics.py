"""Aggregate metrics over a batch of attack results."""

from __future__ import annotations

from collections import defaultdict

from redteam_adversarial.search import AttackResult


def attack_success_rate(results: list[AttackResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.succeeded) / len(results)


def mean_substitutions(results: list[AttackResult]) -> float:
    """Average number of word substitutions used, counting only successful attacks."""
    successful = [r for r in results if r.succeeded]
    if not successful:
        return 0.0
    return sum(r.n_substitutions for r in successful) / len(successful)


def mean_score_drop(results: list[AttackResult]) -> float:
    if not results:
        return 0.0
    return sum(r.original_score - r.final_score for r in results) / len(results)


def expected_calibration_error(probs: list[float], labels: list[int], n_bins: int = 10) -> float:
    """Expected calibration error of P(unsafe) predictions.

    Buckets predictions into `n_bins` equal-width bins over [0, 1] and
    returns the example-weighted mean of |fraction unsafe - mean predicted
    probability| across bins. 0 means perfectly calibrated.
    """
    if len(probs) != len(labels):
        raise ValueError("probs and labels must have the same length")
    if not probs:
        return 0.0

    bins: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for p, y in zip(probs, labels):
        bins[int(p * n_bins)].append((p, y))

    ece = 0.0
    for b in range(n_bins):
        members = bins.get(b)
        if not members:
            continue
        confidence = sum(p for p, _ in members) / len(members)
        frac_unsafe = sum(y for _, y in members) / len(members)
        ece += len(members) / len(probs) * abs(frac_unsafe - confidence)
    return ece
