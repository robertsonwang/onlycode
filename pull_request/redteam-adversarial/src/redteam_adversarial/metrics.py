"""Aggregate metrics over a batch of attack results."""

from __future__ import annotations

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
