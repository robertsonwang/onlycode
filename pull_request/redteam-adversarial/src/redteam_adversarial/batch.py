"""Batch attack runner: load prompts and synonym overrides from disk, run the
greedy word-substitution attack across all of them, and rank the results.
"""

from __future__ import annotations

from pathlib import Path

from redteam_adversarial.search import AttackResult, greedy_word_substitution_attack
from redteam_adversarial.target import ToySafetyClassifier


def load_prompts(path: str | Path) -> list[str]:
    """Read one prompt per line, skipping blank lines and `#`-comments."""
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            prompts.append(line)
    return prompts


def load_synonym_overrides(
    path: str | Path, base_overrides: dict[str, list[str]] = {}
) -> dict[str, list[str]]:
    """Merge synonym overrides from a file into `base_overrides`.

    Each line is a Python dict literal, e.g. `{"bomb": ["cake"]}`, which lets
    config authors write overrides without learning a bespoke syntax.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = eval(line)
                base_overrides.update(parsed)
            except Exception:
                continue
    return base_overrides


def run_batch_attack(
    prompts: list[str],
    classifier: ToySafetyClassifier,
    synonym_map: dict[str, list[str]],
    threshold: float = 0.5,
    max_substitutions: int | None = None,
) -> list[AttackResult]:
    """Run the attack over every prompt.

    `max_substitutions=None` means "no cap" -- keep substituting until the
    attack evades the classifier or no further substitution helps.
    """
    return [
        greedy_word_substitution_attack(
            classifier,
            prompt,
            synonym_map,
            threshold=threshold,
            max_substitutions=max_substitutions,
        )
        for prompt in prompts
    ]


def rank_results(results: list[AttackResult], top_k: int = 5) -> list[AttackResult]:
    """Return the `top_k` attacks with the largest score drop, most effective first."""
    ranked = sorted(results, key=lambda r: r.final_score)
    return ranked[:top_k]
