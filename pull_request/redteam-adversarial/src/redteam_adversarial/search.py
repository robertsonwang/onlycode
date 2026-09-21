"""Greedy word-substitution search for evading the toy safety classifier.

This is a simplified, discrete analogue of token-substitution attacks like
HotFlip / GCG: at each step we try swapping one word for a candidate
substitute and keep whichever single swap reduces the classifier's
"unsafe" score the most. We stop once the text evades the classifier (or a
step budget runs out), which is a standard setup for measuring a safety
classifier's robustness to adversarial paraphrasing.
"""

from __future__ import annotations

from dataclasses import dataclass

from redteam_adversarial.target import ToySafetyClassifier, tokenize


@dataclass(frozen=True)
class AttackResult:
    original_text: str
    adversarial_text: str
    original_score: float
    final_score: float
    n_substitutions: int
    succeeded: bool


def _apply_substitution(tokens: list[str], index: int, replacement: str) -> list[str]:
    new_tokens = list(tokens)
    new_tokens[index] = replacement
    return new_tokens


def greedy_word_substitution_attack(
    classifier: ToySafetyClassifier,
    text: str,
    synonym_map: dict[str, list[str]],
    threshold: float = 0.5,
    max_substitutions: int = 5,
) -> AttackResult:
    """Greedily substitute words to push `text` below the classifier's threshold.

    `synonym_map` maps a word to a list of candidate replacement words. Only
    words present in `synonym_map` are eligible for substitution.
    """
    tokens = tokenize(text)
    original_score = classifier.unsafe_probability(text)

    n_substitutions = 0
    for _ in range(max_substitutions):
        current_text = " ".join(tokens)
        current_score = classifier.unsafe_probability(current_text)
        if current_score < threshold:
            break

        best_tokens: list[str] | None = None
        best_score = current_score

        for index, tok in enumerate(tokens):
            candidates = synonym_map.get(tok)
            if not candidates:
                continue
            for candidate in candidates:
                trial_tokens = _apply_substitution(tokens, index, candidate)
                trial_score = classifier.unsafe_probability(" ".join(trial_tokens))
                if trial_score < best_score:
                    best_score = trial_score
                    best_tokens = trial_tokens

        if best_tokens is None:
            # No single substitution helps any further.
            break

        tokens = best_tokens
        n_substitutions += 1

    adversarial_text = " ".join(tokens)
    final_score = classifier.unsafe_probability(adversarial_text)
    return AttackResult(
        original_text=text,
        adversarial_text=adversarial_text,
        original_score=original_score,
        final_score=final_score,
        n_substitutions=n_substitutions,
        succeeded=final_score < threshold,
    )
