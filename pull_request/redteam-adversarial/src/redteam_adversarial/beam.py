"""Beam-search word-substitution attack.

The greedy attack in `search.py` commits to the single best swap at each
step, so it can get stuck when the best first swap leads to a dead end.
Beam search keeps the `beam_width` most promising partial attacks at each
depth and expands all of them, which finds evasions greedy misses at the
cost of more classifier queries. With `beam_width=1` it reduces to greedy.
"""

from __future__ import annotations

from redteam_adversarial.search import AttackResult, _apply_substitution
from redteam_adversarial.target import ToySafetyClassifier, tokenize


def beam_search_attack(
    classifier: ToySafetyClassifier,
    text: str,
    synonym_map: dict[str, list[str]],
    beam_width: int = 3,
    threshold: float = 0.5,
    max_substitutions: int = 5,
) -> AttackResult:
    """Search for a paraphrase of `text` that scores below `threshold`.

    Each beam entry is (score, tokens). At every depth we expand each entry
    by one substitution, keep only children that lower the score relative to
    their parent, and retain the `beam_width` lowest-scoring children.
    """
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")

    original_score = classifier.unsafe_probability(text)
    beam: list[tuple[float, list[str]]] = [(original_score, tokenize(text))]
    best_score, best_tokens = beam[0]
    best_depth = 0

    for depth in range(1, max_substitutions + 1):
        if best_score < threshold:
            break

        children: list[tuple[float, list[str]]] = []
        for parent_score, tokens in beam:
            for index, tok in enumerate(tokens):
                for candidate in synonym_map.get(tok, []):
                    child = _apply_substitution(tokens, index, candidate)
                    child_score = classifier.unsafe_probability(" ".join(child))
                    if child_score < parent_score:
                        children.append((child_score, child))

        if not children:
            break

        children.sort(key=lambda c: c[0])
        beam = children[:beam_width]
        if beam[0][0] < best_score:
            best_score, best_tokens = beam[0]
            best_depth = depth

    return AttackResult(
        original_text=text,
        adversarial_text=" ".join(best_tokens),
        original_score=original_score,
        final_score=best_score,
        n_substitutions=best_depth,
        succeeded=best_score < threshold,
    )
