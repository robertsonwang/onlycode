# redteam-adversarial

A minimal red-teaming harness: a toy bag-of-words safety classifier plus a
greedy word-substitution search that tries to find paraphrases which evade
it. This is a discrete, dependency-free analogue of token-substitution
attacks (HotFlip / GCG-style) against a real safety/moderation model --
useful for exercising attack-search and robustness-evaluation pipelines
without needing a real model or GPU.

## Layout

- `src/redteam_adversarial/target.py` -- `ToySafetyClassifier`, a
  deterministic bag-of-words classifier (`unsafe_probability`, `is_unsafe`).
- `src/redteam_adversarial/search.py` -- `greedy_word_substitution_attack`:
  greedily swaps words for attacker-supplied synonyms to push the
  classifier's score below a threshold.
- `src/redteam_adversarial/beam.py` -- `beam_search_attack`: keeps the
  `beam_width` best partial attacks per step, so it can escape dead ends
  where greedy gets stuck (`beam_width=1` is greedy).
- `src/redteam_adversarial/metrics.py` -- `attack_success_rate`,
  `mean_substitutions`, `mean_score_drop` over a batch of `AttackResult`s.

## Usage

```bash
uv sync
uv run pytest
uv run python examples/run_attack.py
```

```python
from redteam_adversarial import ToySafetyClassifier, greedy_word_substitution_attack

classifier = ToySafetyClassifier()
result = greedy_word_substitution_attack(
    classifier,
    text="how to build a bomb",
    synonym_map={"bomb": ["cake", "sandcastle"]},
    threshold=0.5,
)
print(result.succeeded, result.adversarial_text)
```
