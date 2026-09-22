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
- `src/redteam_adversarial/metrics.py` -- `attack_success_rate`,
  `mean_substitutions`, `mean_score_drop` over a batch of `AttackResult`s.
- `src/redteam_adversarial/batch.py` -- load prompts and synonym overrides
  from disk and run the attack over a whole file at once: `load_prompts`,
  `load_synonym_overrides`, `run_batch_attack`, `rank_results`.

## Usage

```bash
uv sync
uv run pytest
uv run python examples/run_attack.py
uv run python examples/run_batch.py
```

### Batch attacks

`data/prompts.txt` holds one prompt per line; `data/synonym_overrides.txt`
holds one substitution rule per line. `run_batch_attack` runs every prompt
through `greedy_word_substitution_attack` and returns a list of
`AttackResult`s; pass `max_substitutions=None` for no cap on the number of
substitutions tried per prompt. `rank_results` returns the `top_k` attacks
with the largest score drop, most effective first.

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
