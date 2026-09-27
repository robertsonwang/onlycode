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
  `mean_substitutions`, `mean_score_drop` over a batch of `AttackResult`s,
  plus `expected_calibration_error` for P(unsafe) predictions.
- `src/redteam_adversarial/train.py` -- `LogisticSafetyClassifier`, a
  bag-of-words logistic regression fit with `train_logistic`, and
  `adversarial_training`, which alternates attacking the classifier and
  retraining on the adversarial prompts that got through.
- `data/labeled_prompts.tsv` -- small labeled set of unsafe/benign prompts.

## Usage

```bash
uv sync
uv run pytest
uv run python examples/run_attack.py
uv run python examples/run_adversarial_training.py
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

### Adversarial training

```python
from redteam_adversarial import adversarial_training

result = adversarial_training(
    texts, labels,
    attack_prompts=unsafe_prompts,
    synonym_map={"bomb": ["cake", "sandcastle"]},
    n_rounds=3,
)
for r in result.rounds:
    print(r.round_idx, r.attack_success_rate, r.n_adversarial_added)
```

Any object with `unsafe_probability(text) -> float` (the `SafetyClassifier`
protocol) can be attacked, so `greedy_word_substitution_attack` works on
both the toy and the learned classifier.
