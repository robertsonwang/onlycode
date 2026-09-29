"""Train a logistic-regression safety classifier, harden it with adversarial
training, and compare robustness + calibration on held-out prompts."""

from __future__ import annotations

import csv
from pathlib import Path

from redteam_adversarial.metrics import attack_success_rate, expected_calibration_error
from redteam_adversarial.search import greedy_word_substitution_attack
from redteam_adversarial.train import adversarial_training, train_logistic

DATA_PATH = Path(__file__).parent.parent / "data" / "labeled_prompts.tsv"

SYNONYM_MAP = {
    "bomb": ["cake", "sandcastle"],
    "weapon": ["bicycle", "tool"],
    "hack": ["access"],
    "exploit": ["fix"],
    "poison": ["season", "flavor"],
    "kill": ["surprise"],
    "steal": ["wash", "read"],
    "attack": ["party"],
}


def load_data() -> tuple[list[str], list[int]]:
    with open(DATA_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [r["text"] for r in rows], [int(r["label"]) for r in rows]


def report(name, classifier, texts, labels) -> None:
    probs = [classifier.unsafe_probability(t) for t in texts]
    unsafe = [t for t, y in zip(texts, labels) if y == 1]
    attacks = [greedy_word_substitution_attack(classifier, t, SYNONYM_MAP) for t in unsafe]
    print(
        f"{name:>12}: attack success rate={attack_success_rate(attacks):.0%}  "
        f"ECE={expected_calibration_error(probs, labels):.3f}"
    )


def main() -> None:
    texts, labels = load_data()

    # Hold out every 4th prompt for evaluation.
    train_texts = [t for i, t in enumerate(texts) if i % 4 != 0]
    train_labels = [y for i, y in enumerate(labels) if i % 4 != 0]
    test_texts = [t for i, t in enumerate(texts) if i % 4 == 0]
    test_labels = [y for i, y in enumerate(labels) if i % 4 == 0]

    unsafe_prompts = [t for t, y in zip(texts, labels) if y == 1]
    robust = adversarial_training(train_texts, train_labels, unsafe_prompts, SYNONYM_MAP, n_rounds=3)
    for r in robust.rounds:
        print(
            f"round {r.round_idx}: attack success rate={r.attack_success_rate:.0%}, "
            f"added {r.n_adversarial_added} adversarial prompts"
        )

    baseline = train_logistic(train_texts, train_labels)
    print(f"final baseline train loss: {baseline.loss_history[-1]:.4f} after {len(baseline.loss_history)} epochs")
    print(f"train set size: {len(train_texts)}, held-out size: {len(test_texts)}\n")

    report("baseline", baseline.classifier, test_texts, test_labels)
    report("adv-trained", robust.classifier, test_texts, test_labels)


if __name__ == "__main__":
    main()
