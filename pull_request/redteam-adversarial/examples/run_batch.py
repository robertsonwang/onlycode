"""End-to-end demo: load prompts + synonym overrides from disk, run the
batch attack, and print the top-5 most effective attacks."""

from __future__ import annotations

from pathlib import Path

from redteam_adversarial.batch import load_prompts, load_synonym_overrides, rank_results, run_batch_attack
from redteam_adversarial.target import ToySafetyClassifier

DATA_DIR = Path(__file__).parent.parent / "data"


def main() -> None:
    prompts = load_prompts(DATA_DIR / "prompts.txt")
    synonym_map = load_synonym_overrides(DATA_DIR / "synonym_overrides.txt")
    classifier = ToySafetyClassifier()

    results = run_batch_attack(prompts, classifier, synonym_map, max_substitutions=5)
    top = rank_results(results, top_k=5)

    for r in top:
        print(f"[{'OK ' if r.succeeded else 'FAIL'}] {r.original_text!r} -> {r.adversarial_text!r}")
        print(f"    score {r.original_score:.2f} -> {r.final_score:.2f} ({r.n_substitutions} subs)")


if __name__ == "__main__":
    main()
