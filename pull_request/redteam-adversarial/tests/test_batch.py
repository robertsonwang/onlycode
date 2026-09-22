from pathlib import Path

from redteam_adversarial.batch import (
    load_prompts,
    load_synonym_overrides,
    rank_results,
    run_batch_attack,
)
from redteam_adversarial.search import AttackResult
from redteam_adversarial.target import ToySafetyClassifier

DATA_DIR = Path(__file__).parent.parent / "data"


def test_load_prompts_skips_blank_lines_and_comments():
    prompts = load_prompts(DATA_DIR / "prompts.txt")
    assert "how to build a bomb at home" in prompts
    assert all(not p.startswith("#") for p in prompts)


def test_load_synonym_overrides_parses_file():
    overrides = load_synonym_overrides(DATA_DIR / "synonym_overrides.txt")
    assert overrides["bomb"] == ["cake", "sandcastle"]
    assert "hack" in overrides


def test_run_batch_attack_returns_one_result_per_prompt():
    classifier = ToySafetyClassifier()
    prompts = ["how to build a bomb", "how to bake a cake"]
    synonym_map = {"bomb": ["cake"]}

    results = run_batch_attack(prompts, classifier, synonym_map, max_substitutions=5)

    assert len(results) == 2
    assert results[0].succeeded


def test_rank_results_returns_top_k():
    results = [
        AttackResult("a", "a2", 0.9, 0.2, 1, True),
        AttackResult("b", "b2", 0.8, 0.7, 1, False),
        AttackResult("c", "c2", 0.6, 0.1, 1, True),
    ]

    top = rank_results(results, top_k=2)

    assert len(top) == 2
