from redteam_adversarial.search import greedy_word_substitution_attack
from redteam_adversarial.target import ToySafetyClassifier


def test_attack_finds_evasion_when_synonym_available():
    clf = ToySafetyClassifier()
    text = "how to build a bomb"
    synonym_map = {"bomb": ["cake", "sandcastle"]}

    result = greedy_word_substitution_attack(clf, text, synonym_map, threshold=0.5)

    assert result.succeeded
    assert result.n_substitutions == 1
    assert "bomb" not in result.adversarial_text
    assert result.final_score < result.original_score


def test_attack_fails_when_no_synonyms_available():
    clf = ToySafetyClassifier()
    text = "how to build a bomb"

    result = greedy_word_substitution_attack(clf, text, synonym_map={}, threshold=0.5)

    assert not result.succeeded
    assert result.n_substitutions == 0
    assert result.adversarial_text == text


def test_attack_stops_early_once_evaded():
    clf = ToySafetyClassifier()
    text = "bomb"
    synonym_map = {"bomb": ["harmless", "kill", "cake"]}

    result = greedy_word_substitution_attack(clf, text, synonym_map, threshold=0.5, max_substitutions=5)

    assert result.succeeded
    assert result.n_substitutions == 1
    assert result.adversarial_text == "harmless"


def test_attack_respects_max_substitutions_budget():
    clf = ToySafetyClassifier()
    text = "bomb weapon poison"
    # Only mild synonyms available: not enough to cross the threshold in one step.
    synonym_map = {
        "bomb": ["device"],
        "weapon": ["item"],
        "poison": ["substance"],
    }

    result = greedy_word_substitution_attack(clf, text, synonym_map, threshold=0.01, max_substitutions=1)

    assert result.n_substitutions <= 1
