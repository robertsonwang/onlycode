import pytest

from redteam_adversarial.beam import beam_search_attack
from redteam_adversarial.search import greedy_word_substitution_attack
from redteam_adversarial.target import ToySafetyClassifier


def test_beam_finds_evasion_when_synonym_available():
    clf = ToySafetyClassifier()
    synonym_map = {"bomb": ["cake", "sandcastle"]}

    result = beam_search_attack(clf, "how to build a bomb", synonym_map)

    assert result.succeeded
    assert result.n_substitutions == 1
    assert "bomb" not in result.adversarial_text


def test_beam_width_one_matches_greedy():
    clf = ToySafetyClassifier()
    text = "hack the weapon and poison it"
    synonym_map = {"hack": ["access"], "weapon": ["tool"], "poison": ["season"]}

    beam = beam_search_attack(clf, text, synonym_map, beam_width=1, threshold=0.2)
    greedy = greedy_word_substitution_attack(clf, text, synonym_map, threshold=0.2)

    assert beam.adversarial_text == greedy.adversarial_text
    assert beam.n_substitutions == greedy.n_substitutions


def test_beam_escapes_greedy_dead_end():
    clf = ToySafetyClassifier()
    # Greedy takes the bigger first drop (bomb -> hack) and then has nowhere
    # to go; beam also keeps bomb -> weapon, which leads on to "tool".
    synonym_map = {"bomb": ["weapon", "hack"], "weapon": ["tool"]}

    greedy = greedy_word_substitution_attack(clf, "bomb", synonym_map)
    beam = beam_search_attack(clf, "bomb", synonym_map, beam_width=2)

    assert not greedy.succeeded
    assert beam.succeeded
    assert beam.adversarial_text == "tool"
    assert beam.n_substitutions == 2


def test_beam_respects_max_substitutions_budget():
    clf = ToySafetyClassifier()
    synonym_map = {"bomb": ["device"], "weapon": ["item"], "poison": ["substance"]}

    result = beam_search_attack(
        clf, "bomb weapon poison", synonym_map, threshold=0.01, max_substitutions=1
    )

    assert result.n_substitutions <= 1


def test_beam_rejects_nonpositive_width():
    with pytest.raises(ValueError):
        beam_search_attack(ToySafetyClassifier(), "bomb", {}, beam_width=0)
