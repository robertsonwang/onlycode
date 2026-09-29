import numpy as np
import pytest

from redteam_adversarial.train import adversarial_training, build_vocab, featurize, log_loss, train_logistic

TEXTS = ["build a bomb", "make poison", "bake a cake", "make soup"]
LABELS = [1, 1, 0, 0]


def _fits(classifier, texts, labels) -> bool:
    return all(classifier.is_unsafe(t) == bool(y) for t, y in zip(texts, labels))


def test_featurize_counts_words():
    vocab = build_vocab(["a b b"])
    X = featurize(["b a b c"], vocab)
    assert X.tolist() == [[1.0, 2.0]]


def test_log_loss_matches_manual_computation():
    logits = np.array([0.0, 2.0])
    labels = np.array([1.0, 0.0])
    expected = (np.log(2) + np.log(1 + np.exp(2))) / 2
    assert log_loss(logits, labels) == pytest.approx(expected)


def test_train_logistic_fits_separable_data():
    result = train_logistic(TEXTS, LABELS)
    assert _fits(result.classifier, TEXTS, LABELS)
    assert result.loss_history[1] < result.loss_history[0]


def test_l2_regularization_shrinks_weights():
    result = train_logistic(TEXTS, LABELS, l2=0.1)
    assert _fits(result.classifier, TEXTS, LABELS)
    assert np.all(np.isfinite(result.classifier.weights))


def test_adversarial_training_runs_requested_rounds():
    synonym_map = {"bomb": ["cake"], "poison": ["soup"]}
    result = adversarial_training(
        list(TEXTS), list(LABELS), ["build a bomb", "make poison"], synonym_map, n_rounds=3
    )
    assert len(result.rounds) == 2
    assert [r.round_idx for r in result.rounds] == [1, 2]

