"""Learned safety classifier + adversarial training.

`ToySafetyClassifier` uses hand-picked word weights. Here we instead fit a
bag-of-words logistic regression on labeled prompts, then harden it with
adversarial training: attack the current classifier, add the successful
adversarial prompts to the training set (labeled unsafe), and retrain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from redteam_adversarial.metrics import attack_success_rate
from redteam_adversarial.search import greedy_word_substitution_attack
from redteam_adversarial.target import tokenize


def build_vocab(texts: list[str]) -> dict[str, int]:
    words = sorted({tok for text in texts for tok in tokenize(text)})
    return {word: i for i, word in enumerate(words)}


def featurize(texts: list[str], vocab: dict[str, int]) -> np.ndarray:
    """Bag-of-words count matrix of shape (len(texts), len(vocab))."""
    X = np.zeros((len(texts), len(vocab)))
    for i, text in enumerate(texts):
        for tok in tokenize(text):
            j = vocab.get(tok)
            if j is not None:
                X[i, j] += 1
    return X


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def log_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean binary cross-entropy of `labels` (0/1) under P(unsafe) = sigmoid(logits)."""
    p = sigmoid(logits)
    return float(np.mean(-(labels * np.log(p) + (1 - labels) * np.log(1 - p))))


@dataclass
class LogisticSafetyClassifier:
    """Drop-in replacement for `ToySafetyClassifier` with learned weights."""

    vocab: dict[str, int]
    weights: np.ndarray
    bias: float = 0.0

    def logits(self, texts: list[str]) -> np.ndarray:
        return featurize(texts, self.vocab) @ self.weights + self.bias

    def unsafe_probability(self, text: str) -> float:
        return float(sigmoid(self.logits([text]))[0])

    def is_unsafe(self, text: str, threshold: float = 0.5) -> bool:
        return self.unsafe_probability(text) >= threshold


@dataclass
class TrainResult:
    classifier: LogisticSafetyClassifier
    loss_history: list[float]


def train_logistic(
    texts: list[str],
    labels: list[int],
    vocab: dict[str, int] | None = None,
    lr: float = 0.5,
    epochs: int = 2000,
    l2: float = 1e-2,
    patience: int = 20,
) -> TrainResult:
    """Fit L2-regularized logistic regression with full-batch gradient descent.

    Stops early once the training loss hasn't improved for `patience`
    epochs, and returns the weights from the best epoch.
    """
    if vocab is None:
        vocab = build_vocab(texts)
    X = featurize(texts, vocab)
    y = np.asarray(labels, dtype=float)

    w = np.zeros(X.shape[1])
    b = 0.0
    best_loss, best_w, best_b = float("inf"), w.copy(), b
    history: list[float] = []
    epochs_since_best = 0

    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        grad_w = X.T @ (p - y) / len(y) - l2 * w
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b

        loss = log_loss(X @ w + b, y)
        history.append(loss)
        if loss < best_loss:
            best_loss, best_w, best_b = loss, w.copy(), b
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                break

    return TrainResult(LogisticSafetyClassifier(vocab, best_w, best_b), history)


@dataclass(frozen=True)
class RoundStats:
    round_idx: int
    attack_success_rate: float
    n_adversarial_added: int


@dataclass
class AdversarialTrainingResult:
    classifier: LogisticSafetyClassifier
    rounds: list[RoundStats]


def adversarial_training(
    texts: list[str],
    labels: list[int],
    attack_prompts: list[str],
    synonym_map: dict[str, list[str]],
    n_rounds: int = 3,
    threshold: float = 0.5,
    **train_kwargs,
) -> AdversarialTrainingResult:
    """Alternate attacking and retraining for `n_rounds` rounds.

    Each round runs the greedy substitution attack against the current
    classifier on every prompt in `attack_prompts`, adds the adversarial
    prompts that evaded it to the training set with label 1, and retrains.
    """
    # Include substitute words in the vocab so the classifier can learn weights for them.
    substitutes = [word for subs in synonym_map.values() for word in subs]
    vocab = build_vocab(texts + substitutes)
    classifier = train_logistic(texts, labels, vocab=vocab, **train_kwargs).classifier

    rounds: list[RoundStats] = []
    for round_idx in range(1, n_rounds):
        results = [
            greedy_word_substitution_attack(classifier, prompt, synonym_map, threshold=threshold)
            for prompt in attack_prompts
        ]
        adversarial = [r.adversarial_text for r in results if r.succeeded]
        rounds.append(RoundStats(round_idx, attack_success_rate(results), len(adversarial)))
        if not adversarial:
            break

        texts += adversarial
        labels += [1] * len(adversarial)
        classifier = train_logistic(texts, labels, vocab=vocab, **train_kwargs).classifier

    return AdversarialTrainingResult(classifier, rounds)
