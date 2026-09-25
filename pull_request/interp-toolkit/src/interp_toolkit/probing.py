"""Linear probes: fit a simple classifier on cached activations to test
whether a layer's representation linearly encodes a concept."""

from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from interp_toolkit.hooks import run_with_cache


def train_linear_probe(
    activations: torch.Tensor,
    labels: torch.Tensor,
    n_steps: int = 200,
    lr: float = 0.1,
) -> nn.Linear:
    """Fit a logistic-regression probe: activations [n, d_model] -> labels [n] in {0, 1}."""
    d_model = activations.shape[-1]
    probe = nn.Linear(d_model, 1)

    for _ in range(n_steps):
        logits = probe(activations).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        loss.backward()
        with torch.no_grad():
            for param in probe.parameters():
                param -= lr * param.grad

    return probe


def evaluate_probe(probe: nn.Linear, activations: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return the probe's classification accuracy on (activations, labels)."""
    with torch.no_grad():
        predictions = (probe(activations).squeeze(-1) > 0).long()
        accuracy = (predictions == labels).float().mean()
    return accuracy.item()


def label_examples_by_keyword_pattern(texts: list[str], pattern: str) -> torch.Tensor:
    """Label each text 1 if `pattern` (a regex) matches, else 0."""
    return torch.tensor([1 if re.search(pattern, text) else 0 for text in texts])


def probe_accuracy_by_layer(
    model: nn.Module,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    layers: list[int],
) -> dict[int, float]:
    """Train and evaluate a linear probe at each layer in `layers`.

    For each layer, extracts the residual-stream activation at the final
    sequence position, fits a probe, and reports its classification
    accuracy, as a rough proxy for how linearly decodable the concept is
    at that layer.
    """
    _, cache = run_with_cache(model, tokens)

    accuracies = {}
    for layer in layers:
        activation = cache[f"blocks.{layer}.hook_resid_post"][:, 0, :]
        probe = train_linear_probe(activation, labels)
        accuracies[layer] = evaluate_probe(probe, activation, labels)
    return accuracies
