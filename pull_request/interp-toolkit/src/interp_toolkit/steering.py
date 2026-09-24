"""Contrastive activation steering: extract a "concept direction" from
contrastive prompt pairs and use it to steer a forward pass."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from interp_toolkit.hooks import run_with_cache, run_with_patch


def compute_steering_vector(
    model: nn.Module,
    positive_tokens: torch.Tensor,
    negative_tokens: torch.Tensor,
    hook_name: str,
) -> torch.Tensor:
    """Compute a steering direction from contrastive prompt pairs.

    `positive_tokens` / `negative_tokens` each have shape [n_examples, seq_len].
    Returns a direction of shape [d_model], pointing from "negative" towards
    "positive": adding it (scaled) to an activation should push the model's
    behavior towards whatever the positive examples have in common.
    """
    _, positive_cache = run_with_cache(model, positive_tokens)
    _, negative_cache = run_with_cache(model, negative_tokens)

    positive_mean = positive_cache[hook_name].mean(dim=(0, 1))
    negative_mean = negative_cache[hook_name].mean(dim=(0, 1))

    return negative_mean - positive_mean


def generate_with_steering(
    model: nn.Module,
    tokens: torch.Tensor,
    hook_name: str,
    steering_vector: torch.Tensor,
    coefficient: float = 1.0,
) -> torch.Tensor:
    """Run a forward pass with `coefficient * steering_vector` added to the
    activation at `hook_name`."""
    _, cache = run_with_cache(model, tokens)
    steered_activation = cache[hook_name] + coefficient * steering_vector
    return run_with_patch(model, tokens, {hook_name: steered_activation})


def steering_sweep(
    model: nn.Module,
    tokens: torch.Tensor,
    hook_name: str,
    steering_vector: torch.Tensor,
    coefficients: list[float],
) -> dict[float, torch.Tensor]:
    """Run `generate_with_steering` once per coefficient in `coefficients`."""
    return {
        c: generate_with_steering(model, tokens, hook_name, steering_vector, c) for c in coefficients
    }


def save_steering_vector(
    vector: torch.Tensor, label: str, cache_dir: str | Path = "steering_cache"
) -> Path:
    """Save a steering vector to `{cache_dir}/{label}.pt`."""
    path = Path(f"{cache_dir}/{label}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vector, path)


def load_steering_vector(label: str, cache_dir: str | Path = "steering_cache") -> torch.Tensor:
    """Load a steering vector previously saved with `save_steering_vector`."""
    path = Path(f"{cache_dir}/{label}.pt")
    return torch.load(path)
