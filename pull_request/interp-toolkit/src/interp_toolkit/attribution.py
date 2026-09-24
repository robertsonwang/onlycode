"""Direct logit attribution and ablation sweeps for a MiniTransformer."""

from __future__ import annotations

import torch
import torch.nn as nn

from interp_toolkit.hooks import run_with_cache, run_with_patch
from interp_toolkit.metrics import logit_diff


def per_layer_logit_attribution(
    model: nn.Module,
    cache: dict[str, torch.Tensor],
    correct_token: int,
    incorrect_token: int,
) -> torch.Tensor:
    """Decompose the final logit difference into each layer's contribution.

    Each block's contribution is `hook_resid_post - hook_resid_pre` for that
    block at the final sequence position, projected onto the unembedding
    direction for `correct_token - incorrect_token`. The returned values
    should sum to (approximately) the total `logit_diff` at the final
    position. Returns a tensor of shape [n_layers].
    """
    direction = model.unembed.weight[correct_token] - model.unembed.weight[incorrect_token]

    contributions = []
    for i in range(1, model.cfg.n_layers):
        pre = cache[f"blocks.{i}.hook_resid_pre"][:, -1, :]
        post = cache[f"blocks.{i}.hook_resid_post"][:, -1, :]
        delta = post - pre
        contributions.append((delta * direction).sum(dim=-1))

    return torch.stack(contributions)


def layer_ablation_sweep(
    model: nn.Module,
    tokens: torch.Tensor,
    correct_token: int,
    incorrect_token: int,
) -> torch.Tensor:
    """Mean-ablate each layer's residual-stream write in turn and measure the
    resulting change in `logit_diff(correct_token, incorrect_token)`.

    Returns a tensor of shape [n_layers].
    """
    _, cache = run_with_cache(model, tokens)

    results = []
    for i in range(model.cfg.n_layers):
        hook_name = f"blocks.{i}.hook_resid_post"
        activation = cache[hook_name]
        mean_activation = activation.mean(dim=0, keepdim=True).expand_as(activation)
        patched_logits = run_with_patch(model, tokens, {hook_name: mean_activation})
        results.append(logit_diff(patched_logits, correct_token, incorrect_token))

    return torch.stack(results)
