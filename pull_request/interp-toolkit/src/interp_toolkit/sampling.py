"""Autoregressive sampling from a `MiniTransformer`.

`sample_next_token` turns final-position logits into a token id, with the
usual knobs: temperature, top-k, and top-p (nucleus) filtering. `generate`
calls it in a loop, appending one token per step.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def softmax_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Softmax over the last dimension of `logits / temperature`.

    Filtered-out entries (logit = -inf) get probability exactly 0.
    """
    exp = torch.exp(logits / temperature)
    return exp / exp.sum(dim=-1, keepdim=True)


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Set every logit outside the `k` largest (per row) to -inf."""
    kth_largest = torch.topk(logits, k, dim=-1).values[:, -1]
    return logits.masked_fill(logits < kth_largest, float("-inf"))


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of highest-probability tokens
    whose cumulative probability is at least `top_p`; set the rest to -inf.
    """
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumulative = softmax_with_temperature(sorted_logits).cumsum(dim=-1)
    remove = cumulative > top_p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token id per row from `logits` of shape [batch, d_vocab].

    `temperature=0` means greedy decoding. Returns a [batch] tensor.
    """
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    probs = softmax_with_temperature(logits)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Extend `prompt` ([batch, seq]) by `max_new_tokens` sampled tokens.

    Returns a [batch, seq + max_new_tokens] tensor that starts with `prompt`.
    """
    tokens = prompt
    for _ in range(max_new_tokens):
        logits = model(tokens)[:, -1, :]
        next_token = sample_next_token(
            logits, temperature=temperature, top_k=top_k, top_p=top_p, generator=generator
        )
        tokens = torch.cat([tokens, next_token[:, None]], dim=1)
    return tokens
