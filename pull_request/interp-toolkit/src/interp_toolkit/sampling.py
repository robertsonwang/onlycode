"""Next-token sampling from final-position logits.

`sample_next_token` turns logits into a token id, with the usual knobs:
temperature, top-k, and top-p (nucleus) filtering.
"""

from __future__ import annotations

import torch


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

