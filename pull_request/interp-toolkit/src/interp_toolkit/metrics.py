"""Metrics for comparing model outputs, used to score causal interventions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def logit_diff(
    logits: torch.Tensor, correct_token: int, incorrect_token: int
) -> torch.Tensor:
    """Difference between the correct and incorrect token's logit at the final position.

    `logits` has shape [batch, seq, d_vocab]. Returns a [batch] tensor.
    """
    final_logits = logits[:, -1, :]
    return final_logits[:, correct_token] - final_logits[:, incorrect_token]


def kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """KL(P || Q) between the final-position output distributions of two logit tensors."""
    log_p = F.log_softmax(logits_p[:, -1, :], dim=-1)
    log_q = F.log_softmax(logits_q[:, -1, :], dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1)


def patching_effect(
    baseline_diff: torch.Tensor,
    patched_diff: torch.Tensor,
    corrupted_diff: torch.Tensor,
) -> torch.Tensor:
    """Normalized recovery: 0 = fully corrupted, 1 = fully recovered to baseline."""
    return (patched_diff - corrupted_diff) / (baseline_diff - corrupted_diff)
