"""A minimal sparse autoencoder (SAE) for decomposing residual-stream
activations into a larger, sparser dictionary of "features"."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """Encode activations into a sparse, overcomplete feature basis and
    reconstruct them: features = ReLU(encoder(x)), reconstruction = decoder(features).
    """

    dead_feature_counts: dict[int, int] = {}

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.W_enc = nn.Parameter(torch.randn(d_model, d_hidden) * 0.1)
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.randn(d_hidden, d_model) * 0.1)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Project activations into sparse feature space."""
        return F.relu(x @ self.W_enc + self.b_enc)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Reconstruct activations from feature space."""
        return features @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features

    def track_dead_features(self, features: torch.Tensor) -> None:
        """Record, per feature index, how many batches it failed to activate on."""
        active = (features > 0).any(dim=0)
        for i in range(self.d_hidden):
            if not active[i].item():
                self.dead_feature_counts[i] = self.dead_feature_counts.get(i, 0) + 1


def sae_loss(
    x: torch.Tensor,
    reconstruction: torch.Tensor,
    features: torch.Tensor,
    l1_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total_loss, reconstruction_loss, sparsity_loss)."""
    recon_loss = ((reconstruction - x) ** 2).mean()
    sparsity_loss = features.mean()
    total_loss = recon_loss + l1_coefficient * sparsity_loss
    return total_loss, recon_loss


def top_activating_examples(
    sae: SparseAutoencoder,
    texts: list[str],
    activations: torch.Tensor,
    feature_idx: int,
    k: int = 3,
) -> list[tuple[str, float]]:
    """Return the `k` texts whose activation most strongly activates `feature_idx`."""
    with torch.no_grad():
        features = sae.encode(activations)
    values = features[:, feature_idx]
    top_k = torch.topk(values, k=min(k, len(texts)))
    return [(texts[i], values[i].item()) for i in top_k.indices]


def export_feature_dashboard_html(
    top_examples: list[tuple[str, float]], feature_idx: int, path: str | Path
) -> None:
    """Write a simple HTML report showing the top activating examples for one feature."""
    rows = "\n".join(
        f"<tr><td>{text}</td><td>{activation:.3f}</td></tr>" for text, activation in top_examples
    )
    html = f"""<html><body>
<h1>Feature {feature_idx}</h1>
<table>{rows}</table>
</body></html>"""
    Path(path).write_text(html, encoding="utf-8")
