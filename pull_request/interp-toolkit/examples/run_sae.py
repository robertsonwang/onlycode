"""Demo: train a tiny SAE on cached residual-stream activations, then
export a feature dashboard showing each feature's top activating examples."""

from __future__ import annotations

import torch

from interp_toolkit.hooks import run_with_cache
from interp_toolkit.model import MiniTransformer, ModelConfig
from interp_toolkit.sae import SparseAutoencoder, export_feature_dashboard_html, sae_loss, top_activating_examples

TEXTS = [
    "the cat sat on the mat",
    "quantum entanglement is weird",
    "the dog ran in the park",
    "general relativity bends spacetime",
    "a kitten chased the yarn",
    "black holes warp light",
    "puppies love to play fetch",
    "the double slit experiment",
]

DASHBOARD_PATH = "/tmp/interp_toolkit_sae_dashboard.html"


def main() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(n_layers=3, d_model=32, n_heads=4, d_vocab=64, n_ctx=16)
    model = MiniTransformer(cfg)
    model.eval()

    tokens = torch.randint(0, cfg.d_vocab, (len(TEXTS), 6))
    with torch.no_grad():
        _, cache = run_with_cache(model, tokens)
    activations = cache["blocks.1.hook_resid_post"][:, -1, :]

    sae = SparseAutoencoder(d_model=cfg.d_model, d_hidden=64)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-2)

    for step in range(100):
        optimizer.zero_grad()
        reconstruction, features = sae(activations)
        total_loss, recon_loss = sae_loss(activations, reconstruction, features, l1_coefficient=1e-3)
        total_loss.backward()
        optimizer.step()
        sae.track_dead_features(features)
        if step % 25 == 0:
            print(f"step {step}: total_loss={total_loss.item():.4f} recon_loss={recon_loss.item():.4f}")

    print(f"dead feature count: {len(sae.dead_feature_counts)} / {sae.d_hidden}")

    feature_idx = 4
    top = top_activating_examples(sae, TEXTS, activations, feature_idx=feature_idx, k=3)
    for text, value in top:
        print(f"  feature {feature_idx}: {value:.3f} -- {text!r}")

    export_feature_dashboard_html(top, feature_idx=feature_idx, path=DASHBOARD_PATH)
    print(f"wrote dashboard to {DASHBOARD_PATH}")


if __name__ == "__main__":
    main()
