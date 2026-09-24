"""Demo: direct logit attribution + a layer ablation sweep, with the
activation cache round-tripped through disk in between."""

from __future__ import annotations

import torch

from interp_toolkit.attribution import layer_ablation_sweep, per_layer_logit_attribution
from interp_toolkit.cache_io import load_cache, save_cache
from interp_toolkit.hooks import run_with_cache
from interp_toolkit.metrics import logit_diff
from interp_toolkit.model import MiniTransformer, ModelConfig

CACHE_PATH = "/tmp/interp_toolkit_demo_cache.pt"


def main() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(n_layers=3, d_model=32, n_heads=4, d_vocab=64, n_ctx=16)
    model = MiniTransformer(cfg)
    model.eval()

    tokens = torch.randint(0, cfg.d_vocab, (1, 6))
    correct_token, incorrect_token = 5, 9

    with torch.no_grad():
        logits, cache = run_with_cache(model, tokens)
        total_diff = logit_diff(logits, correct_token, incorrect_token)
        print(f"total logit diff: {total_diff.item():.4f}")

        save_cache(cache, CACHE_PATH)
        reloaded_cache = load_cache(CACHE_PATH)

        attribution = per_layer_logit_attribution(model, reloaded_cache, correct_token, incorrect_token)
        print(f"per-layer attribution: {attribution.squeeze().tolist()}")
        print(f"attribution sum: {attribution.sum().item():.4f} (should be close to total logit diff)")

        sweep = layer_ablation_sweep(model, tokens, correct_token, incorrect_token)
        print(f"ablation sweep logit diffs: {sweep.squeeze().tolist()}")


if __name__ == "__main__":
    main()
