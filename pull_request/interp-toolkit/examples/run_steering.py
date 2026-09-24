"""Demo: extract a steering direction from contrastive prompt pairs, save
and reload it, then sweep a few coefficients and compare logit diffs."""

from __future__ import annotations

import torch

from interp_toolkit.hooks import run_with_cache
from interp_toolkit.metrics import logit_diff
from interp_toolkit.model import MiniTransformer, ModelConfig
from interp_toolkit.steering import (
    compute_steering_vector,
    load_steering_vector,
    save_steering_vector,
    steering_sweep,
)

HOOK_NAME = "blocks.1.hook_resid_post"


def main() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(n_layers=3, d_model=32, n_heads=4, d_vocab=64, n_ctx=16)
    model = MiniTransformer(cfg)
    model.eval()

    # Two small sets of contrastive prompts -- everything but the final
    # token is shared boilerplate, only the last token differs.
    positive_tokens = torch.stack(
        [torch.tensor([1, 2, 3, 4, 5, 10]) for _ in range(4)]
    )
    negative_tokens = torch.stack(
        [torch.tensor([1, 2, 3, 4, 5, 20]) for _ in range(4)]
    )

    with torch.no_grad():
        direction = compute_steering_vector(model, positive_tokens, negative_tokens, HOOK_NAME)

        save_steering_vector(direction, label="demo-concept")
        reloaded_direction = load_steering_vector(label="demo-concept")

        tokens = torch.tensor([[1, 2, 3, 4, 5, 20]])
        correct_token, incorrect_token = 10, 20

        base_logits, _ = run_with_cache(model, tokens)
        print(f"unsteered logit diff: {logit_diff(base_logits, correct_token, incorrect_token).item():.4f}")

        sweep = steering_sweep(model, tokens, HOOK_NAME, reloaded_direction, coefficients=[0.0, 2.0, 5.0])
        for coefficient, logits in sweep.items():
            diff = logit_diff(logits, correct_token, incorrect_token)
            print(f"coefficient={coefficient}: logit diff={diff.item():.4f}")


if __name__ == "__main__":
    main()
