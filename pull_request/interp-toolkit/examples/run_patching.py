"""Toy activation-patching experiment.

We construct a "clean" and "corrupted" input that differ at one token
position, then patch each layer's residual stream from the clean run into
the corrupted run to see which layer's activation is most responsible for
recovering the clean output.

This model is untrained (random weights), so the numbers aren't meaningful
on their own -- the point is to exercise the caching/patching pipeline
end-to-end.
"""

from __future__ import annotations

import torch

from interp_toolkit.hooks import run_with_cache, run_with_patch
from interp_toolkit.metrics import logit_diff, patching_effect
from interp_toolkit.model import MiniTransformer, ModelConfig


def main() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(n_layers=3, d_model=32, n_heads=4, d_vocab=64, n_ctx=16)
    model = MiniTransformer(cfg)
    model.eval()

    seq_len = 6
    clean_tokens = torch.randint(0, cfg.d_vocab, (1, seq_len))
    corrupted_tokens = clean_tokens.clone()
    corrupted_tokens[0, 2] = (corrupted_tokens[0, 2] + 1) % cfg.d_vocab

    correct_token, incorrect_token = 5, 9

    with torch.no_grad():
        clean_logits, clean_cache = run_with_cache(model, clean_tokens)
        corrupted_logits, _ = run_with_cache(model, corrupted_tokens)

        clean_diff = logit_diff(clean_logits, correct_token, incorrect_token)
        corrupted_diff = logit_diff(corrupted_logits, correct_token, incorrect_token)

        print(f"clean logit diff:     {clean_diff.item():.4f}")
        print(f"corrupted logit diff: {corrupted_diff.item():.4f}")
        print()

        for layer in range(cfg.n_layers):
            hook_name = f"blocks.{layer}.hook_resid_post"
            patched_logits = run_with_patch(
                model, corrupted_tokens, {hook_name: clean_cache[hook_name]}
            )
            patched_diff = logit_diff(patched_logits, correct_token, incorrect_token)
            effect = patching_effect(clean_diff, patched_diff, corrupted_diff)
            print(f"layer {layer} ({hook_name}): recovery = {effect.item():.2%}")


if __name__ == "__main__":
    main()
