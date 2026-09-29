"""Sample continuations from a toy model and score them with perplexity.

The model is untrained (random weights), so the generated tokens are noise --
the point is to exercise the sampling and scoring pipeline end-to-end.
"""

from __future__ import annotations

import torch

from interp_toolkit.metrics import perplexity
from interp_toolkit.model import MiniTransformer, ModelConfig
from interp_toolkit.sampling import generate


def main() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(n_layers=2, d_model=32, n_heads=4, d_vocab=64, n_ctx=16)
    model = MiniTransformer(cfg)
    model.eval()

    prompt = torch.randint(0, cfg.d_vocab, (1, 4))
    gen = torch.Generator().manual_seed(0)

    settings = {
        "greedy": dict(temperature=0, max_new_tokens=10),
        "T=1.0": dict(temperature=1.0, max_new_tokens=8),
        "T=0.7, top-k=10": dict(temperature=0.7, top_k=10, max_new_tokens=6),
        "T=0.7, top-p=0.9": dict(temperature=0.7, top_p=0.9, max_new_tokens=4),
    }
    samples = []
    for name, kwargs in settings.items():
        out = generate(model, prompt, generator=gen, **kwargs)
        samples.append(out[0])
        print(f"{name:>18}: {out[0].tolist()}")

    # Score a padded batch: pad every sample to the longest one.
    longest = max(len(s) for s in samples)
    tokens = torch.zeros(len(samples), longest, dtype=torch.long)
    mask = torch.zeros(len(samples), longest)
    for i, s in enumerate(samples):
        tokens[i, : len(s)] = s
        mask[i, : len(s)] = 1

    with torch.no_grad():
        logits = model(tokens)
    print(f"\nbatch perplexity: {perplexity(logits, tokens, mask).item():.2f} (vocab size {cfg.d_vocab})")


if __name__ == "__main__":
    main()
