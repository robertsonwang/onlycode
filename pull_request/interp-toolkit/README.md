# interp-toolkit

A minimal, from-scratch toolkit for mechanistic-interpretability experiments
on a toy decoder-only transformer: activation caching, activation patching,
and a couple of standard comparison metrics (logit difference, KL divergence).

Everything here is CPU-only and intentionally tiny (a few thousand
parameters) so experiments run in milliseconds -- the goal is to exercise the
causal-intervention *plumbing*, not to train a real model.

## Layout

- `src/interp_toolkit/model.py` -- a small transformer (`MiniTransformer`)
  with named hook points at every attention/MLP/residual-stream location.
- `src/interp_toolkit/hooks.py` -- `run_with_cache` (record every hook
  point's activation) and `run_with_patch` (overwrite specific hook points
  during a forward pass).
- `src/interp_toolkit/metrics.py` -- `logit_diff`, `kl_divergence`,
  `patching_effect`.
- `src/interp_toolkit/sae.py` -- a minimal sparse autoencoder:
  `SparseAutoencoder` (encode/decode/forward), `sae_loss`,
  `top_activating_examples`, and `export_feature_dashboard_html` for
  inspecting what a feature responds to.
- `examples/run_patching.py` -- an end-to-end activation-patching sweep
  across layers.
- `examples/run_sae.py` -- train a tiny SAE on cached activations and
  export a feature dashboard.

## Usage

```bash
uv sync
uv run pytest
uv run python examples/run_patching.py
uv run python examples/run_sae.py
```

```python
from interp_toolkit.model import MiniTransformer, ModelConfig
from interp_toolkit.hooks import run_with_cache, run_with_patch

model = MiniTransformer(ModelConfig())
logits, cache = run_with_cache(model, tokens)
patched_logits = run_with_patch(model, tokens, {"blocks.0.hook_resid_post": some_tensor})
```

Hook names follow `blocks.{layer}.<point>`, e.g. `blocks.0.hook_resid_pre`,
`blocks.1.attn.hook_pattern`, `blocks.1.mlp.hook_post`.
