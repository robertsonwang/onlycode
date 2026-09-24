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
- `src/interp_toolkit/steering.py` -- contrastive activation steering:
  `compute_steering_vector` (extract a direction from contrastive prompt
  pairs), `generate_with_steering` / `steering_sweep` (apply it, at one
  coefficient or several), and `save_steering_vector` / `load_steering_vector`
  to persist a named direction to disk.
- `examples/run_patching.py` -- an end-to-end activation-patching sweep
  across layers.
- `examples/run_steering.py` -- extract a steering direction, round-trip
  it through disk, and sweep coefficients.

## Usage

```bash
uv sync
uv run pytest
uv run python examples/run_patching.py
uv run python examples/run_steering.py
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
