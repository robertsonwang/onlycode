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
- `src/interp_toolkit/probing.py` -- linear probing: `train_linear_probe` /
  `evaluate_probe` (fit and score a probe on cached activations),
  `label_examples_by_keyword_pattern` (quick regex-based labeling), and
  `probe_accuracy_by_layer` (probe every layer at once).
- `examples/run_patching.py` -- an end-to-end activation-patching sweep
  across layers.
- `examples/run_probing.py` -- label a few texts by keyword and probe
  each layer's residual stream for that label.

## Usage

```bash
uv sync
uv run pytest
uv run python examples/run_patching.py
uv run python examples/run_probing.py
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
