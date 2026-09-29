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
  `patching_effect`, plus `sequence_log_probs` and `perplexity` (with an
  optional padding mask).
- `src/interp_toolkit/sampling.py` -- `sample_next_token` (temperature,
  top-k, top-p) and `generate` for autoregressive sampling.
- `examples/run_patching.py` -- an end-to-end activation-patching sweep
  across layers.

## Usage

```bash
uv sync
uv run pytest
uv run python examples/run_patching.py
uv run python examples/run_generation.py
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

### Sampling and perplexity

```python
from interp_toolkit import generate, perplexity

out = generate(model, prompt, max_new_tokens=8, temperature=0.8, top_k=10, top_p=0.9)
ppl = perplexity(model(out), out)
```
