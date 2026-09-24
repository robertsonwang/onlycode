# Answer key — PR: "Add direct logit attribution, ablation sweep, cache I/O"

Branch: `rw/interp_toolkit_1`. Diff adds `src/interp_toolkit/attribution.py`
(`per_layer_logit_attribution`, `layer_ablation_sweep`),
`src/interp_toolkit/cache_io.py` (`save_cache`, `load_cache`),
`examples/run_attribution.py`, `tests/test_attribution.py`,
`tests/test_cache_io.py`, and touches `README.md` / `__init__.py` for wiring.

Excluded from chat during review. Grade the reviewer's findings against
this list; any real, defensible finding not listed here should still count
in their favor.

## Planted issues (5)

### 1. ML-specific pitfall (domain-specific) — attribution ignores the final LayerNorm
**File:** `src/interp_toolkit/attribution.py`, `per_layer_logit_attribution`.
```python
direction = model.unembed.weight[correct_token] - model.unembed.weight[incorrect_token]
...
delta = post - pre
contributions.append((delta * direction).sum(dim=-1))
```
**Why it's a bug:** this is the single most common, most-discussed footgun
in direct logit attribution (DLA): the residual stream is not in "logit
space" until it passes through `model.ln_final`, which per-token
mean-centers, divides by std, and rescales. Projecting a raw residual
delta straight onto an unembedding direction, skipping that normalization,
gives numbers that are systematically wrong and — crucially — don't
actually decompose the true logit difference. The docstring even states
the standard sanity check practitioners use to validate a DLA
implementation ("the returned values should sum to ... the total logit
diff"), and `examples/run_attribution.py` prints exactly that comparison —
run it and `attribution.sum()` is nowhere close to `total_diff` (0.03 vs.
1.08 in a typical run). No test checks this invariant, so it ships
looking plausible. A correct version needs to fold in `ln_final`'s
per-token scale (roughly: divide the delta by the residual stream's std at
that position before projecting, and apply `ln_final.weight`) before
projecting onto `direction`.
**Severity:** high — this is the kind of bug that produces a plausible-
looking bar chart with a wrong story, which is exactly the failure mode
interpretability tooling most needs to avoid.

### 2. Correctness — off-by-one drops layer 0 from the attribution
**File:** `src/interp_toolkit/attribution.py`, `per_layer_logit_attribution`.
```python
for i in range(1, model.cfg.n_layers):
```
**Why it's a bug:** starts at index 1, so layer 0's contribution is never
computed, and the returned tensor has `n_layers - 1` entries — worse, the
entry at position 0 of the result is actually layer *1*'s contribution,
silently shifting every index by one relative to what a caller would
reasonably assume from "index i = layer i." `test_per_layer_logit_attribution_returns_a_value_per_layer`
only checks `attribution.shape[-1] == 1` (the batch dim), never the
number of layers, so this ships invisibly. Should be `range(model.cfg.n_layers)`.
**Severity:** high — silently wrong output shape and silently
mislabeled indices are both bad; together, worse.

### 3. Design smell (interp-specific) — ablation sweep doesn't wrap forward passes in `torch.no_grad()`
**File:** `src/interp_toolkit/attribution.py`, `layer_ablation_sweep`.
**Why it's a bug:** this function runs `n_layers` forward passes purely
for inference (measuring a metric, not training), but never disables
autograd. Every `run_with_patch` call inside the loop builds and retains a
full computation graph it will never use. On this toy model the cost is
negligible, but the function is exactly the kind of utility a real interp
codebase would reuse for much larger sweeps (many layers × many heads ×
many prompts) — at that scale, an un-guarded loop like this is a well-
known way to quietly blow through memory. `examples/run_attribution.py`
happens to call it from inside an outer `with torch.no_grad():` block, so
the demo doesn't reveal the problem — but the function isn't safe to call
on its own, which is the actual bug: a reusable inference utility
shouldn't depend on every caller remembering to wrap it. Fix: wrap the
loop body in `torch.no_grad()` inside the function itself.
**Severity:** medium — no incorrect output, but a real reliability/scaling
trap once someone reuses this at realistic size.

### 4. Security — `load_cache` uses `torch.load` without `weights_only=True`
**File:** `src/interp_toolkit/cache_io.py`, `load_cache`.
```python
def load_cache(path: str | Path) -> dict[str, torch.Tensor]:
    return torch.load(path)
```
**Why it's a bug:** `torch.load` is pickle-based by default; loading a
file from an untrusted or tampered source can execute arbitrary code
during deserialization — a real, actively-documented vulnerability class
(this is exactly why PyTorch added `weights_only=True`, now default in
recent torch releases, and why loading third-party checkpoints/caches is
treated as a genuine supply-chain risk in ML security discussions). This
is a fourth distinct vector across the practice rounds so far (`eval()`
code exec → `os.system` shell injection → CSV formula injection → now
insecure deserialization) — worth checking whether "don't trust
deserialization of data you don't control" is recognized as the general
principle rather than memorized per-function-name. Activation-cache
sharing (exactly what `save_cache`/`load_cache` are for) is a completely
normal real workflow in interp research, which is what makes this
realistic rather than contrived. `tests/test_cache_io.py` only round-trips
a file this process itself wrote, so it never exercises loading anything
untrusted. Fix: `torch.load(path, weights_only=True)` (and note this
restricts what object types can be unpickled, which is the point).
**Severity:** high — arbitrary code execution on load.

### 5. Typing / shape contract — `layer_ablation_sweep` returns `[n_layers, batch]`, not `[n_layers]`
**File:** `src/interp_toolkit/attribution.py`, `layer_ablation_sweep`.
```python
results.append(logit_diff(patched_logits, correct_token, incorrect_token))
...
return torch.stack(results)
```
**Why it's a bug:** `logit_diff` always returns a `[batch]`-shaped tensor
(see `metrics.py`), never a scalar. Stacking `n_layers` of those gives
shape `[n_layers, batch]`, not the `[n_layers]` the docstring promises.
This is a classic "worked because I only ever tested with batch_size=1"
bug: at batch=1 the extra dimension has size 1, so `.squeeze()` (used in
`examples/run_attribution.py`'s print statements) or casual indexing
makes it look exactly like a clean 1-D vector of length `n_layers` — the
bug is fully camouflaged until someone passes a batch of more than one
prompt, at which point downstream code expecting a flat `[n_layers]`
vector gets silently wrong values (or a shape-mismatch error somewhere
else entirely, far from this function). `test_layer_ablation_sweep_runs_for_every_layer`
asserts `sweep.numel() == model.cfg.n_layers`, which happens to hold at
batch=1 regardless of the actual shape — it would fail to catch this even
at other batch sizes, since `numel()` doesn't distinguish `[n_layers]`
from `[n_layers, batch]` unless batch is compared against explicitly.
**Severity:** medium-high — silently wrong for any non-trivial batch size,
in a way that's specifically invisible in whatever code the author used
to validate it.

## Categories covered
ML-specific/domain pitfall (#1), correctness (#2), design smell —
interp-specific reliability trap (#3), security — new vector each round so
far (#4), typing/shape-contract, also an authentic ML "only tested at
batch=1" bug (#5).

## Not planted, but fair game if flagged
- `examples/run_attribution.py` uses `batch_size=1` throughout, which
  means `layer_ablation_sweep`'s mean-ablation (`activation.mean(dim=0, ...)`)
  is a mathematical no-op — the mean of one sample is itself — so the
  printed "ablation sweep logit diffs" are identical across all three
  layers and the demo doesn't actually demonstrate anything about
  ablation. This is the same root cause (batch=1 everywhere) as bug #5's
  camouflage, just manifesting as a vacuous demo rather than a hidden
  shape bug — worth flagging either way.
- `save_cache`/`load_cache` don't validate that the loaded object is
  actually a `dict[str, torch.Tensor]` before returning it — reasonable
  defensive-programming note, secondary to the security issue in #4.
- `per_layer_logit_attribution` and `layer_ablation_sweep` both re-derive
  `model.cfg.n_layers` rather than taking it as an explicit parameter or
  inferring it from `cache`/`patches` keys — minor API-design point, not a
  planted bug.
