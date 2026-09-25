# Answer key — PR: "Add sparse autoencoder"

Branch: `rw/interp_toolkit_3` (cut fresh from `main` — no relation to any
prior interp-toolkit PR; none of their bugs are present here). Diff adds
`src/interp_toolkit/sae.py` (`SparseAutoencoder`, `sae_loss`,
`top_activating_examples`, `export_feature_dashboard_html`),
`examples/run_sae.py`, `tests/test_sae.py`, and touches `README.md` /
`__init__.py` for wiring.

Excluded from chat during review. Grade the reviewer's findings against
this list; any real, defensible finding not listed here should still count
in their favor.

## Planted issues (5)

### 1. ML-specific pitfall (SAE-specific) — encoder doesn't subtract the decoder bias
**File:** `src/interp_toolkit/sae.py`, `SparseAutoencoder.encode`.
```python
def encode(self, x: torch.Tensor) -> torch.Tensor:
    return F.relu(x @ self.W_enc + self.b_enc)
```
**Why it's a bug:** the standard SAE architecture (as used throughout
Anthropic's SAE work and virtually every open-source SAE implementation,
e.g. SAELens) encodes as `ReLU(W_enc @ (x - b_dec) + b_enc)`, subtracting
the *decoder's* bias before the encoder projection — this is often called
the "pre-encoder bias" and it matters because `b_dec` is trained to sit at
the centroid of the activation distribution, so subtracting it centers
the input before the encoder has to do anything with it; skipping this
step is a specific, documented mistake that shows up in early/naive SAE
implementations. This SAE has an unused `b_dec` parameter sitting right
there (used in `decode`, never referenced in `encode`) — that alone is a
signal worth following. Doesn't crash, doesn't even prevent the SAE from
training (loss still goes down in `examples/run_sae.py`) — it just means
the SAE isn't the architecture it's claimed to be, and won't match the
behavior of any reference SAE implementation trained the standard way.
**Severity:** high — an architectural correctness bug specific to SAEs
that's invisible unless you know the expected formula, which is exactly
why it's worth having memorized before a from-scratch SAE review.

### 2. Correctness — sparsity loss reduces over the wrong dimensions
**File:** `src/interp_toolkit/sae.py`, `sae_loss`.
```python
sparsity_loss = features.mean()
```
**Why it's a bug:** `features` has shape `[batch, d_hidden]` (here
`[batch, 64]`). `.mean()` with no `dim` argument averages over *every*
element — both the batch dimension and the 64 feature dimensions at once.
The L1 sparsity penalty is supposed to penalize each example's total
activation mass (summed across features), then average that per-example
penalty over the batch — i.e. `features.sum(dim=-1).mean()`. Averaging
over the feature dimension too means the effective sparsity penalty is
silently `d_hidden` times weaker than the `l1_coefficient` the caller
picked (64x weaker here) — the SAE trains, converges, and looks
successful (`examples/run_sae.py`'s loss drops smoothly to ~0.0007), but
the actual sparsity pressure applied is a small fraction of what was
intended, which is exactly why `dead feature count: 23 / 64` is lower than
you'd expect for a properly-sparsified 4x-overcomplete dictionary trained
this long. This specific sum-vs-mean confusion over the feature axis is
one of the most common real SAE training bugs in practice.
**Severity:** high — doesn't crash, silently defeats the entire point of
the L1 term, and produces a dictionary that looks "trained" while being
meaningfully less sparse/interpretable than intended.

### 3. Design smell — dead-feature tracking is a shared class attribute
**File:** `src/interp_toolkit/sae.py`, `SparseAutoencoder`.
```python
class SparseAutoencoder(nn.Module):
    dead_feature_counts: dict[int, int] = {}   # <- class attribute, not set in __init__
```
**Why it's a bug:** same root cause as the `CachingModel._cache` bug a
few rounds back in eval-harness (and the mutable-default-argument bug in
the very first redteam PR) — a mutable object created once, at class-body
execution time, shared by every instance that doesn't explicitly
reassign it in `__init__`. `track_dead_features` mutates it via
`self.dead_feature_counts[i] = ...` (never `self.dead_feature_counts =
{...}`), so no instance attribute ever gets created and every
`SparseAutoencoder` in the process shares one dict. Training two SAEs
side by side to compare hyperparameters (dictionary size, L1 coefficient
— an extremely standard SAE research workflow) would silently blend their
dead-feature statistics together. Third exposure to this exact pattern
across the whole series (missed both previous times) — first time
specifically within interp-toolkit. Fix: `self.dead_feature_counts = {}`
in `__init__`.
**Severity:** medium — doesn't corrupt the SAE's actual weights or
training, only a diagnostic/analysis-side statistic, but a genuinely
common real workflow (comparing multiple SAEs) is exactly what triggers
it.

### 4. Security — unescaped model/user text embedded directly into HTML (XSS)
**File:** `src/interp_toolkit/sae.py`, `export_feature_dashboard_html`.
```python
rows = "\n".join(
    f"<tr><td>{text}</td><td>{activation:.3f}</td></tr>" for text, activation in top_examples
)
```
**Why it's a bug:** `text` — an arbitrary example string, potentially
model-generated or drawn from an adversarial/red-team dataset (exactly
the kind of data SAE feature dashboards are built to inspect, since a
major use case is finding out what triggers a feature on unusual or
adversarial inputs) — is interpolated directly into HTML with no
escaping. Any example containing `<script>...</script>`, an `onerror=`
attribute, or similar gets written verbatim into the output file. SAE
feature dashboards are essentially always *opened in a browser* by a
human afterward (this is the standard workflow — e.g. Neuronpedia-style
static feature pages) — so this is a real stored-XSS vector (CWE-79): a
crafted example in the input data results in arbitrary JavaScript
executing in whoever's browser opens the report. A seventh distinct
security vector across the whole series so far (`eval()`, `os.system`,
CSV formula injection, `torch.load` pickle deserialization, path
traversal, ReDoS, now HTML injection) — worth checking whether "escape
untrusted text before embedding it in a format with its own execution
semantics" is recognized as the general principle, independent of which
specific format (shell, SQL, HTML, regex...) is involved.
`tests/test_sae.py` only exercises plain, punctuation-free example text,
so this ships invisibly. Fix: use `html.escape(text)` (stdlib) before
interpolating, or a templating engine with autoescaping.
**Severity:** high — arbitrary script execution in a viewer's browser.

### 5. Typing — `sae_loss` declared to return a 3-tuple, returns 2
**File:** `src/interp_toolkit/sae.py`, `sae_loss`.
```python
def sae_loss(...) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total_loss, reconstruction_loss, sparsity_loss)."""
    ...
    return total_loss, recon_loss
```
**Why it's a bug:** the signature and docstring both promise three
values; the function returns two, silently dropping `sparsity_loss`.
Every current call site (`examples/run_sae.py`,
`tests/test_sae.py::test_sae_loss_runs_and_backpropagates`) only
destructures `total_loss, recon_loss = sae_loss(...)`, matching the
*actual* 2-tuple, so nothing breaks today — but any new caller who reads
the type hint or docstring and writes
`total, recon, sparsity = sae_loss(...)` (entirely reasonable, since
that's what's documented) gets an immediate
`ValueError: not enough values to unpack`. A third distinct manifestation
of "declared return contract vs. actual return value" across the
interp-toolkit rounds — this time an arity mismatch rather than a missing
`return` or a wrong scalar type.
**Severity:** low-medium — no current caller is affected, but it's a
landmine for the next one, and worth noticing purely from reading the
signature against the body, without needing to run anything.

## Categories covered
ML-specific/SAE-domain pitfall (#1), correctness — wrong reduction
dimension (#2), design smell — shared mutable class state, third
exposure to this pattern across the series (#3), security — new vector,
HTML/XSS injection (#4), typing — return-arity mismatch, a new
manifestation of the declared-vs-actual-contract theme (#5).

## Not planted, but fair game if flagged
- `top_activating_examples` calls `sae.encode(activations)` from scratch
  on every call; building dashboards for many features in a loop would
  redundantly recompute the same encoding repeatedly — the same
  "redundant recomputation" design smell used in an earlier round, not
  re-planted here but still genuinely present if the reviewer notices it.
- `SparseAutoencoder.__init__` initializes `W_enc`/`W_dec` with
  `torch.randn(...) * 0.1` and never normalizes `W_dec`'s columns to unit
  norm — a well-known SAE stabilization technique (without it, the L1
  penalty can be gamed by shrinking feature activations while inflating
  decoder weights, since the reconstruction is invariant to that
  rescaling). Related to, but distinct from, bug #2 — a fair, sophisticated
  catch if raised, not required.
- `track_dead_features` calls `.item()` inside a Python-level loop over
  `d_hidden` elements (64 individual GPU-CPU syncs in the naive case) —
  a performance nitpick, not a planted bug.
