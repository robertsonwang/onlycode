# Answer key — PR: "Add linear probing"

Branch: `rw/interp_toolkit_pr_2` (cut fresh from `main` — no relation to
either prior interp-toolkit PR; none of their bugs are present here).
Diff adds `src/interp_toolkit/probing.py` (`train_linear_probe`,
`evaluate_probe`, `label_examples_by_keyword_pattern`,
`probe_accuracy_by_layer`), `examples/run_probing.py`,
`tests/test_probing.py`, and touches `README.md` / `__init__.py` for
wiring.

All bug *categories* repeat from the prior two rounds (as requested by the
brief), but every specific mechanism is new — no bug here is a rerun of
anything from `rw/interp_toolkit_1` or `rw/interp_toolkit_2`.

Excluded from chat during review. Grade the reviewer's findings against
this list; any real, defensible finding not listed here should still count
in their favor.

## Planted issues (5)

### 1. ML-specific pitfall — train/eval leakage: the probe is evaluated on its own training data
**File:** `src/interp_toolkit/probing.py`, `probe_accuracy_by_layer`.
```python
activation = cache[f"blocks.{layer}.hook_resid_post"][:, 0, :]
probe = train_linear_probe(activation, labels)
accuracies[layer] = evaluate_probe(probe, activation, labels)
```
**Why it's a bug:** this is *the* canonical ML methodology bug: the same
`(activation, labels)` pair is used to fit the probe and to score it.
There is no held-out split. A linear probe with `d_model + 1` parameters
(here 33) trained for 200 unregularized gradient steps against 8 labeled
points will essentially always memorize them regardless of whether the
underlying representation encodes anything real — which is exactly what
`examples/run_probing.py` demonstrates: a *completely untrained, randomly
initialized* model reports **87.5% probe accuracy, identically, at every
single layer**. Random weights cannot possibly encode a text's keyword
label at 87.5% — the number is measuring "can a probe with this much
capacity fit these 8 points," not "does this layer represent the concept."
This is the whole reason held-out evaluation is treated as non-negotiable
in real probing work (a probe that merely memorizes training examples is
not evidence of anything). `tests/test_probing.py` never checks accuracy
on a distinct held-out set, so this ships looking like a working feature.
Fix: split into train/test (or run k-fold), and only report accuracy on
the held-out portion.
**Severity:** highest of the round — a probing tool whose headline metric
is uninterpretable by construction is actively worse than no tool, since
it produces a confident-looking number people will cite.

### 2. Correctness — probes the wrong sequence position
**File:** `src/interp_toolkit/probing.py`, `probe_accuracy_by_layer`.
```python
activation = cache[f"blocks.{layer}.hook_resid_post"][:, 0, :]
```
**Why it's a bug:** `[:, 0, :]` takes the *first* sequence position, not
the last. For a decoder-only model, the residual stream at any early
position has only seen the tokens up to that point — the concept-bearing
information (whatever the rest of the sentence conveys) mostly hasn't
arrived yet. Probing position 0 largely measures what's encoded about the
first token in isolation (frequently positional/BOS-adjacent information),
not "what does this layer know about the whole example" — which is
presumably the actual question `probe_accuracy_by_layer` is meant to
answer, matching the pattern used everywhere else in this codebase
(`per_layer_logit_attribution`, `logit_diff`, etc. all read the *final*
position). Should be `[:, -1, :]`.
**Severity:** high — silently probes a materially different, less
meaningful representation than intended, and combines with #1 to make the
reported numbers doubly uninformative.

### 3. Design smell / PyTorch hygiene — gradients accumulate across training steps
**File:** `src/interp_toolkit/probing.py`, `train_linear_probe`.
```python
for _ in range(n_steps):
    logits = probe(activations).squeeze(-1)
    loss = F.binary_cross_entropy_with_logits(logits, labels.float())
    loss.backward()
    with torch.no_grad():
        for param in probe.parameters():
            param -= lr * param.grad
```
**Why it's a bug:** `.backward()` in PyTorch *accumulates* into
`.grad` by default rather than overwriting it — this loop never resets
`param.grad` between iterations (no `probe.zero_grad()` / no
`param.grad = None`), so by step `k` each parameter's update uses the
*sum* of every gradient computed so far, not just the current step's
gradient. The effective step size silently grows every iteration instead
of staying at `lr`. This happens to still produce a number that "looks
like a working training loop" on this toy problem (loss generally still
trends down enough for a good training accuracy, partly *because* of bug
#1 making the target trivial to hit), which is exactly why it's easy to
ship without noticing — the failure mode is "converges to something
technically fittable, following the wrong dynamics," not a crash.
Fix: `probe.zero_grad()` (or set each `param.grad = None`) at the top of
every iteration, before `.backward()`.
**Severity:** medium — doesn't crash, but the actual optimization being
run is not gradient descent with a fixed learning rate, which is a real
correctness problem for anyone using this to compare probe quality across
layers (an artifact of the *training dynamics*, not the representation,
is contaminating the comparison).

### 4. Security — unbounded regex against untrusted input (ReDoS)
**File:** `src/interp_toolkit/probing.py`, `label_examples_by_keyword_pattern`.
```python
def label_examples_by_keyword_pattern(texts: list[str], pattern: str) -> torch.Tensor:
    return torch.tensor([1 if re.search(pattern, text) else 0 for text in texts])
```
**Why it's a bug:** `pattern` is run directly through Python's `re`
module with no complexity bound, no timeout, and no validation. Python's
backtracking regex engine is vulnerable to catastrophic backtracking on
certain pattern shapes (e.g. nested quantifiers like `(a+)+$`) matched
against a crafted input — a classic ReDoS (CWE-1333). If `pattern` ever
comes from outside the immediate caller (a config file, a labeling rule
generated by another part of a pipeline, a user-facing "define your own
labeling heuristic" feature — all very plausible uses for exactly this
function), a malicious or even just poorly-written pattern can hang the
process indefinitely on ordinary-looking input, with no exception to
catch. This is a different vector from every earlier round's security bug
(`eval()` code exec, `os.system` shell injection, CSV formula injection,
`torch.load` pickle deserialization, path traversal) — worth checking
whether "don't run unbounded computation over untrusted input" registers
as its own category, distinct from the injection/deserialization bugs
seen so far. `tests/test_probing.py` only exercises a simple, safe
pattern, so this ships invisibly.
**Severity:** high — denial of service, and notably `pattern` here isn't
even a file path or a network payload, which makes it an easy thing to
overlook as a "trust boundary" at all.

### 5. Typing — `evaluate_probe` declared `-> torch.Tensor`, returns a `float`
**File:** `src/interp_toolkit/probing.py`, `evaluate_probe`.
```python
def evaluate_probe(probe: nn.Linear, activations: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    ...
    accuracy = (predictions == labels).float().mean()
    return accuracy.item()
```
**Why it's a bug:** `.item()` extracts a plain Python scalar from a
0-dimensional tensor — the function returns a `float`, not a
`torch.Tensor`, contradicting its own signature. It happens to cause zero
runtime problems here because the only caller,
`probe_accuracy_by_layer`, stores the result directly into a
`dict[int, float]` — exactly the type `.item()` produces — so nothing
downstream ever notices. That's precisely what makes this bug class worth
scanning for on its own: a type checker (mypy/pyright) would flag it
immediately, but nothing about running the code ever will, because the
one place it's used happens to want a `float` anyway. A different
manifestation of the same underlying lesson as the missing-`return`
bugs from the last two rounds (declared type vs. actual type silently
diverging) — this time via an unnoticed `.item()` rather than a forgotten
`return`.
**Severity:** low-medium — no functional impact today, but breaks
silently for the next caller who takes the type hint at face value and
tries to do further tensor ops (e.g. `torch.stack`-ing several probes'
accuracies) on the result.

## Categories covered
ML-specific pitfall — train/eval leakage, the most canonical ML bug, used
for the first time in this series (#1); correctness — wrong sequence
position, a third distinct flavor of correctness bug across the
interp-toolkit rounds (#2); design smell — PyTorch gradient-hygiene, a
brand-new mechanism (#3); security — ReDoS, a sixth distinct vector across
all rounds so far (#4); typing — `.item()` truncation, a new manifestation
of the declared-vs-actual-type theme (#5).

## Not planted, but fair game if flagged
- `train_linear_probe` has no convergence check / early stopping and
  always runs the full `n_steps` — minor inefficiency, not a planted bug.
- `probe_accuracy_by_layer` re-derives the activation per layer from a
  single shared `cache` (fine) but re-trains a probe from scratch for
  every layer with no shared initialization/seeding control — reasonable
  if the reviewer wants layer-to-layer comparisons to be on equal footing
  training-wise, but not something this PR claims to guarantee.
- `label_examples_by_keyword_pattern` doesn't handle `re.error` for an
  invalid pattern (crashes with a raw exception rather than a clear
  message) — a legitimate, secondary point to the ReDoS issue in #4.
