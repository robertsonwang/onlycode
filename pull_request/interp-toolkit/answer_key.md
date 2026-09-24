# Answer key — PR: "Add contrastive activation steering"

Branch: `rw/interp_toolkit_2` (cut fresh from `main`, no relation to the
earlier `rw/interp_toolkit_1` attribution/ablation/cache-io PR — that PR's
bugs are not present here). Diff adds `src/interp_toolkit/steering.py`
(`compute_steering_vector`, `generate_with_steering`, `steering_sweep`,
`save_steering_vector`, `load_steering_vector`), `examples/run_steering.py`,
`tests/test_steering.py`, and touches `README.md` / `__init__.py` /
`.gitignore` for wiring.

Excluded from chat during review. Grade the reviewer's findings against
this list; any real, defensible finding not listed here should still count
in their favor.

## Planted issues (5)

### 1. ML-specific pitfall (domain-specific) — pooling over the whole sequence contaminates the direction
**File:** `src/interp_toolkit/steering.py`, `compute_steering_vector`.
```python
positive_mean = positive_cache[hook_name].mean(dim=(0, 1))
negative_mean = negative_cache[hook_name].mean(dim=(0, 1))
```
**Why it's a bug:** `dim=(0, 1)` averages over *both* the example
dimension (0) and the sequence-position dimension (1) at once. In a
contrastive-pair setup, the whole point is that positive and negative
prompts share the same boilerplate/context tokens and differ only at the
final (or some specific) position — that's what isolates the "concept."
Averaging in every shared context-token position along with the one
position that actually differs washes most of the signal out with
identical, non-discriminative activations from both sets. The standard
approach (used throughout CAA / RepE-style steering work) is to first
select the relevant position(s) — typically the final token —
*then* average only over the example dimension. `examples/run_steering.py`
deliberately constructs prompts that are identical except for the last
token specifically to make this obvious if you inspect the resulting
direction's usefulness, but no test compares steering effectiveness with
vs. without proper position selection, so this ships looking like
reasonable code. Fix: `positive_cache[hook_name][:, -1, :].mean(dim=0)` (or
whichever position(s) are actually meant to carry the concept).
**Severity:** high — this is a methodological bug that produces a "direction"
which still runs, still has the right shape, and still does *something*
when added to activations, making it especially easy to ship without
noticing the extraction itself was compromised.

### 2. Correctness — sign flip: the direction points backwards
**File:** `src/interp_toolkit/steering.py`, `compute_steering_vector`.
```python
return negative_mean - positive_mean
```
**Why it's a bug:** the docstring states the direction "points from
'negative' towards 'positive': adding it (scaled) ... should push the
model's behavior towards whatever the positive examples have in common."
The implementation computes the opposite subtraction — `negative - positive`
instead of `positive - negative` — so increasing the steering coefficient
pushes *away* from the positive behavior. This is directly visible if you
run `examples/run_steering.py`: `correct_token=10` is the token every
"positive" example ends with, yet the printed logit diff between
`correct_token` and `incorrect_token` *decreases* as the coefficient
increases (0.89 → 0.50 → 0.10) — exactly backwards from what a working
steering vector should do. No test checks the direction of the effect
(only shapes/keys), so this ships silently. This is the single most
dangerous bug in this PR: a steering vector that reliably does the
opposite of what's documented is worse than one that does nothing, in any
context where steering is used for safety work (e.g. suppressing an
unwanted behavior) — it would reliably amplify exactly what it claims to
suppress.
**Severity:** high, arguably the highest of the round given how visibly
and silently it inverts the tool's entire stated purpose.

### 3. Design smell — redundant recomputation in the coefficient sweep
**File:** `src/interp_toolkit/steering.py`, `steering_sweep`.
```python
return {
    c: generate_with_steering(model, tokens, hook_name, steering_vector, c) for c in coefficients
}
```
**Why it's a bug:** `generate_with_steering` calls `run_with_cache(model, tokens)`
internally to get the base activation to add the steering vector to.
`steering_sweep` calls `generate_with_steering` once per coefficient, so
for a sweep of N coefficients over the *same* `model`/`tokens`, the
identical forward pass to populate the cache gets rerun N times — pure
duplicated work that only depends on `model`/`tokens`, not on
`coefficient`. Negligible on this toy model, but a real cost multiplier on
any real-sized model, and exactly the kind of thing that turns "sweep 20
coefficients across 10 layers" into 200 redundant forward passes instead
of 10. Fix: compute the base cache/activation once, outside the loop, and
reuse it for every coefficient.
**Severity:** medium — a scaling/performance trap, not incorrect output.

### 4. Security — path traversal via unsanitized `label`
**File:** `src/interp_toolkit/steering.py`, `save_steering_vector` and
`load_steering_vector`.
```python
path = Path(f"{cache_dir}/{label}.pt")
```
**Why it's a bug:** `label` is concatenated directly into a file path with
no validation. A `label` containing `../` sequences (e.g.
`"../../../../tmp/evil"`) or an absolute path escapes `cache_dir` entirely
— `save_steering_vector` would write, and `load_steering_vector` would
read, anywhere on the filesystem the process has permissions for (CWE-22).
"Label a concept by name and save/load it" is exactly the kind of
user-facing, string-typed identifier that tends to end up passed through
from a config file, CLI arg, or even a model's own output in an automated
pipeline — this is a different vector again from every prior round's
security bug (`eval()` code exec, `os.system` shell injection, CSV
formula injection, `torch.load` pickle deserialization) — the point is
checking whether "sanitize/validate any user-supplied string used to
build a filesystem path" registers as its own category, separate from the
deserialization-specific lesson from the very last interp-toolkit round.
`tests/test_steering.py` only exercises a well-formed label
(`"my-concept"`), so this ships invisibly. Fix: validate `label` (e.g.
reject anything containing `/`, `\`, or `..`, or use
`Path(label).name` and compare against the original to detect traversal)
before building the path.
**Severity:** high — arbitrary file read/write.

### 5. Typing — `save_steering_vector` declared `-> Path`, has no `return`
**File:** `src/interp_toolkit/steering.py`, `save_steering_vector`.
```python
def save_steering_vector(vector, label, cache_dir="steering_cache") -> Path:
    path = Path(f"{cache_dir}/{label}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vector, path)
    # no return -- implicitly returns None
```
**Why it's a bug:** same shape as a bug from two rounds ago in
eval-harness (`export_results_to_csv`) — a function typed to hand back the
resolved path (so a caller can log/confirm/chain on it) simply never
returns anything. `examples/run_steering.py` doesn't use the return value,
so it ships invisibly. Planted again, in a different repo and function, to
check whether "does every declared return type have a matching `return`"
has become a reflexive check yet, independent of the specific earlier
example.
**Severity:** low-medium — doesn't crash anything by itself, silently
breaks any caller relying on the documented return type.

## Categories covered
ML-specific/domain pitfall (#1), correctness — sign flip (#2), design
smell — redundant recomputation (#3), security — new vector, path
traversal (#4), typing — same pattern class as a prior round, different
repo/function (#5).

## Not planted, but fair game if flagged
- `steering.py` writes its own `torch.save`/`torch.load` pair instead of
  reusing `cache_io.py`'s `save_cache`/`load_cache` (from the other
  interp-toolkit branch, not present on this one) — moot here since that
  PR isn't merged into this branch, but worth a mention if the reviewer
  is thinking about eventual convergence between the two.
- `load_steering_vector` has the exact same insecure-deserialization
  property as `cache_io.py`'s `load_cache` in the sibling PR
  (`torch.load` without `weights_only=True`) — not counted as a *separate*
  planted bug from #4, but worth acknowledging if raised, since it's a
  real, independent issue layered on top of the path-traversal one: even
  with a safe `label`, loading a tampered `.pt` file is still unsafe.
- `generate_with_steering` and `steering_sweep` don't validate that
  `steering_vector.shape == (model.cfg.d_model,)` before broadcasting it
  against the cached activation — a malformed vector would fail with a
  broadcast error somewhere inside `run_with_patch` rather than a clear
  message at the call site. Reasonable defensive-programming note.
