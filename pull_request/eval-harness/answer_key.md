# Answer key — PR: "Add caching, sampling, and CSV export"

Branch: `rw/eval_harness_pr_2` (cut from `main`, no relation to the earlier
`rw/eval_harness_pr` suite-runner PR). Diff adds `src/eval_harness/caching.py`,
`src/eval_harness/sampling.py`, `src/eval_harness/export.py`,
`examples/run_export_demo.py`, `tests/test_caching.py`,
`tests/test_sampling.py`, `tests/test_export.py`, and touches
`README.md` / `__init__.py` / `.gitignore` for wiring.

Excluded from chat during review. Grade the reviewer's findings against
this list; any real, defensible finding not listed here should still count
in their favor.

## Planted issues (5)

### 1. Correctness — class-level cache is shared across every `CachingModel` instance
**File:** `src/eval_harness/caching.py`.
```python
class CachingModel:
    _cache: dict[str, str] = {}   # <- class attribute, not set in __init__

    def __init__(self, model: ModelClient):
        self._model = model
        ...
```
**Why it's a bug:** `_cache` is a class attribute, created once when the
class body executes, not per-instance. Every `CachingModel` wrapping every
different underlying model in the same process shares one dict. Build two
`CachingModel`s around two different models (e.g. two different prompt/
response fixtures, or a real API-backed model swapped in for a dev/staging
comparison) and the second instance can silently return the *first*
instance's cached response for the same prompt text — cross-contamination
between what should be independent model runs. `tests/test_caching.py`
only ever constructs one `CachingModel` per test (with unique prompt keys
across tests specifically to avoid tripping this), so it never exercises
two live instances at once. Fix: initialize `self._cache = {}` in
`__init__`.
**Severity:** high, and the same *pattern* (shared mutable state fixed at
definition time rather than per-instance/per-call) as the mutable-default-
argument bug from the very first PR — this is that bug's sibling shape
(class attribute instead of default argument value), planted to see
whether the pattern generalizes rather than being remembered as "one
specific line to check."

### 2. ML-specific pitfall — `seed` is accepted but never used
**File:** `src/eval_harness/sampling.py`, `sample_examples`.
```python
def sample_examples(task: Task, n: int, seed: int | None = None) -> Task:
    """... Pass `seed` to make the sample reproducible across runs."""
    sampled = random.sample(task.examples, n)
    ...
```
**Why it's a bug:** the docstring explicitly promises reproducibility via
`seed`, and the signature accepts one, but the implementation calls the
module-level `random.sample`, which draws from the global RNG state and
never references `seed` at all. Two calls with the same `seed` will (in
general) return different samples. This is the single most on-theme bug
for an eval harness: "reproducible smoke test" is exactly the kind of
claim a research team would rely on without re-checking, and a silently-
ignored seed means two people (or two runs) debugging "the same" eval
subset are actually looking at different examples. None of
`tests/test_sampling.py` calls `sample_examples` twice with the same seed
and compares results, so this ships invisibly. Fix: use
`random.Random(seed).sample(...)`, or call `random.seed(seed)` before
sampling (less good — mutates global state) — either way, `seed` needs to
actually reach the RNG.
**Severity:** high for a research-tooling context specifically — silent
non-reproducibility is exactly the failure mode that erodes trust in eval
results without ever throwing an error.

### 3. Security — CSV built by string concatenation → formula/CSV injection
**File:** `src/eval_harness/export.py`, `export_results_to_csv`.
```python
row = (
    f"{r.example.example_id},{r.example.prompt},{r.response},"
    f"{r.example.reference},{r.score}\n"
)
```
**Why it's a bug:** two distinct problems from the same root cause (hand-
rolled CSV instead of the `csv` module):
(a) **Correctness** — any prompt, response, or reference containing a
comma or a newline (extremely likely for real model output) silently
shifts or splits columns, corrupting the file with no error.
(b) **Security (CWE-1236, CSV/formula injection)** — if a model response
happens to start with `=`, `+`, `-`, or `@` (e.g. a model asked to
role-play a spreadsheet formula, or an adversarial/red-team prompt
designed to produce exactly this), the cell is interpreted as a formula
by Excel/Google Sheets/LibreOffice when a human opens the exported file —
a known vector for both data exfiltration (`=WEBSERVICE(...)`-style
formulas) and arbitrary command execution in some Excel configurations.
This is a different *vector* from both prior rounds' `eval()`/`os.system`
bugs (code execution in-process vs. shell command injection vs. this:
injection into a *downstream* application the harness doesn't control) —
worth checking whether security review generalizes to "untrusted text
flowing into any interpreter, including ones outside this process."
Neither `tests/test_export.py` nor `examples/run_export_demo.py` uses
responses containing commas or a leading `=`/`+`/`-`/`@`, so this ships
invisibly. Fix: use `csv.writer`, which handles quoting/escaping
correctly; for the formula-injection half specifically, also worth
prefixing any cell starting with `=+-@` with a `'` or space before writing
(a standard mitigation), since `csv.writer` alone doesn't defend against
that on its own.
**Severity:** high — this is model-generated/attacker-influenced text
being exported for a human to open in a program the harness doesn't
control.

### 4. Design smell — file handle opened without a context manager, never closed
**File:** `src/eval_harness/export.py`, `export_results_to_csv`.
```python
f = open(resolved, "w", encoding="utf-8")
f.write(...)
...
# no f.close(), no `with`
```
**Why it's a bug:** the file is never explicitly closed or flushed. In
CPython, relying on GC/refcounting to eventually close it works most of
the time, but it's not guaranteed (other interpreters, early process exit,
an exception raised mid-loop that skips the rest of the writes) — and even
setting reliability aside, it's a real resource leak if this function is
called in a loop (e.g. exporting many tasks), since each open handle isn't
released until GC gets to it. The idiomatic, deterministic fix is a `with
open(...) as f:` block.
**Severity:** medium — not usually catastrophic in a short-lived script,
but a real reliability smell, and exactly the kind of thing that turns
into a "why did the last few rows go missing" ticket in a longer-running
process.

### 5. Typing — declared to return `Path`, has no `return` statement
**File:** `src/eval_harness/export.py`, `export_results_to_csv`.
```python
def export_results_to_csv(result: EvalResult, path: str | Path) -> Path:
    ...
    # falls off the end of the function; implicitly returns None
```
**Why it's a bug:** the signature promises a `Path` (presumably so a
caller can chain `export_results_to_csv(...).read_text()` or log the
resolved path), but the function body never returns anything, so it always
returns `None` — a straightforward violation of its own declared contract
that a type checker (mypy/pyright) would flag immediately. Neither the
tests nor `examples/run_export_demo.py` use the return value (the demo
only reads `OUT_PATH`, which it already had), so this ships invisibly. A
different shape of typing bug than the earlier rounds' "returns `None` on
an error branch when typed non-Optional" — this one is just a forgotten
`return`, the more mundane and probably more common real-world version of
"the type hint and the implementation disagree."
**Severity:** low-medium — doesn't crash anything by itself, but silently
breaks any caller who relies on the documented return type.

## Categories covered
correctness / shared-state generalization test (#1), ML-specific pitfall —
reproducibility (#2), security — new injection vector (#3), design smell —
resource handling (#4), typing — missing return (#5).

## Not planted, but fair game if flagged
- `sample_examples(task, n, ...)` doesn't validate `n <= len(task.examples)`;
  `random.sample` raises `ValueError: Sample larger than population or is
  negative` if `n` exceeds the task size. Reasonable to flag as missing
  input validation, though it's arguably acceptable to let the stdlib's
  own error propagate.
- `CachingModel` doesn't expose a way to clear/bound the cache (unbounded
  growth for a long-running process) — legitimate scalability observation,
  not a planted bug.
- `export_results_to_csv` takes a single `EvalResult`, not the multi-task
  `SuiteResult` from the (separate, not-yet-merged) suite-runner PR — a
  reasonable API-consistency question if the reviewer is tracking both
  branches, but out of scope for this PR alone.
