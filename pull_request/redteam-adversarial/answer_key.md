# Answer key — PR: "Add batch attack runner"

Branch: `rw/redteam-adversarial-test`. Diff adds `src/redteam_adversarial/batch.py`,
`examples/run_batch.py`, `data/prompts.txt`, `data/synonym_overrides.txt`,
`tests/test_batch.py`, and touches `README.md` / `__init__.py` for wiring.

This file is intentionally excluded from what gets shown in chat during the
review. Grade the reviewer's findings against this list; anything not listed
here that's still a real, defensible finding should count in their favor.

## Planted issues (5)

### 1. Security — arbitrary code execution via `eval()`
**File:** `src/redteam_adversarial/batch.py`, `load_synonym_overrides`, the
`parsed = eval(line)` call.
**Why it's a bug:** `eval()` executes arbitrary Python from the overrides
file. Any line in `data/synonym_overrides.txt` — or any file path a caller
points this at — runs as code with the full privileges of the process, not
just dict-literal parsing. This is a classic OWASP-style injection risk
(equivalent to `pickle.loads` on untrusted input). Should be
`json.loads(line)` with the file using JSON syntax, or `ast.literal_eval`
if Python-literal syntax is truly wanted (still restricted to literals, no
code execution).
**Severity:** high (real RCE risk on attacker-controlled input, and this is
a red-teaming/safety tool — ships a code-exec hole under the banner of
testing a safety classifier).

### 2. Silent failure — bare `except Exception: continue`
**File:** `src/redteam_adversarial/batch.py`, `load_synonym_overrides`.
**Why it's a bug:** Any malformed, malicious, or merely differently-formatted
line is silently dropped with no logging, no count, no exception surfaced.
A user could have most of their overrides silently fail to load and never
know — masks both bugs and the exploit surface in #1 (a failed exploit
attempt looks identical to a typo). Should at minimum log the skipped line
and reason, or fail loudly.
**Severity:** medium. Related to #1 — compounds it by hiding evidence.

### 3. Correctness — mutable default argument
**File:** `src/redteam_adversarial/batch.py`,
`load_synonym_overrides(path, base_overrides: dict[str, list[str]] = {})`.
**Why it's a bug:** The classic Python gotcha — the `{}` default is created
once at function-definition time and reused across every call that doesn't
pass `base_overrides` explicitly. Calling `load_synonym_overrides` twice in
the same process (e.g. once per file, or in a test suite) silently
accumulates entries from the first call into the second's "fresh" dict.
Doesn't show up in a single call (which is why the shipped tests don't
catch it) — only on repeated/sequential use. Fix: default to `None`, then
`if base_overrides is None: base_overrides = {}` inside the function.
**Severity:** medium — a real, well-known bug class; low blast radius here
but a canonical review catch.

### 4. Typing/logic mismatch — `max_substitutions=None` crashes
**Files:**
`src/redteam_adversarial/batch.py`, `run_batch_attack` signature and
docstring ("`max_substitutions=None` means 'no cap'"); also claimed in
`README.md`'s new "Batch attacks" section.
`src/redteam_adversarial/search.py`, `greedy_word_substitution_attack`,
`max_substitutions: int = 5` and `for _ in range(max_substitutions):`.
**Why it's a bug:** `run_batch_attack` types `max_substitutions` as
`int | None` and documents `None` as "unlimited," then passes it straight
through to `greedy_word_substitution_attack`, whose parameter is typed
plain `int` and used directly in `range(max_substitutions)`. Calling
`range(None)` raises `TypeError: 'NoneType' object cannot be interpreted
as an integer`. This is both a runtime crash on the documented "no cap"
usage and a real static-typing error (mypy would flag passing
`Optional[int]` into a plain-`int` parameter). Not exercised by
`test_run_batch_attack_returns_one_result_per_prompt` or
`examples/run_batch.py`, both of which pass an explicit `max_substitutions=5`.
Fix: either don't advertise `None` as valid (`int` only, no default cap
change), or handle `None` explicitly (e.g. translate to a large int, or
loop with a `while True` + break condition instead of `range`).
**Severity:** high — crashes on documented, seemingly-reasonable usage.

### 5. Correctness — `rank_results` doesn't rank by what it claims
**File:** `src/redteam_adversarial/batch.py`, `rank_results`.
**Docstring:** "Return the `top_k` attacks with the largest score drop, most
effective first."
**Implementation:** `sorted(results, key=lambda r: r.final_score)[:top_k]`
— sorts by `final_score` ascending only, ignoring `original_score`
entirely. Score *drop* is `original_score - final_score`; a prompt that
started at 0.99 and dropped to 0.3 (drop = 0.69, very effective) can rank
below a prompt that started at 0.4 and dropped to 0.35 (drop = 0.05,
barely effective) purely because 0.35 > 0.3... — actually more precisely,
low-`final_score` prompts are favored regardless of how much they moved,
so two prompts with identical `final_score` but wildly different
`original_score` (and thus wildly different actual attack effectiveness)
rank identically, and a prompt with a low starting score that moved barely
at all can outrank a prompt that dropped enormously. `test_rank_results_returns_top_k`
only checks the returned length, not the ordering, so it doesn't catch
this. Fix: sort by `r.original_score - r.final_score`, descending.
**Severity:** medium — silently wrong output, exactly the kind of bug that
survives because nobody diffed the docstring against the one-line
implementation.

## Categories covered
security/safety (#1), code quality / silent-failure design smell (#2),
classic correctness gotcha (#3), typing (#4), logic/doc-mismatch
correctness (#5).

## Not planted, but fair game if flagged
- `load_prompts` has no encoding/error handling for missing files (raises a
  raw `FileNotFoundError` with no context) — reasonable nitpick, not counted
  as a "miss" either way.
- No test exercises `load_synonym_overrides` called twice in one process
  (would have caught #3) — worth noting as a test-coverage gap if the
  reviewer brings it up.
- `run_batch_attack`'s list comprehension re-tokenizes/re-scores from
  scratch per prompt with no shared caching — a legitimate efficiency
  observation, not a planted bug.
