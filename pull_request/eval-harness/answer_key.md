# Answer key — PR: "Add multi-task suite runner with retries"

Branch: `rw/eval_harness_pr`. Diff adds `src/eval_harness/suite.py`, a new
`numeric_tolerance` scorer in `scoring.py`, `examples/run_suite.py`,
`data/task_small.jsonl` / `data/task_large.jsonl`, `tests/test_suite.py`,
and touches `README.md` / `__init__.py` / `tests/test_scoring.py` for wiring.

Excluded from chat during review. Grade the reviewer's findings against
this list; any real, defensible finding not listed here should still count
in their favor.

## Planted issues (5)

### 1. Correctness (statistics) — `overall_mean_score` is an unweighted mean of per-task means
**File:** `src/eval_harness/suite.py`, `SuiteResult.overall_mean_score`.
```python
return sum(tr.result.mean_score for tr in self.task_results) / len(self.task_results)
```
**Why it's a bug:** this is a mean of means, not a mean over all examples.
Whenever tasks have different numbers of examples it's silently wrong — see
`examples/run_suite.py`: a 2-example task scored 1.0 and an 8-example task
scored 0.0 produce `overall_mean_score = 0.5`, but the true fraction correct
across all 10 examples is `2/10 = 0.2`. The bug is invisible in
`test_run_suite_returns_one_task_result_per_task` and
`test_run_suite_retries_on_transient_failure` because both use single-
example tasks, where mean-of-means and the true weighted mean coincide.
Fix: `sum(r.score for tr in task_results for r in tr.result.example_results) / total_n_examples`,
or weight each task's mean by `tr.result.n_examples`.
**Severity:** high — this is the number a suite's headline "how good is
the model" report would show; a large task dragging the true average down
(or up) gets diluted to equal footing with a tiny one.

### 2. Correctness / silent data corruption — exhausted retries are scored as a normal (wrong) answer
**File:** `src/eval_harness/suite.py`, `_complete_with_retries`.
```python
response = ""
for _ in range(max_retries):
    try:
        response = model.complete(prompt)
        break
    except Exception:
        continue
...
return response
```
**Why it's a bug:** if every attempt raises, the function returns `""`
with no indication anything failed. That empty string then gets scored
normally by whatever scorer the task uses — indistinguishable from the
model genuinely answering wrong. In an eval harness this corrupts the
signal: "the model failed to respond" and "the model responded
incorrectly" are very different findings, and this code silently
conflates them. Not exercised by `test_run_suite_retries_on_transient_failure`,
which only fails once and succeeds on the second attempt — no test drives
every attempt to failure. Fix: raise, or return a sentinel / mark the
`ExampleResult` as errored, rather than silently substituting `""`.
**Severity:** high — silently biases eval results with no way to detect it
after the fact.

### 3. Typing — `numeric_tolerance` can return `None`, crashing aggregation far from the cause
**File:** `src/eval_harness/scoring.py`, `numeric_tolerance`.
```python
except ValueError:
    return None
```
**Why it's a bug:** the `Scorer` type alias is `Callable[[str, Example], float]`
and `ExampleResult.score: float` (a frozen dataclass field). Returning
`None` when the response can't be parsed as a number violates that
contract. It won't crash where the bad value is created — it crashes later,
in `EvalResult.mean_score`'s `sum(r.score for r in ...)`, with
`TypeError: unsupported operand type(s) for +: 'float' and 'NoneType'`,
far from the actual root cause. No test passes an unparseable response
(e.g. "I don't know") to `numeric_tolerance`, so this ships invisibly.
Fix: return `0.0` for an unparseable response (treat "didn't answer with a
number" as simply wrong), not `None`.
**Severity:** high — a real model, asked a math question, absolutely will
sometimes answer in prose; this crashes the whole suite run on that input
rather than scoring it.

### 4. Design smell / silent failure — bare `except Exception: continue`, no logging
**File:** `src/eval_harness/suite.py`, `_complete_with_retries`.
**Why it's a bug:** every exception is swallowed identically — a
transient network error, a bug in the `ModelClient` implementation, an
auth failure, all look the same: silent, then retried, then (per #2)
silently scored as an empty wrong answer if retries run out. There's no
log of what failed, how many attempts were made, or why. This is the same
pattern class as the previous PR's silent `except` — the point is to check
whether it's recognized as a pattern (any place a bare `except` discards
information) rather than a single memorized location.
**Severity:** medium — compounds #2 by making the corruption in #2
undiagnosable after the fact.

### 5. Security — command injection via `os.system` in `_debug_log`
**File:** `src/eval_harness/suite.py`, `_debug_log`.
```python
def _debug_log(log_path, prompt, response) -> None:
    line = f"{prompt!r} -> {response!r}"
    os.system(f"echo {line!r} >> {log_path}")
```
**Why it's a bug:** `prompt` and `response` are raw, attacker- or model-
controlled text, interpolated directly into a shell command string passed
to `os.system`. Any prompt or response containing shell metacharacters
(`` ` ``, `$( )`, `;`, `|`) breaks out of the intended `echo` and executes
as a shell command with the privileges of the eval process. This is a
classic CWE-78 command injection, and it's a different vector from the
previous PR's `eval()` bug — the point is to check whether the reviewer's
security pattern-matching generalizes beyond "eval is dangerous" to
"shelling out with unsanitized input is dangerous," which is arguably the
more common real-world instance of this bug class. Especially serious
here: this is a debug-log path in an *eval harness*, and eval harnesses
for capability/red-team work routinely process exactly the kind of
adversarial, attacker-crafted text this pattern is most dangerous with.
Not exercised by any test or by `examples/run_suite.py` (neither passes
`log_path`), so it ships invisibly. Fix: write the file directly with
Python I/O (`Path(log_path).open("a").write(line + "\n")`), never shell out
for local file appends.
**Severity:** high — remote/attacker-influenced code execution.

## Categories covered
correctness/statistics (#1), correctness/silent-corruption (#2), typing
(#3), design smell/silent-failure (#4, intentionally same pattern class as
last round's #2 to test generalization), security (#5, different vector
than last round's `eval()`).

## Not planted, but fair game if flagged
- `_complete_with_retries`'s retry-count semantics are ambiguous:
  `max_retries=2` with `range(max_retries)` gives 2 *total* attempts, not
  2 retries *after* an initial attempt (i.e. 3 total). Neither the
  docstring nor the parameter name disambiguates this. Worth noting if
  raised, though it's a documentation/naming ambiguity rather than a
  functional defect given the current implementation is at least
  internally consistent.
- No test constructs a suite where every task is the same size (to prove
  #1's weighting bug doesn't currently matter) vs. a mixed-size suite —
  worth flagging as a test-coverage gap either way.
- `run_suite` builds one shared `_RetryingModel` wrapper across all tasks,
  meaning `max_retries`/`log_path` can't vary per task — a legitimate
  design/extensibility observation, not a planted bug.
