# User experience: what a first session runs into

> **A dated log, not a task list.**  Walked 2026-09-09 against **0.6.9**; the dispositions were
> re-checked on 2026-09-10 against 0.8.0 and are marked inline.  Names in the quoted output are
> those of 0.6.9: `check()`, `res.status` and `resolution_ok` were removed in 0.7 and 0.8 and are
> reproduced here as they were said at the time.  The current spellings are in the
> [README](../README.md) and [docs/api.md](api.md).

A walk through the package as a new user.  Each item is what was observed, what it cost, and what was
done.  The measurements are reproducible from the commands quoted.

The short version: the diagnostics are the strength here.  The friction is that the shipped defaults
trip them, and the one call that looks like "is this result sound?" does not consult them.

## Fixed

### An agent with no signal rows crashed inside LAPACK

`signals` is optional in the schema and validation checked `controls` but had no counterpart for rows, so
a signal-less agent reached the linear algebra:

    RuntimeWarning: invalid value encountered in scalar divide
     ** On entry to DSYRK  parameter number 10 had an illegal value      (x3, on stderr)
    IndexError: index 0 is out of bounds for axis 0 with size 0

Now `agent a has no signal rows: a strategy reads its rows, so there is nothing to solve for (give it a
row, or drop the agent and its controls)`.  Guarded by `tests/test_guards.py`.

### A saddle and a coarse grid read the same on a finite horizon

`kyle_back_prior` flags `NOT A MINIMUM` on trader1 at every resolution --- while the curvature it reports
goes to zero: `-1.45e-02, -1.02e-02, -8.26e-03, -6.71e-03, -5.63e-03` at 8, 12, 16, 20, 24 nodes, about
`n^-0.85`.  The offending eigenvector sits on the diagonal `a = t` and alternates in sign between
neighbouring age nodes: the quadrature's direction, not a strategy.

The stationary engine already settled the analogous window question with its embedded-curvature hook; the
finite engines had nothing.  `res.refine()` now records the finer grid's curvature in
`res.refinement["second_order"]`, and a shrinking one overturns the verdict (`second_order_grid:` says
which way it went).  The cost is the refinement, not a solve.

### A copied transition silently read the original past

`save()` wrote the past's absolute path, so a pair copied elsewhere resolved back to the original file ---
no error, different numbers.  Demonstrated by editing the copied past to `p1 = 99.0` and watching the
reload still read `3.0`.  A past under the saved file's own directory is now written relative to it; a
past elsewhere keeps the absolute path, which is what it means.

### The fast suite had drifted to six times its budget

`tests/SLOW.md` documented "under two minutes" and per-test seconds "at four BLAS threads", but nothing
set the threads: 743 s against 118 s capped.  `tests/conftest.py` caps them, and `./run-tests` caps them
again in the environment.

### `check()` did less than its name implied, and the CLI said 0 either way

`res.require_converged()` tested convergence only, so it passed on a result whose guards had failed; `noisestate
solve` exited 0 on the same result, and a script reading only the exit status would have taken it as
sound.  `res.require_ok()` is now check() plus the guards, `noisestate solve --require-ok` is the same
split on the command line, and `noisestate describe model.yaml` reaches the model explanation that was
only available from Python.  (Found by an external review, [docs/design/reviews/2026-09-09-external-ux-review.txt](design/reviews/2026-09-09-external-ux-review.txt), which also notes that
`noisestate plot` used to re-solve the model rather than plotting the stored payload --- the help text said
so, but the name did not.  It now renders the saved kernels and transition paths directly; `--re-solve`
asks explicitly for the old reproduction path.)

## Open, and deliberate

### ~~`check()` passes while `status["ok"]` is False~~ --- RESOLVED in 0.8

*As observed (0.6.9):* `res.require_converged()` raises only when the fixed point did not converge.  It
says nothing about the guards, so on `ch3_two_player` it passes happily while `status["ok"]` is `False`.
It is the obvious call to reach for and the one most likely to be misread as "the result is sound".

*Now:* the two questions have two names and neither is the obvious-but-wrong one.  `require_converged()`
says only what it says; `require_ok(policy)` is the full assessment and is the call the README puts in
front of a reader.  `status` is gone: a collection of checks does not have one boolean, and
`res.diagnostics.assess()` reports which checks blocked and why.

### Four of the seven shipped examples flag on their defaults

    ch3_two_player         WINDOW TOO SHORT
    kyle_back_prior        NOT A MINIMUM
    ch5_cycle_market       UNDER-RESOLVED
    ch3_precision_change   UNDER-RESOLVED; PAST WINDOW TOO SHORT; TRANSITION NOT SETTLED

Not carelessness, and mostly not fixable by tuning:

* `ch3_two_player` carries the dissertation's `window: 3.0`, and `test_ch3_matches_spectral_solver`
  cross-checks its kernels against a reference from the dissertation's own solver **at L = 3**.  Raising
  the window decouples the benchmark.  `window 9, nodes 32` clears both guards with room (tail 5e-05,
  representation 9e-12) and moves the cost 0.4% --- the truncation, not noise.
* `kyle_back_prior` is the grid, as above.
* `ch5_cycle_market` sits at representation error 1.3e-05 against a 1e-06 threshold and does not reach it
  at 10 or 12 nodes either (9.9 s, 25 s): more than an example should cost.
* `ch3_precision_change`'s `PAST WINDOW TOO SHORT` is inherited --- its past is `ch3_two_player`.

Each example now says which flag it raises and why, and the README's first solve shows the guard working
rather than tripping over it.  Deciding to raise the shipped windows anyway is a judgement about the
benchmark, not about the code.

## Open

*Re-checked 2026-09-10 against 0.8.0: all three below are still open.*

### `res.second_order` does not expose the offending direction

It reports `{min, max, ok, converged}`.  Finding *which* deviation is negative --- the thing that tells a
saddle from a quadrature artefact by eye --- meant hooking `_dense_form` and swapping `eigvalsh` for
`eigh`.  The eigenvector for the smallest eigenvalue is the natural companion to the number, and the
refinement test above now infers what it shows without exposing it.

### A bare `except` turns a broken check into a missing verdict

`stationary.py`'s window-edge embedding is wrapped in `except Exception: return None`.  When a signature
change broke the call during the 0.6 work, the check did not fail --- it silently reported no verdict, and
surfaced as an unrelated curvature assertion in `test_stability`.  Narrowing it, or logging what it
caught, would make the next one visible.

### `docs/model_file.md` says it is generated, but no generator is committed

Its header reads "generated from `noisestate.schema("model")`".  It is maintained by hand, so it can drift
from the schema --- it did, over the deprecated keys, until 0.6.9.  Commit the generator or drop the claim.
