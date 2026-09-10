# Consolidation pass — 2026-09-10 (after 0.8)

Scope: the package after the 0.8 redesign and the user-experience pass.  No new capability, no
numerical method changed.  Every step kept the suite green, and every step that touched arithmetic
was verified against the baseline record, which reports the kernels bit-identical.

## The shape before and after

| measure | before | after |
|---|---|---|
| package lines | 14218 | 14317 |
| results.py | 1491 | 1238 |
| duplicated 6-line blocks | 8 site-groups | 0 |
| pyflakes on the package | 7 findings | clean |
| modules importing matplotlib | 3 | 2 (plotting.py, kernel.py) |

**The line count went up.**  That is the honest result: this pass removed duplicate *definitions*,
not lines, and each shared abstraction carries a docstring saying what it is for and why it is one
thing.  Three copies of a twenty-five line assembly became one function plus its explanation.

## What was consolidated

**Rendering is a module.**  results.py held 266 lines of matplotlib under the four Result classes,
plus a plot() body in each: a file about what a solve returns was also a rendering back end, and
matplotlib was reachable from anything holding a result.  Now `noisestate/plotting.py`, the only
module that imports it, with the four `plot_*(res, path)` functions and the payload plotters.
Nothing imports matplotlib at package import time (checked, not assumed).

**One summary format.**  The stationary, triangle and cell results each wrote their own `summary()`,
repeating the header line and the per-agent cost loop verbatim and diverging only in the grid
description, the cost's label and the means line.  A change to the header had to be made three
times, and the formats had already drifted — the triangle prints costs to eight figures and the
others to six.  `Result.summary()` is now the template; the engines supply `_grid_line()`,
`_means_line()`, `COST_LABEL` and `COST_FIGURES`.  The eight-figure triangle cost is kept as a class
attribute rather than quietly unified: that is a display decision, and not this pass's to take.

**The second-order form, built twice.**  Twenty-five lines assembling M = sum_k T_k' G_(k) T_k
appeared in `Engine._second_order` and again in `finite_free._dense_form`.  Only the assembly is
shared — how `Resp`, `Gk` and `forms` are obtained is exactly what differs between the two engines —
so only the assembly moved, to `engine.dense_curvature_form()`.  This decides `NOT A MINIMUM`, so it
was checked by capturing every agent's min, max and ok to full float repr over eight models on both
engines: identical character for character.

**The diagnostics fill, and the agent order.**  `_diagnostics` was duplicated between the stationary
and spectral engines, the second differing only by recording where the representation error sits.
The shared body is `EngineBase._fill_diagnostics` with a `_diagnostics_extra` hook.  The base
`_diagnostics` stays a no-op rather than becoming concrete: an engine without a decomposition (the
cell engine) must inherit nothing it cannot honour.  The reason this wants one definition is the
agent ORDER — a tied agent's representative has to be evaluated before its followers — which was
written out twice with nothing tying the copies together.

**The point-observation columns.**  Compiled's dense, per-panel and symmetric closed loops each
spelled out how a row's point observations reach the forcing columns, including the `nW + index`
offset for an impulse control.  Three copies of one mapping; the row selector was the only real
difference, so it became the argument.

**Placement and lint.**  `SweepPoint` moved from expr.py, the equations DSL, to sweep.py, its only
consumer — `MarchPoint` already sets that precedent next to the march.  The package is pyflakes
clean.

## Two bugs the pass surfaced

Neither was the point of the work; both were shipping.

1. **`noisestate plot` could not draw a finite payload.**  `plot_payload` read
   `payload["horizon"]["window"]` for its time axis, a key a finite horizon stopped carrying when
   0.8 split T from the lag window, so it raised `KeyError: 'window'` on every finite result.  The
   suite had payload tests for a stationary, a cells and a transition result and none for the plain
   finite triangle — which is the DEFAULT engine for a finite model.  The cells case takes the
   imshow branch and the transition short-circuits on kind before reading the horizon, so between
   them they covered every branch except the broken one.

2. **`from noisestate.expr import *` raised `AttributeError`.**  `__all__` still named `settings`,
   renamed to `using_settings` in 0.7.

## What was deliberately not done

**The dense/sparse pairs — revised, and then done.**  This section first said they stayed apart as
D1 territory.  That was wrong, and the distinction is worth writing down: D1 is about dense and
sparse *assemblies of different formulations*, where unifying them changes the arithmetic and the
last bit with it.  These pairs are not that.  Each is one operator built into two containers, from
the same calls with the same weights, where only the sink differs — so the traversal can be shared
with the arithmetic untouched, and the results are bit-identical by construction rather than by
tolerance.

Four were shared, in the second half of this pass:

* `map_shift` / `map_shift_sparse` → `Compiled._shift_ops`, the geometry deciding which pieces shift
  node to node, which are triangles cut by the buffer's diagonal, which lie beyond unit_range.
* `interp` / `interp_sparse` → `interp_factors`, which the sparse form already used while the dense
  one kept its own copy of the loop.
* `read` / `read_sparse` → `_read_zeros`, the predicate for which nodes must return zero.
* The `mass` / `mass_sparse` pairing was a false positive of the name heuristic: `mass` is a weight
  vector and `mass_sparse` is a Gram matrix whose dense twin no longer exists.  A stale comment
  still named the removed method.

**What that left behind.**  The sparse builder used to reach its misalignment error by *calling*
`map_shift(delay)` — allocating a dense N x N array for the side effect of raising, 1012 KiB on the
smallest delayed example and 62 MiB on a transition grid of 2850 nodes.  Forcing that branch now
shows peak allocation of 3.3 KiB against 1132 KiB.

And nothing had ever asserted that a dense form and its sparse twin agree — the invariant the whole
two-container design rests on.  Both pairs now have tests that compare them across the branches that
differ, each verified to fail against an injected regression rather than assumed to.

**Model was not split.**  `Model` is 1054 lines, of which 253 are twelve `_check_*` methods, and
spec.py is now the largest file.  Extracting the validation into its own module was considered and
rejected for this pass: nothing is duplicated there, the checks are intimately tied to the fields
they read, and moving them would be relocation rather than consolidation.  It remains a reasonable
thing to do on its own terms, as a readability change with its own justification.
