<!-- Design record, copied verbatim from the consolidation pass's working directory (consolidation/plan.md), 2026-09-06. -->

# Consolidation pass — plan (2026-09-06)

Scope: the noisestate package after v0.4.0 + the size stage (master faaffbf) and the pending
matrix-free branch (`focfree`, adds noisestate/finite_free.py, ~750 lines).  No new capability.
Every step keeps the suite green and the closed-form guards exact; the shipped examples are
compared against master after every commit (bit identity where the step allows it; see D1).

## The shape today

| measure | value |
|---|---|
| package lines | 8499 (+750 pending) |
| finite_spectral.py | 2170 lines, 113 methods in two classes |
| longest function | SpectralCompiled.__init__, 174 lines |
| closed-loop assemblies | dense, per-panel, past/band, buffer (+ matrix-free pending) |
| maps_from_world | two copies (with and without a past) |
| mean layer | three copies (stationary, spectral, cells) of mean_system / solve_means / mean_cost |
| kernel-algebra interface | 17 method names shared by the stationary and spectral compiled classes by convention |
| tests | 37 files, 3241 lines, ~230 s fast suite, 10 slow-gated |
| README | ~640 lines |

## Decisions to take first

**D1. One assembly path or three.**  Bit identity between the dense and per-panel closed loops is
impossible on this BLAS (a sliced product differs in the last bit).  Keeping the dense path as the
reference means every operation exists in three forms forever.  Recommendation: one assembly
(per-panel, with "all panels" as the dense case), accept a one-time last-bit change, re-baseline the
bit-identity script, and rely on the closed-form guards (exact delay, discounted Riccati, same-model
identities, the mean sweep) as the correctness bar.  Cost: every pinned number in tests must be checked
to hold at its stated tolerance (they are 1e-8 or looser; none pins bits).  If Sam prefers the
reference path kept, steps C2 and C3 shrink to "share the block sources" and the three paths stay.

**D2. Version.**  0.4.1 if bits are unchanged (D1 rejected), 0.5.0 if re-baselined (D1 accepted).

## Steps (one agent each, sequential, suite + guards after every commit)

C0. **Wait for the matrix-free branch** to land or be shelved; consolidating under it would conflict.

C1. **Tests first** (half a day).  tests/helpers.py with the builders every transition test repeats
(the same-model past/continuation setups, the example loaders, the master-comparison script);
the fast suite target under 120 s by moving every solve above ~5 s that duplicates a pin under
NOISESTATE_SLOW; one place that lists what each slow test pins.  Nothing else changes, so this is
the safety net for the rest.

C2. **One closed-loop assembly** (1 day; the D1 step).  A "row block source" per time panel that the
dense, band, buffer and matrix-free cases all draw from; `closed_loop` = assemble(panels) + causal
solve; `_closed_loop_past` and `_closed_loop_panels` disappear.  Guards: every closed form; the
same-model identities to their floors; the examples to 1e-12 (re-baselined bits under D1).

C3. **One best response** (1 day).  `_maps_from_world_past` folded into `maps_from_world` (the past
is an optional row segment); the matrix-free operators become the only definition of the row,
response, continuation and projection operators, with "dense" meaning "apply to the identity" only
where a dense matrix is genuinely cheaper (below the threshold), so the operator algebra is written
once.  Also vectorise TriangleGrid.path (today a Python loop over every output node with per-node
cuts: 140 s per path at 29k nodes; numpy array operations over all nodes first, a compiled step
only if measured necessary).  PROFILE at the two target sizes (the ch1 transition at 5 and 7 nodes,
the two-firm market): wall time split into interpreted Python (path construction, per-node
bookkeeping, sparse assembly) versus BLAS/LAPACK/scipy compiled time (cProfile plus the
threadpoolctl-free BLAS timing by wrapping the dense calls), reported as a table, so the decision
on a compiled step (numba/Cython) is made on numbers.  Guards as C2 plus the Krylov-vs-dense agreement test.

C4. **Split the file** (half a day).  finite_spectral.py into: grid-facing operators (the kernel
algebra), the closed loop, the best response, the means on the time line, the solver class; each
under 800 lines; `SpectralCompiled.__init__` split into named steps (breakpoints and closure, grid,
past wiring, buffer wiring, caches).  Pure moves: bit identity required here.

C5. **One mean layer** (half a day).  mean_system / solve_means / mean_cost in EngineBase over a
small per-engine interface (the mean dynamics operator, the mean FOC operator, the initial
condition); the three copies go.  Guards: the mean closed forms, the Chapter 1 sweep, the
same-model means.

C6. **The kernel algebra as an explicit interface** (half a day).  An abstract base listing the 17
operators with their shapes (from the hook-contract docstrings), implemented by both compiled
classes; unused members deleted; results.py reads engines only through it.

C7. **Docs** (half a day).  README to a user's document (~250 lines: install, use, model file,
transitions, guards, settings); the validation tables, limits detail and method notes to docs/ files
linked from it; CHANGELOG untouched.

C8. **Skeptic** over the whole diff: behaviour changes, dropped checks, anything the guards do not
cover; then merge and tag.

Total: about 5 days of agent time, sequential, plus reviews.

## Not in this pass (follow-ups, each its own stage)
- The second-order check at a positive discount (the discounted quadratic form).
- The diagonal quadrature artefact at small trading cost (a product rule for an interpolant times
  its own Volterra integral on s = 0).
- Vector states.
- Path-factor storage (the remaining memory bulk at large N).

## Metrics to hold at the end
No function over 80 lines; no file over 900 lines; one closed-loop assembly; one best response;
one mean layer; fast suite under 120 s; every pinned number within its tolerance; every closed
form exact to its stated floor.

## API stage (approved by Sam 2026-09-06: forks A–G of api_design.md all accepted)
Lands after C4 (the module split), before C5–C7 so the mean layer and the docs are written once
against the new surface.  Deliverables: Numerics object and the `numerics:` block (old nested keys
accepted with a deprecation note for one version); explicit solve() signature; one Result type with
named axes, res.paths, res.status, res.extra; noisestate.schema() for the model file and the payload
and `noisestate schema` in the CLI; names evaluations/world with aliases; one transition default;
CLI transition/schema/plot.  Version 0.5.0.  Guards: the whole suite, the baseline check (costs,
evaluation counts, 12-digit hashes unchanged: an API change must not move a number), the examples
and the payload round trips rewritten to the new surface with the old forms tested through the aliases.
