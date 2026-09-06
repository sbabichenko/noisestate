<!-- Design record, copied verbatim from the consolidation pass's working directory (consolidation/api_design.md), 2026-09-06. -->

# noisestate API: how it should function (design memo, 2026-09-06)

## The surface today
solve(model, refine, stability, **kw routed by introspection to the engine constructor or its solve());
sweep(model, param, values, solver_kw, solve_kw); transition(old, new, T, nodes, continuation, stationary, **kw);
load/read_yaml/read_json; make_solver; ENGINES; clear_grid_cache; Settings (27 tunables);
Model / ModelBuilder (param, channel, state, define, agent, signal, tie, stationary|finite|transition, build);
three engine classes exposed; four result classes (Stationary, Triangle, Transition, Cell) with overlapping
but different attributes (maps vs Z, ages vs times, kernel() shapes differ, map_axes as the workaround);
CLI: solve / validate / sweep; a 650-line README as the only reference.

## What is wrong with it
1. The model file mixes the economics (states, agents, horizon kind, window, discount, past) with the
   numerics (nodes, unit, unit_range, breakpoints, the cell-engine kind).  Changing the resolution is a
   model edit; refine() is a special case of a numerics change; the cell engine is a "horizon kind".
2. solve(**kw) routes by inspecting signatures: options are undiscoverable, the two layers
   (constructor vs solve) are invisible, and every engine-specific option is a hidden branch.
3. Results are four shapes.  A consumer must know which engine ran to read a kernel.
4. Three ways to say "transition" (helper, keyword, file block) with slightly different defaults
   (start="stationary" vs "zero").
5. Names: iterations means evaluations; Z vs maps vs actions; costs means a flow (stationary) or an
   integral (finite) with cost_kind to tell them apart.
6. No machine-readable schema for either the model file or the payload; the front end must read code.

## Forks (decide these; the rest follows)
A. Separate economics from numerics: Model (the problem) + Numerics (engine, nodes, unit, unit_range,
   breakpoints, tolerances, settings).  solve(model, numerics=None).  Sweeps and refine() vary
   numerics; the model file keeps `horizon: {kind, window|T, discount, past, continuation}` and gains an
   optional `numerics:` block, with the old nested keys accepted and deprecated.  RECOMMENDED.
B. Explicit solve() signature: solve(model, *, numerics, init, start, tol, max_evaluations, deadline,
   progress, diagnostics, refine, stability, verbose); no introspection; engine classes stay public under
   noisestate.engines for power users.  RECOMMENDED.
C. One Result type with named axes: res.kernel(q, channel) returns an array plus res.axes (time/age or
   age; shock time for a band); res.paths (means, loss path, belief errors as (name -> path over
   res.times)); res.costs + res.cost_kind + res.cost_parts; res.status (ok, flags, diagnose rows);
   engine-specific extras under res.extra; to_dict() is the versioned payload.  The four classes become
   internal.  RECOMMENDED; it is the front end's contract.
D. Schema: noisestate.schema() emitting JSON Schema for the model file and for the payload; the CLI's
   `validate` uses it; the front end validates against it.  RECOMMENDED, cheap.
E. Naming: evaluations (iterations kept as an alias for one version), world (Z kept as alias), the
   rest unchanged.  RECOMMENDED with aliases.
F. Layers for transitions: keep all three but with one default (start="stationary" everywhere a
   continuation is given).  RECOMMENDED.
G. CLI: solve / validate / sweep / transition / schema / plot.  Optional.

## What this costs
A+B+C+D+E is a 0.5.0 API break with aliases for one version: about 3 agent-days after the
consolidation's C4 (the module split) so that it lands on the tidied engine, not before.
Docs: README to ~250 lines of user document; docs/model_file.md, docs/payload.md (from the schema),
docs/validation.md, docs/method.md.
