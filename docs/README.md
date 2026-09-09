# noisestate documentation

The [README](../README.md) is the user's document: install, a first solve, the model file in brief,
transitions, sweeps, the guards, the settings, the error contract and the CLI.  These pages hold the rest.

| page | holds |
|---|---|
| [model_file.md](model_file.md) | every key of a model file: type, meaning, default; the prose on atoms, states, definitions, signals, losses, ties, horizon and numerics |
| [payload.md](payload.md) | every key of `to_dict()` and the CLI's JSON, with its meaning and shape |
| [transitions.md](transitions.md) | a regime change from a stationary past: the construction, the result's fields, the identities that pin it, the two examples |
| [guards.md](guards.md) | the checks a result carries, the flag text each prints, its threshold and advice |
| [settings.md](settings.md) | the tuning constants of `noisestate.Settings`, each with its default and meaning |
| [validation.md](validation.md) | what the tests reproduce against the dissertation's solvers and closed forms, with the numbers; the resolved open items |
| [user_experience.md](user_experience.md) | what a first session runs into: the rough edges found by using the package, each with what it cost and whether it is fixed, deliberate or open |
| [method.md](method.md) | how it works: the best response and the fixed point, the stationary and finite forms, the means, the grids and caches, the stability guarantees |
| [limits.md](limits.md) | what the grammar and the engines do not do |
| [architecture.md](architecture.md) | the modules and what each holds; one best response and one transition followed through them |
| [design/](design/README.md) | the design record: the transition engine's plan and three design notes, the dissertation passages it implements, the size stage's brief, the consolidation plan and the API memo |

`CHANGELOG.md` at the root records every change by release.
