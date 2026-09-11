# noisestate documentation

Start with the [README](../README.md) for installation and a worked example.
These pages provide detailed reference material, numerical methods, and development history.

| page | holds |
|---|---|
| [api.md](api.md) | every name `import noisestate as ns` gives you, grouped by the job it does: getting a model in, changing it, solving, reading the result, deciding whether to believe it, families of solves, and the CLI |
| [model_file.md](model_file.md) | every key of a model file: type, meaning, default; the prose on atoms, states, definitions, signals, losses, ties, horizon and numerics |
| [payload.md](payload.md) | every key of `to_dict()` and the CLI's JSON, with its meaning and shape |
| [transitions.md](transitions.md) | a regime change from a stationary past: the construction, the result's fields, the identities that pin it, the two examples |
| [guards.md](guards.md) | the checks a result carries, the flag text each prints, its threshold and advice |
| [settings.md](settings.md) | the tuning constants of `noisestate.Settings`, each with its default and meaning |
| [validation.md](validation.md) | what the tests reproduce against the dissertation's solvers and closed forms, with the numbers; the resolved open items |
| [notes/](notes/README.md) | technical notes on the best-response Jacobian and spectral radius, with numerical examples |
| [user_experience.md](user_experience.md) | historical usability review of version 0.6.9, with follow-up notes for 0.8.0 |
| [method.md](method.md) | how it works: the best response and the fixed point, the stationary and finite forms, the means, the grids and caches, the stability guarantees |
| [limits.md](limits.md) | what the grammar and the engines do not do |
| [architecture.md](architecture.md) | the modules and what each holds; one best response and one transition followed through them |
| [design/](design/README.md) | the design record: the 0.8 API specification and what it left behind, the transition engine's plan and design notes, the dissertation passages it implements, the size brief, the consolidation plan, and dated reviews kept as history |

`CHANGELOG.md` at the root records every change by release.
