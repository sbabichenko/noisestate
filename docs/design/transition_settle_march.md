# Choosing the transition horizon by a march in T

Design note, 2026-09-06, from a discussion with the maintainer.  Not built yet.  Records the decision to
make the transition's window an output of a settle tolerance rather than an input, and how.

## The problem in three regions

A regime that begins at date zero inherits a past.  Before zero the economy is in the stationary
equilibrium of the old model; its kernels depend on a shock only through its age.  At date zero the
model changes.  The shocks that arrived before zero remain in the state, in what every agent has
observed, and in both until they are older than the window.  The problem is the equilibrium of the
new model from date zero on, given that inheritance.

On the domain of pairs (t, s), s the arrival date of a shock and t the date its effect is measured,
with s <= t and t - s <= L, three intervals of t play distinct roles.

For t < 0 nothing is solved.  The old kernels enter only through their values on the line t = 0: for
each shock arrived before zero, the loading of every state, of every lagged quantity a drift or a
loss reads across zero, and of each agent's observations.  The first two are the initial condition of
the state; the third is what each agent already knows.

For 0 <= t <= T the strategies are unknown.  Each agent's action is a linear functional of its
observations, those made before zero included, and the equilibrium condition is that of any finite
horizon: best responses, the first-order condition at each date balancing the instantaneous loss
against the discounted continuation, the projection onto the agent's own information fixing the
weights it may use.  Shocks arrived before zero evolve from their inherited loadings, shocks arriving
after zero from impulses at their arrival dates; both families obey the same dynamics, first-order
conditions and truncation at age L.

For T < t <= T + L the strategies are held at the new model's stationary equilibrium.  This closes
the problem without ending it: a date near T has a future, and the continuation term of its
first-order condition integrates over that future under the stationary rules.  What is fixed is the
rules, not the stationary world: the world after T is whatever the transition has produced by T,
continued under stationary behaviour.  Past T + L every shock arrived before T is forgotten.

The date T is not part of the economics.  It is where the unknown strategies hand over to the
stationary ones, and the size of the handover, the distance between the strategies on the last window
and the stationary ones (`settled`), is the diagnostic.  A short T gives the exact solution of a
slightly different problem, wrong only near T: the end effect leaks back at the closed-loop rate, so
the early part of a short-T solution is already right.

## The monitor is a best-response pass

The assumption past T is that every player keeps the stationary rules.  The check is one best-response
pass: given that assumption and the transition solved on [0, T], compute each player's optimal rule
from its first-order condition, continuation values included, and measure its distance from the
stationary rule on the last window.  Once the transient has passed the two coincide.  This is the
same object as the adjoint gap (the first-order condition is the adjoint applied to the responses),
and it is what the same-model identity tests already compute as the one-shot deviation: at T = 0 the
diagnostic is a single best response from the stationary rules with the inheritance attached.

## The march

Start at T = 0.  There is no unknown: the strategies are the stationary ones from date zero, the
world is the inherited shocks evolving under them (one closed loop over the band), and the adjoints,
the backward kernels that price the state and the beliefs in each first-order condition, follow from
one backward pass.  Their distance from the stationary adjoints is the first-order condition's
residual at date zero: how far the stationary rules are from optimal given the inheritance.  Within
tolerance means the stationary equilibrium is the transition; stop before solving anything.

Otherwise grow T.  Each extension adds a stretch of unknown strategies and moves the frozen region
forward.  Warm-start the fixed point at the new T from the old one, with the stationary rules on the
new stretch; because the end effect is local, it converges in a few evaluations that do their work on
the last window.  The monitor is the adjoint gap one window before the new T (the gap at T itself is
the handover and is never zero): it falls at the closed-loop rate as T grows, so it is also the step
control: small steps while it is large, a whole window at a time once it falls steadily, stop when it
is under tolerance.  Steps are multiples of the unit so the new panels align with the delay cuts and
the buffer.  Nothing is predicted; the rate shows up in the sequence of gaps.

## What makes it efficient

Reuse.  A step by whole units adds time panels at the end; the panels before are unchanged, and so
are their path factors, closed-loop row blocks and preconditioner blocks.  A grid that grows by
panels, with caches keyed by panel rather than by grid, makes a step cost the new panels plus the
buffer, and the whole march about one solve at the final T plus a small overhead per step.  Without
that reuse (today: a fresh grid per solve) the steps should be coarse, a window at a time, and the
warm start alone (8 evaluations against 21 from scratch when measured) carries the saving.

Tail extrapolation.  The excess cost of the transition has a piece past T that decays at the
closed-loop rate; its integral is the last window's excess times a known factor.  Reporting the
excess cost with that extrapolated tail makes the number converge in T much faster than the
strategies, and the number is usually what is wanted.

## Surface

`transition(old, new, settle=1e-4)` (and the `transition` horizon kind with `settle:` instead of
`window:`): the window is returned in the result (`res.extra["window"]`, `res.settled`), the
`settled` guard stays for an explicit window.  Two identities check the construction: with no
inheritance the problem is the ordinary finite one; with the old model equal to the new one the
solution is the stationary equilibrium at every point, which the march finds at T = 0.

## Alternatives considered and set aside

Predicting T from the new kernels' decay rate and the regimes' mismatch, then verifying once: works
but fits a rate the march observes directly.  Linearising the best-response map at the new
equilibrium and solving the transition as one linear (block-Toeplitz-in-time) problem: exact only to
first order in the change, and its matrix-free form costs one strip evaluation per Krylov step, no
better than Anderson unless the translation invariance is exploited; shelved.
