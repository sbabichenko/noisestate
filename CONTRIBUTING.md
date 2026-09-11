# Contributing

Run the test suite through `./run-tests`, not `pytest` directly:

    ./run-tests                      # the fast suite, about 2 minutes
    NOISESTATE_SLOW=1 ./run-tests    # everything, about 6 minutes

The wrapper takes a lock so two suites cannot overlap, and caps the BLAS threads.  Both matter: two
suites at once corrupt every timing they touch, and an uncapped BLAS runs these small solves 16-way
and takes six times as long.

Compare against `HEAD` before calling something a regression, and use a git worktree rather than
`git stash`.

`tests/refs/` holds 5.4 MB of reference data, including the bit-identity baseline that
`tests/test_baseline.py` checks every shipped example against.  It is excluded from the published
distribution, so clone the repository to run the suite.
