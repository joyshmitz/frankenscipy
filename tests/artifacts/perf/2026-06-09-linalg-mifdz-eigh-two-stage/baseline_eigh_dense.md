# frankenscipy-mifdz baseline: dense `eigh`

## Command

```text
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked -- eigh_dense
```

Worker: `vmi1227854`

## Criterion baseline

```text
eigh_dense/256x256      time: [13.303 ms 13.677 ms 14.019 ms]
eigh_dense/512x512      time: [103.50 ms 105.45 ms 107.58 ms]
```

Raw artifact:

```text
tests/artifacts/perf/2026-06-09-linalg-mifdz-eigh-two-stage/baseline_criterion_eigh_dense_rch.txt
sha256=73aa1b1085bbfed6103f38f024eeee4f7488c96342079baecb13a3063e9798c7
```

## Route

The current public `eigh` path delegates dense symmetric eigensolve to
`nalgebra::symmetric_eigen()`. The `frankenscipy-mifdz` implementation target is
a safe-Rust two-stage blocked tridiagonalization path:

1. Reduce full dense symmetric matrix to band form with blocked Householder
   panels.
2. Reduce band to tridiagonal form with local bulge chasing.
3. Solve the tridiagonal eigenproblem and replay accumulated transformations.

Keep only with same-worker RCH improvement and unchanged eigenvalue ordering,
eigenvector sign/tie contract, reconstruction tolerance, and golden output hash.
