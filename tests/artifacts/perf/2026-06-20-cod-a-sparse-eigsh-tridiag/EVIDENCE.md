# 2026-06-20 cod-a sparse eigsh projected-residual certificate

Bead: `frankenscipy-8l8r1.131`
Agent: cod-a / BlackThrush

## Lever

Use the symmetric Arnoldi/Lanczos projected tridiagonal residual certificate
for `eigsh` convergence instead of always recomputing `k` full sparse
`A*x - lambda*x` residual matvecs after the eigensolve. This removes six
post-hoc sparse matvecs for the live `k=6` rows.

The lever is guarded to `k <= 6`. A same-worker sample showed the raw projected
residual path regressed the `k=8` row despite fewer matvecs, so the final source
keeps the older explicit residual check for `k > 6`.

## Same-worker rch proof

Command:

```text
AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo run --release -p fsci-sparse --bin perf_eigsh
```

Baseline/current and candidate were both run on `vmi1227854`.

| Workload | Baseline Rust | Candidate/final k<=6 route | Internal delta | Prior SciPy oracle | Candidate vs SciPy |
| --- | ---: | ---: | ---: | ---: | --- |
| `eigsh n=2000 k=6` | 1.169 ms, 26 matvecs | 1.024 ms, 20 matvecs | 1.14x faster | 3.000 ms | Rust 2.93x faster |
| `eigsh n=8000 k=6` | 4.789 ms, 26 matvecs | 4.003 ms, 20 matvecs | 1.20x faster | 2.768 ms | Rust 1.45x slower |
| `eigsh n=20000 k=8` raw projected candidate | 10.672 ms, 28 matvecs | 12.289 ms, 20 matvecs | 1.15x slower | 43.023 ms | reject raw k=8 route |

Decision: keep the `k<=6` projected-residual certificate; reject the
unconditional route and guard `k>6` back to explicit residual checks.

## Final-source sanity

Final source, rch worker `vmi1152480`:

| Workload | Final Rust | Matvecs | Converged | Max residual |
| --- | ---: | ---: | --- | ---: |
| `eigsh n=2000 k=6` | 1.091 ms | 20 | true | 1.96e-11 |
| `eigsh n=8000 k=6` | 4.797 ms | 20 | true | 3.45e-11 |
| `eigsh n=20000 k=8` | 12.042 ms | 28 | true | 2.57e-11 |

Fresh local SciPy oracle on Python 3.13.7 / NumPy 2.4.3 / SciPy 1.17.1:

| Workload | SciPy median | Final remote Rust vs fresh SciPy |
| --- | ---: | --- |
| `eigsh n=2000 k=6` | 1.538154 ms | Rust 1.41x faster |
| `eigsh n=8000 k=6` | 3.424127 ms | Rust 1.40x slower |
| `eigsh n=20000 k=8` | 7.874257 ms | Rust 1.53x slower on this cross-host oracle |

The fresh SciPy rows are recorded as oracle drift/routing evidence because the
Rust timings above were remote while SciPy ran locally. A local Rust release
run against the shared target was attempted but blocked by worker-built release
artifacts from a newer nightly (`E0514`, worker rustc `beae78130` versus local
`f20a92ec0`). The target was not cleaned.

## Gates

- PASS: rch `cargo check -p fsci-sparse --all-targets`
- PASS: rch focused `cargo test -p fsci-sparse eigsh --lib -- --nocapture`
- PASS: rch `cargo clippy -p fsci-sparse --all-targets --no-deps -- -D warnings`
- PASS: local SciPy-backed `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test diff_sparse_eigsh_largest -- --nocapture`
- PASS: `git diff --check`
- PASS: `ubs crates/fsci-sparse/src/linalg.rs` completed with 0 critical findings and no new hunk-specific finding
- BLOCKED: `cargo fmt -p fsci-sparse -- --check` remains blocked by pre-existing sparse crate formatting drift in `sparse_bench.rs`, `src/lib.rs`, and older `src/linalg.rs` hunks outside this patch

## Retry condition

Do not retry row-major Arnoldi basis arenas, mutable operator scratch, or
unconditional projected residual acceptance for `k>6` without a same-worker
benchmark. Next sparse eigsh work should target the remaining mid-size
`n=8000, k=6` loss with a deeper primitive: implicit restart/thick restart,
a symmetric tridiagonal-only eigensolver path, or deterministic warm-start
subspace reuse.
