# frankenscipy-i0ghz - pdist Chebyshev dim-4 fast path

Date: 2026-06-20
Agent: cod-b / BlackThrush
Decision: KEEP

## Lever

`pdist(..., DistanceMetric::Chebyshev)` for dim-4 now uses the same
structure-of-arrays row staging as the existing dim-4 Euclidean, cosine,
sqeuclidean, and cityblock fast paths. The row filler computes eight condensed
pairs per SIMD batch, performs a branchless max tree across four absolute
coordinate deltas, and explicitly selects `NaN` when any coordinate delta is
`NaN`, preserving the scalar helper semantics.

The chosen route follows the alien-graveyard/vectorized-layout guidance:
specialize the fixed-width hot shape, keep the memory layout vector-friendly,
and prove the exact SciPy-facing contract rather than trusting a broad profile.

## Head-to-Head Evidence

Primary optimized sweep:

```text
AGENT_NAME=BlackThrush RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
  rch exec -- cargo run --release -p fsci-spatial --bin perf_pdist_sweep
```

SciPy oracle:

```text
python3 - <<'PY'
from scipy.spatial import distance as sd
...
PY
```

| Workload | Prior Rust | Final Rust | SciPy 1.17.1 | Verdict |
| --- | ---: | ---: | ---: | --- |
| `pdist/chebyshev/n512/d4` | 1.142 ms | 0.139 ms | 0.176 ms | KEEP: 8.22x faster than prior Rust; final Rust 1.27x faster than SciPy |
| `pdist/chebyshev/n512/d16` | 3.841 ms | 3.433 ms | 0.712 ms | LOSS remains: final Rust 4.82x slower than SciPy |
| `pdist/chebyshev/n512/d64` | 11.092 ms | 9.517 ms | 2.176 ms | LOSS remains: final Rust 4.37x slower than SciPy |
| `pdist/chebyshev/n2048/d64` | 90.382 ms | 124.573 ms | 37.941 ms | LOSS remains/noisy: final Rust 3.28x slower than SciPy; unchanged generic path |

Related final-source rows from the same sweep remain useful release context:

| Workload | Final Rust | SciPy 1.17.1 | Verdict |
| --- | ---: | ---: | --- |
| `pdist/euclidean/n512/d4` | 0.151 ms | 0.309 ms | Rust 2.05x faster |
| `pdist/cityblock/n512/d4` | 0.105 ms | 0.192 ms | Rust 1.83x faster |
| `pdist/sqeuclidean/n512/d4` | 0.103 ms | 0.177 ms | Rust 1.72x faster |
| `pdist/euclidean/n4096/d4` | 20.430 ms | 51.920 ms | Rust 2.54x faster |
| `pdist/cosine/n4096/d4` | 19.249 ms | 49.421 ms | Rust 2.57x faster |
| `pdist/cityblock/n2048/d64` | 20.050 ms | 46.723 ms | Rust 2.33x faster |

Criterion focused bench:

```text
cargo bench -p fsci-spatial --bench spatial_bench -- pdist/chebyshev/512 --noplot --sample-size 10 --measurement-time 1 --warm-up-time 1
```

The contended RCH Criterion row reported `[222.03 us, 237.98 us, 257.02 us]`,
which is 1.35x slower than the local SciPy oracle. Treat that row as noisy
supporting evidence only; the same-run optimized sweep above is the keep gate.

## Correctness And Gates

| Gate | Result | Artifact |
| --- | --- | --- |
| Format | PASS | `cargo fmt --check -p fsci-spatial` |
| Check | PASS | `cargo-check-fsci-spatial.txt` |
| Spatial clippy, no deps | PASS | `cargo-clippy-fsci-spatial-no-deps-retry.txt` |
| Spatial lib tests | PASS: 214 passed / 0 failed / 2 ignored | `cargo-test-fsci-spatial-lib.txt` |
| Focused post-UBS test cleanup | PASS: 1 passed / 0 failed | `cargo-test-delaunay-find-simplex-many.txt` |
| Spatial E2E conformance | PASS: 16 passed / 0 failed | `cargo-test-e2e-spatial.txt` |
| SciPy pdist/cdist differential | PASS locally with SciPy 1.17.1: 1 passed / 0 failed | `cargo-test-diff-spatial-pdist-cdist-local.txt` |
| Changed-file UBS | PASS: exit 0, 0 critical | `ubs-changed-files.txt` |

`cargo clippy -p fsci-spatial --all-targets -- -D warnings` remains blocked
before this patch by existing `fsci-linalg` dependency lints. The same
pdist/cdist differential test failed under RCH only because worker
`vmi1149989` has no Python SciPy package; the local SciPy-backed run passed.

## Negative Evidence

Do not route more work into dim-4 Chebyshev for `n=512` without a new measured
regression. The remaining `pdist` Chebyshev losses are high-dimension generic
path losses: d16 and d64 are still 3.28x to 4.82x slower than SciPy and need a
dimension-loop/vector-reduction specialization, not another dim-4 branch.
