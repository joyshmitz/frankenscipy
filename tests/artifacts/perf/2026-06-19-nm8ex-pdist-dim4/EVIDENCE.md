# frankenscipy-nm8ex: spatial pdist dim-4 fast path

Agent: cod-a / MistyBirch
Date: 2026-06-19
Crate: `fsci-spatial`

## Lever

Specialize `pdist` for the measured 4-D Euclidean and Cosine rows. The kept
path uses direct dim-4 squared-distance, dot, and squared-norm helpers so the
hot all-pairs loop avoids metric dispatch, generic slice reductions, and
SIMD-tail setup. It also forces these now-cheap dim-4 rows through the serial
gate at n=256/512, avoiding thread-spawn overhead. The existing pair-balanced
condensed-output row split remains available for other metrics and shapes.

No unsafe code was added.

## Bench Commands

Baseline and candidate:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a \
  rch exec -- cargo bench -p fsci-spatial --bench spatial_bench -- pdist --noplot
```

SciPy oracle:

```bash
python3 docs/perf_oracle_pdist.py
```

## Same-Worker Internal A/B

Both baseline and candidate were measured on rch worker `hz2`.

| Workload | Baseline mean | Candidate mean | Speedup | Verdict |
| --- | ---: | ---: | ---: | --- |
| `pdist/euclidean/256` | 2.9264 ms | 107.45 us | 27.24x | keep |
| `pdist/cosine/256` | 3.3658 ms | 114.04 us | 29.51x | keep |
| `pdist/euclidean/512` | 3.5554 ms | 425.75 us | 8.35x | keep |
| `pdist/cosine/512` | 2.5548 ms | 461.16 us | 5.54x | keep |

Criterion reported statistically significant improvements for all four rows
with p=0.00.

## Head-To-Head Vs SciPy

Local SciPy oracle: SciPy 1.17.1, NumPy 2.4.3.

| Workload | Rust candidate mean | SciPy p50 | Rust/SciPy time | Verdict |
| --- | ---: | ---: | ---: | --- |
| Euclidean n=256 d=4 | 107.45 us | 94.30 us | 1.14x slower | residual near-parity |
| Cosine n=256 d=4 | 114.04 us | 87.20 us | 1.31x slower | residual loss |
| Euclidean n=512 d=4 | 425.75 us | 310.83 us | 1.37x slower | residual loss |
| Cosine n=512 d=4 | 461.16 us | 283.75 us | 1.63x slower | residual loss |

## Correctness And Gates

- PASS: `rch exec -- cargo test -p fsci-spatial pdist --lib -- --nocapture`
  (10 passed).
- PASS: `rch exec -- cargo test -p fsci-spatial --lib -- --nocapture`
  (206 passed, 2 ignored).
- PASS: `rch exec -- cargo check -p fsci-spatial --all-targets`; existing
  warning remains `point_in_circumcircle` dead code.
- PASS: `git diff --check`.
- BLOCKED: `rch exec -- cargo test -p fsci-conformance -- --nocapture` fails
  before this spatial lane in `crates/fsci-conformance/tests/e2e_sparse.rs`
  with `SolveResult` passed where `&[f64]` is expected.
- BLOCKED: `rch exec -- cargo clippy -p fsci-spatial --all-targets -- -D warnings`
  stops in dependency crate `fsci-linalg` on existing lints.
- BLOCKED: `cargo fmt -p fsci-spatial --check` reports pre-existing
  `fsci-spatial` source/bench rustfmt drift outside this patch.

## Decision

Keep this patch because it is a same-worker 5.54-29.51x internal speedup with
targeted and full-crate spatial tests green.

Do not retry generic metric-dispatch removal, dim-4 helper extraction, or dim-4
serial gating for this workload. The next credible lever is a
deeper layout/kernel change: packed SoA or flat point storage, batch multiple
pair outputs per inner loop, and/or generated AVX-style dim-specialized kernels
that remove `Vec<Vec<f64>>` pointer chasing and close the remaining 1.14-1.63x
SciPy loss.
