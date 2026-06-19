# frankenscipy-nm8ex.1: spatial pdist dim-4 flat row staging

Agent: cod-a / MistyBirch
Date: 2026-06-19
Crate: `fsci-spatial`

## Lever

The previous dim-4 `pdist` fast path still loaded every pair through
`Vec<Vec<f64>>` rows. This patch stages validated 4-column inputs into compact
`[f64; 4]` points once per call, then computes Euclidean and Cosine pairs over
that fixed-width row layout.

Arithmetic order is unchanged and no unsafe code was added. This commit does
not mix in output batching or SIMD so the measured delta is attributable to the
layout change.

## Commands

Rust baseline and candidate:

```bash
RCH_WORKER=ovh-b \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a \
  rch exec -- cargo bench -p fsci-spatial --bench spatial_bench -- pdist --noplot
```

SciPy oracle:

```bash
python3 docs/perf_oracle_pdist.py
```

Correctness guard:

```bash
RCH_WORKER=ovh-b \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a \
  rch exec -- cargo test -p fsci-spatial pdist_dim4_fast_paths_match_metric_helpers -- --nocapture
```

## Same-Worker Internal A/B

Both Rust runs used rch worker `ovh-b`.

| Workload | Baseline median | Candidate median | Speedup | Verdict |
| --- | ---: | ---: | ---: | --- |
| `pdist/euclidean/256` | 263.00 us | 172.83 us | 1.52x | keep |
| `pdist/cosine/256` | 381.98 us | 208.89 us | 1.83x | keep |
| `pdist/euclidean/512` | 794.72 us | 714.58 us | 1.11x | keep |
| `pdist/cosine/512` | 1.1930 ms | 828.70 us | 1.44x | keep |

Criterion reported statistically significant improvements for all four rows.

## Head-To-Head Vs SciPy

Local oracle versions: SciPy 1.17.1, NumPy 2.4.3.

| Workload | Rust candidate median | SciPy p50 | Rust/SciPy time | Verdict |
| --- | ---: | ---: | ---: | --- |
| Euclidean n=256 d=4 | 172.83 us | 88.96 us | 1.94x slower | loss |
| Cosine n=256 d=4 | 208.89 us | 79.69 us | 2.62x slower | loss |
| Euclidean n=512 d=4 | 714.58 us | 309.79 us | 2.31x slower | loss |
| Cosine n=512 d=4 | 828.70 us | 275.14 us | 3.01x slower | loss |

Win/loss/neutral vs SciPy: 0 / 4 / 0.
Internal A/B win/loss/neutral: 4 / 0 / 0.

## Correctness And Gates

- PASS: `pdist_dim4_fast_paths_match_metric_helpers` via rch on `ovh-b`
  (1 passed).
- PASS: `cargo test -p fsci-spatial --lib -- --nocapture` via `rch`
  (206 passed, 2 ignored).
- PASS: `cargo check -p fsci-spatial --all-targets` via `rch`.
- PASS: `cargo clippy -p fsci-spatial --all-targets --no-deps -- -D warnings`
  via `rch` after clearing same-file pre-existing lint blockers.
- PASS: `git diff --check -- crates/fsci-spatial/src/lib.rs`.
- PASS: `git diff --check`.
- PASS: `ubs` on the changed file set exited 0; no critical issues reported.
- BLOCKED: `cargo fmt --check -p fsci-spatial` still reports pre-existing
  unrelated rustfmt drift in `fsci-spatial` bench/source sections outside this
  patch.

## Decision

Keep. The flat row staging is a measured 1.11-1.83x same-worker internal win
with bit-exact dim-4 output preserved.

Do not retry fixed-width row staging alone. The next residual-loss route needs
deeper inner-kernel work: batch multiple condensed outputs per loop, generate
dim-specialized SIMD-style kernels, or move the API internals toward packed
SoA/flat buffers so this copy is not needed.
