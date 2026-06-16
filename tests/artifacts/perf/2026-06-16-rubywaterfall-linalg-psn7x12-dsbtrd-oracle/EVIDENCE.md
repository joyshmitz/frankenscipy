# frankenscipy-psn7x.12 Evidence - Lower-Band DSBTRD Oracle Harness

## Target

- Bead: `frankenscipy-psn7x.12`
- Crate: `fsci-linalg`
- Profile-backed hotspot: `eig_banded(..., lower=true, eigvals_only=false)` still routes through dense/native eigenvectors; prior public DSBTRD wiring regressed because it paid reduction overhead and fell back.
- Environment: local cargo + hyperfine with `RCH_REQUIRE_REMOTE=0`; `ts1` remote RCH worker was offline.

## Baseline

Current-head public route probe, local release build:

| shape | bandwidth | candidate_ms | max abs diff | residual | values digest | vectors digest |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 128x128 | 32 | `99.173048 ms` | `7.56017470848746598e-12` | `1.64845914696343243e-12` | `0xd6dbb9200f65bd92` | `0x6cf3573b5b50c275` |
| 256x256 | 32 | `477.192227 ms` | `4.16093826061114669e-11` | `7.73070496506989002e-12` | `0x09ed4d367faab431` | `0xc32797c0d224a75a` |

Existing scalar-vs-compact full-to-band probe, local release build:

| shape | bandwidth | scalar replay | compact-WY replay | speedup | scalar digest | compact digest |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 256x256 | 32 | `72.530086 ms` | `82.179659 ms` | `0.882580x` | `0x312cbae6f85aa643` | `0xb609dd1775cf3068` |
| 512x512 | 32 | `209.539191 ms` | `124.128144 ms` | `1.688088x` | `0x646bb3412b4b5360` | `0x5d108f82fded2c63` |

## Lever

One source lever was kept: add an isolated lower-band-to-tridiagonal oracle harness that compares:

- Scalar dense similarity oracle: deterministic full-to-band reduction with bandwidth `1`.
- Candidate DSBTRD-class kernel harness: deterministic lower-band fixture expanded into the harness, then safe-Rust trailing symmetric rank-2 Householder similarity (`A := A - v*w^T - w*v^T`) to produce tridiagonal `D/E` and accumulated `Q`.

This commit does not change public `eig_banded`, thresholds, sorting, fallback selection, RNG behavior, or any user-visible numerical route.

## Proof

Command:

```text
RCH_REQUIRE_REMOTE=0 CARGO_TARGET_DIR=/data/projects/.scratch/frankenscipy-rubywaterfall-opt-20260616-1955/.local-target cargo test -j 1 -p fsci-linalg --lib lower_band_tridiagonal_rank2_matches_scalar_oracle --release --locked -- --nocapture
```

Proof output:

```text
lower_band_rank2_oracle n=18 bandwidth=4 tri_drift=8.52651282912120223e-14 scalar_digest=0x4f36279a9fc7fb58 rank2_digest=0xf5a344e20bf95982
lower_band_rank2_oracle n=37 bandwidth=8 tri_drift=9.01494157101723204e-13 scalar_digest=0xd552cac930030969 rank2_digest=0x2c0ca6db29ec6061
lower_band_rank2_oracle n=64 bandwidth=12 tri_drift=2.35900188272353262e-12 scalar_digest=0x9cf661458b0e182c rank2_digest=0x971933511ccecbf9
```

Golden output artifact:

```text
cc4151e32c4ecd968a04692380e2b118dd404b0f0b27cd22a41488d8ed819820  tests/artifacts/perf/2026-06-16-rubywaterfall-linalg-psn7x12-dsbtrd-oracle/lower-band-rank2-golden-output.txt
```

Proof obligations:

- `Q^T A Q = T` reconstruction passed for scalar oracle and rank-2 candidate.
- `Q` orthogonality passed for scalar oracle and rank-2 candidate.
- D/E extraction is bit-identical to the emitted tridiagonal matrix.
- Candidate tridiagonal drift versus scalar oracle stayed within tolerance.
- Ordering/tie-breaking unchanged: no public eigenvalue/eigenvector ordering path changed.
- Floating point: public route unchanged; the new oracle uses tolerance proof because rank-2 similarity changes operation association relative to two full Householder passes.
- RNG unchanged: deterministic fixtures only.
- Safety unchanged: safe Rust only; no C BLAS/LAPACK/MKL/XLA linkage.

## Rebench

Command:

```text
RCH_REQUIRE_REMOTE=0 CARGO_TARGET_DIR=/data/projects/.scratch/frankenscipy-rubywaterfall-opt-20260616-1955/.local-target hyperfine --warmup 1 --runs 5 --show-output 'cargo test -j 1 -p fsci-linalg --lib lower_band_tridiagonal_rank2_oracle_perf_probe --release --locked -- --ignored --nocapture'
```

Hyperfine wall time: `621.0 ms +/- 77.1 ms` over 5 runs.

Internal probe ranges:

| shape | bandwidth | scalar oracle ms | rank-2 oracle ms | speedup range | tri drift | scalar digest | rank-2 digest |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 128x128 | 32 | `3.650413`-`4.235000` | `1.809587`-`2.319893` | `1.732119x`-`2.017263x` | `7.73070496506989002e-12` | `0x100182e84dd71b4e` | `0xf5633055cfd5cc82` |
| 256x256 | 32 | `22.655886`-`24.770361` | `14.590632`-`17.321955` | `1.384673x`-`1.637072x` | `2.46700437855906785e-11` | `0x0eeab02d1f6b78e7` | `0xe8a1c6dd613f20bf` |
| 512x512 | 32 | `177.444358`-`188.801557` | `130.346534`-`148.291297` | `1.273180x`-`1.427941x` | `8.82209860719740391e-11` | `0x7b854cd9c7869930` | `0x3b2d3bec85208397` |

## Score

- Impact: `2.0` (isolated DSBTRD-class oracle kernel is faster at every measured size and establishes the missing proof gate before public wiring).
- Confidence: `4.0` (D/E/Q proof, scalar-oracle comparison, golden output, deterministic digests, repeated local hyperfine).
- Effort: `1.0`.
- Score: `8.0`.

Verdict: KEEP.

## Next Profile Route

Re-profile after this proof harness. The next deeper primitive should move from dense-harness rank-2 proof to a compact lower-band/envelope DSBTRD kernel with explicit bulge frontier and the same `Q^T A Q = T` golden contract before any public `eig_banded` routing.
