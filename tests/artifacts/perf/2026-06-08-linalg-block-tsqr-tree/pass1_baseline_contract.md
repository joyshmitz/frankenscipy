# Pass 1 - Baseline And Contract

Bead: `frankenscipy-8l8r1.56`

## Source State

`crates/fsci-linalg/src/lib.rs` had zero diff before this baseline. The public tall SVD route is the restored
Golub-Kahan bidiagonal candidate.

## Remote Baselines

Public route probe on RCH worker `vmi1153651`:

- `reference_lstsq_ms=174.584885`
- `routed_lstsq_ms=131.225285`
- `lstsq_speedup=1.330421`
- `reference_pinv_ms=134.914952`
- `routed_pinv_ms=118.656896`
- `pinv_speedup=1.137017`
- ranks: `256`, `256`
- max diffs: `1.07647224467655178e-12`, `2.28428387316625958e-14`

Reducer probe on RCH worker `vmi1153651`:

- `shape=1024x512`
- `elapsed_ms=606.096750`
- `digest=0x90cdd3f8f71ed2c1`

Criterion baseline on RCH worker `vmi1167313`:

- `lstsq/512x256=[117.34 ms 120.64 ms 124.31 ms]`
- `pinv/512x256=[114.37 ms 116.98 ms 119.83 ms]`

Golden payload on RCH worker `vmi1153651`:

- `public_svd_lstsq_pinv_golden_payload` passed.
- SHA: `1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225`

## Behavior Contract

- Ordering/ties: keep the existing public descending singular-value and adjacent-gap guard.
- Floating point: route only when the final public reconstruction guard accepts; golden SHA must stay unchanged.
- Rank: full-rank only; rank-deficient or clustered spectra fall back.
- Sign: reuse existing `canonicalize_svd_factor_signs`.
- RNG: none.
- Safety: safe Rust only, no C BLAS/LAPACK/MKL/XLA, no unsafe, no thread fanout.
