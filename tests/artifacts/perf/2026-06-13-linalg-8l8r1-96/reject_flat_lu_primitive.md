# frankenscipy-8l8r1.96 rejection: flat contiguous LU primitive

## Target

Bead: `frankenscipy-8l8r1.96`

Goal: replace the shared safe-Rust LU factorization primitive used by dense LU-family consumers, without changing pivot order, singular fallback, finite-input handling, determinant sign, inverse identity behavior, or public solve residual contracts.

## Baselines

Focused RCH baselines taken before source edits:

- `vmi1149989` `baseline_solve/1000x1000`: `[108.42 ms 112.47 ms 116.64 ms]`
- `vmi1149989` `inv/256x256`: `[8.3082 ms 8.6720 ms 9.1244 ms]`
- `vmi1149989` `det/256x256`: `[1.0401 ms 1.0614 ms 1.0875 ms]`

Because later RCH selection moved to `vmi1227854`, a clean detached worktree at committed `HEAD` was used for a same-worker solve baseline:

- clean `HEAD` on `vmi1227854` `baseline_solve/1000x1000`: `[97.308 ms 98.549 ms 99.886 ms]`

## Candidate Lever

Tried one shared primitive replacement:

- flat row-major `LuFactorsFlat`
- first-max partial pivoting with the same row permutation semantics
- public route thresholds for solve/inverse/determinant
- determinant parity/product route with overflow fallback
- then a second probe of the same primitive with row-chunked parallel trailing update

No `unsafe` and no external BLAS/LAPACK/MKL/XLA were used.

## Behavior Contract

The candidate was not kept. The retained tree has no `crates/fsci-linalg/src/lib.rs` source diff.

Source equivalence artifact:

- `source_equivalence_sha256.txt`

That artifact records the working-tree `crates/fsci-linalg/src/lib.rs` SHA-256, the committed `HEAD` SHA-256, and a clean `git diff --quiet -- crates/fsci-linalg/src/lib.rs` result.

Since no source behavior is retained, ordering/tie-breaking, floating-point operation order, RNG absence, singular fallback, determinant sign handling, and public API behavior remain the committed `HEAD` behavior.

## Results

Same-worker keep gate:

- clean `HEAD` on `vmi1227854` `baseline_solve/1000x1000`: midpoint `98.549 ms`
- flat LU solve route on `vmi1227854`: midpoint `108.90 ms`
- ratio: `0.905x` of baseline, a regression

Additional routing evidence:

- flat LU inverse route on `hz1` `inv/256x256`: `[19.947 ms 20.194 ms 20.544 ms]`, far slower than the focused 8-9 ms class baseline
- flat LU plus parallel trailing update on `vmi1152480` `baseline_solve/1000x1000`: `[174.67 ms 179.14 ms 183.70 ms]`, not promising enough to pursue as a keep candidate
- crate-scoped `cargo check -p fsci-linalg --lib` passed for the parallel-update probe before rejection

## Score

Score: `(Impact 0.0 * Confidence 4.0) / Effort 4.0 = 0.0`

Verdict: reject. Do not repeat flat row-major LU with scalar or row-chunked direct trailing updates.

## Next Route

The next LU-family primitive must replace the trailing update with a panel-packed/register-blocked microkernel or a recursive communication-avoiding LU design, not a scalar flat-buffer rewrite. Required next evidence:

- same-worker baseline before source edits
- pivot tie and singular fallback proof
- solve residual/inverse identity/determinant sign proof
- golden-output digest
- focused same-worker rebench
