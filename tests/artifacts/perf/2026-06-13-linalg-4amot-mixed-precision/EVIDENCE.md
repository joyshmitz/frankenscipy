# frankenscipy-4amot: mixed-precision iterative-refinement LU (solve hot path)

## Lever
Public `solve()` strict/general route at `n >= FLAT_LU_SOLVE_MIN_DIM` (1000) now tries
`lu_solve_mixed_precision` first, falling back to the exact f64 `lu_solve_blocked`:

    let Some(x) = lu_solve_mixed_precision(a, b).or_else(|| lu_solve_blocked(a, b))

Mixed precision = LAPACK `dsgesv` strategy:
1. Factor A **in f32** with a register-blocked blocked-LU (16-wide `Simd<f32,16>` trailing
   update) — the O(n³) factorization at half the bandwidth and double the SIMD width.
2. Initial solve in f32.
3. f64 **iterative refinement**: r = b − A·x (f64), A·d = r via the f32 factors, x += d,
   until the f64 backward error ‖r‖∞ ≤ 8·n·εf64·(‖A‖∞‖x‖∞ + ‖b‖∞).

## Parity safeguard (absolute)
`lu_solve_mixed_precision` returns the refined x ONLY when it reaches the f64-LU
backward-error bar; if the f32 factors are too inaccurate (ill-conditioned) refinement
stalls and it returns None, so the caller solves exactly in f64. No system gets a
worse-than-f64 answer. The f64 LU path (`lu_factor_blocked` / `lu_solve_blocked` /
`inv` / `det`) is untouched.

## Proof bundle
- `cargo test -p fsci-linalg --release --lib -- --include-ignored`: **430 passed, 0 failed,
  0 ignored**. Includes:
  - `flat_lu_golden_digest` (ignored golden) still asserts `0x2fc8ed294ef0427c` — f64 LU
    factors **bit-identical** (golden sha unchanged).
  - new `lu_solve_mixed_precision_matches_f64_and_falls_back`: mixed path matches the f64
    reference to < 1e-9 and residual < 1e-9 at n=130/200/270; near-singular input declines
    (or still matches f64 to tol) — fallback proven.
- Existing parity/reference tests (`lu_solve_blocked_matches_reference`,
  `inv_blocked_matches_reference`) unchanged and green.

## Benchmark — same-worker A/B (vmi1152480, one binary, one run)
`baseline_solve` bench gained a `_f64` arm that flips `DISABLE_MIXED_LU` so both routes are
measured on the SAME worker/binary (kills the ~2x cross-worker variance seen on the fleet):

| arm                          | time (median, [low hi])      |
|------------------------------|------------------------------|
| `baseline_solve/1000x1000`     (mixed) | **66.754 ms** [64.102, 69.431] |
| `baseline_solve/1000x1000_f64` (f64)   | **99.286 ms** [95.497, 103.42] |

**Speedup = 99.286 / 66.754 = 1.49x**, same worker, same conditions.

Corroborating quiet-worker singletons: mixed 61.786 ms (vmi1149989); f64 108.14 ms
(vmi1156319, via `FSCI_DISABLE_MIXED_LU=1`); f64 class 96–104 ms (e45ebe15).

## Score
Impact (1.49x solve speedup, the public hot path) · Confidence (absolute parity, proven
fallback) / Effort (moderate) = clears the keep bar (e45ebe15 shipped a 1.17x keep).

## Next route (per NO-CEILING)
Profiling implies the scalar panel factorization + U-block triangular solve are now the
larger share of the remaining f32 time (f32 only accelerates the SIMD trailing update).
The next algorithmically-different lever is recursive / communication-avoiding LU, which
recasts the panel factorization itself as GEMM-heavy trailing updates — filed separately.
