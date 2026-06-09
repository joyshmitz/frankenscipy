# Keep: solve-only minimum-norm wide `lstsq` via refined `A Aᵀ` Cholesky

Bead: `frankenscipy-g8u9m` (sibling of `frankenscipy-8l8r1.62`, commit `23a92991`)
Worker fleet: rch

## Lever

Underdetermined / wide full-row-rank `lstsq` (`rows < cols`) fell through to
`safe_svd(matrix, true, true)` — a **full** SVD that materializes the `n×n`
right-singular-vector matrix `V`, which is huge when `cols` is large. No fast
route existed (`lstsq_low_rank_tall` and `lstsq_full_rank_tall_cholesky` both
require `rows >= 2*cols`).

`lstsq_min_norm_wide_cholesky` — the wide mirror of the tall Cholesky route:

1. Guard `cols >= 2*rows`, `rows >= FULL_RANK_TALL_PINV_MIN_COLS (128)`, finite.
2. Form the **small** `rows×rows` Gram `A Aᵀ`; reject on non-positive diagonal.
3. `symmetric_eigen(A Aᵀ)` gives `singular_values = √λ` (descending) and a cheap
   rcond sanity floor.
4. Minimum-norm solve: `(A Aᵀ) y = b` via one Cholesky factor, `x = Aᵀ y`
   (keeping `x` in the row space preserves minimum norm), then **one iterative
   refinement step** against the original `A`:
   `r = b − A x`, `dy = (A Aᵀ)⁻¹ r`, `dx = Aᵀ dy`, `x += dx` — recovers the
   κ(A)² → κ(A) accuracy the Gram squaring would cost (Björck).
5. Accept only when `‖dx‖/‖x‖ < FULL_RANK_TALL_LSTSQ_REFINE_REL_TOL (1e-4)`;
   otherwise `None` → fall back to the public full-SVD route (fail-closed).

Wired into `lstsq_with_casp` under `RuntimeMode::Strict`, immediately after the
tall Cholesky route and before the public thin-SVD candidate.

## Isomorphism / behavior parity

- Returns `rank = rows` (full row rank, asserted), `residuals` empty (scipy
  returns an empty residues array for `m <= n`), `singular_values = √eig(A Aᵀ)`
  descending. The minimum-norm property is verified directly in the probe:
  `routed_solution_norm` vs `reference_solution_norm` agree to ~13 digits
  (`5.21900052307169e0` vs `5.21900052307146e0`).
- Public contract unchanged: `public_svd_lstsq_pinv_golden_payload` uses a 10×5
  (tall) matrix, never enters the wide route. Golden SHA-256 byte-identical:
  `1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225`.
- Full `fsci-linalg` release suite: 387 passed, 0 failed.
- Routed solution agrees with the reference full-SVD min-norm solution to
  `lstsq_max_abs_diff = 3.39e-13`.

## Performance (rch, 256×512 probe, same-worker A/B)

The probe's `reference_lstsq_ms` is exactly the pre-change route (full SVD).

- run 1: `reference 162.64 ms → routed 15.42 ms` = **10.55×**
- run 2: `reference 175.89 ms → routed 19.16 ms` = **9.18×**
- `lstsq_rank = 256`, `lstsq_max_abs_diff = 3.39e-13` (stable)

## Score

`≈ 9–10.5×` vs the prior full-SVD wide route. Clears Score ≥ 2.0. **Keep.**
The wide gap is larger than the tall (5.78×) because the full SVD builds the
`n×n` V matrix that the `A Aᵀ` route never forms.
