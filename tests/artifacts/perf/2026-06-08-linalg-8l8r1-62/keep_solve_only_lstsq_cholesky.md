# Keep: solve-only full-rank tall `lstsq` via refined normal-equations Cholesky

Bead: `frankenscipy-8l8r1.62`
Worker fleet: rch (`ovh-*` / `vmi*`)

## Lever

A single solve-only route for the full-rank tall least-squares case, mirroring
the shipped pinv Cholesky route (`frankenscipy-8l8r1.61`, commit `fd34c4f2`).
Before this change, full-rank tall `lstsq` still paid the public thin-SVD
factorization (`routed_lstsq_ms=47.77`) even though the sibling `pinv` call on
the same matrix already routed through the cheap Gram Cholesky path
(`routed_pinv_ms=11.95`).

`lstsq_full_rank_tall_cholesky`:

1. Same shape guard as the pinv Cholesky route: `rows >= 2*cols`,
   `cols >= FULL_RANK_TALL_PINV_MIN_COLS (128)`, all entries finite.
2. Form the Gram `A^T A`, reject on any non-positive / non-finite diagonal.
3. Symmetric eigendecomposition of the Gram supplies the returned
   `singular_values` (σ = √λ, sorted descending) **and** a cheap rcond sanity
   floor.
4. Solve the normal equations `A^T A x = A^T b` with one Cholesky factor, then
   apply **one step of iterative refinement** against the original `A`
   (`r = A^T(b - Ax)`, `dx = (A^T A)^{-1} r`, `x += dx`). This recovers the
   κ(A)² → κ(A) accuracy that forming the Gram would otherwise cost (Björck).
5. Accept only when the refinement correction is below
   `FULL_RANK_TALL_LSTSQ_REFINE_REL_TOL (1e-4)` of the solution norm — the
   convergence certificate. Otherwise return `None` and fall back to the public
   thin-SVD route (fail-closed on conditioning uncertainty).

Wired into `lstsq_with_casp` under `RuntimeMode::Strict`, after the low-rank
tall path and before the public thin-SVD candidate — the exact slot the pinv
Cholesky route occupies in `pinv_with_casp`.

## Isomorphism / behavior parity

- The route accepts a strict subset of full-rank tall matrices (same shape +
  conditioning gate as pinv, plus a refinement-convergence certificate). Any
  uncertainty returns `None`, leaving the prior public thin-SVD route to
  produce the result — so no input changes class without passing the gate.
- Returned `rank = cols` (full rank, asserted by the route probe), `residuals`
  computed directly as `‖b − Ax‖²`, `singular_values = √eig(AᵀA)` descending.
- Public contract unchanged: the `public_svd_lstsq_pinv_golden_payload` test
  uses a 10×5 matrix (`cols=5 < 128`), so it never enters the new route. Golden
  payload SHA-256 is byte-identical:
  `1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225`
  (`public_svd_lstsq_pinv_golden_payload.sha256`).
- Full `fsci-linalg` release suite: 387 passed, 0 failed.
- The 512×256 route probe solution agrees with the reference full-SVD lstsq to
  `lstsq_max_abs_diff = 1.49e-13` — tighter than the prior thin-SVD route's own
  `1.08e-12`, because iterative refinement removes the Gram-squaring error.

## Performance (rch, same probe)

Baseline (`baseline_public_bidiag_svd_route_perf_probe_rch.txt`):

- `routed_lstsq_ms = 47.772198` (public thin-SVD route)
- `reference_lstsq_ms = 85.668199` (full-SVD lstsq solve)

After (`after_solve_only_lstsq_route_perf_probe_rch.txt`):

- `routed_lstsq_ms = 16.749949`
- `reference_lstsq_ms = 96.849138`
- `lstsq_rank = 256`, `lstsq_max_abs_diff = 1.49e-13`

Confirmation runs: `routed_lstsq_ms ∈ {16.44, 20.49, 19.70}`, all with
`max_abs_diff = 1.49e-13`, `rank = 256`.

## Score

- vs prior public thin-SVD route: `47.77 / 16.75 = 2.85×`
- vs full-SVD reference solve: `96.85 / 16.75 = 5.78×`

Both clear Score ≥ 2.0. **Keep.**
