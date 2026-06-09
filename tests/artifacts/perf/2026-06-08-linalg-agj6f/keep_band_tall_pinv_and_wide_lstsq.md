# Keep: extend tall pinv + wide lstsq routes to the 1 < ratio < 2 band

Bead: `frankenscipy-agj6f` (follow-up to `m4avk`, commit `152a73b4`)
Worker fleet: rch

## Lever

`m4avk` relaxed the tall **lstsq** Cholesky guard from `rows >= 2*cols` to
strictly-tall. This commit applies the identical one-line relaxation to the two
remaining rectangular normal-equations routes:

- `pinv_full_rank_tall_cholesky_with_min_cols`: `rows < cols*2` → `rows <= cols`
  (reject only non-tall), `cols >= min_cols` kept.
- `lstsq_min_norm_wide_cholesky`: `cols < rows*2` → `cols <= rows` (reject only
  non-wide), `rows >= FULL_RANK_TALL_PINV_MIN_COLS` kept.

The public thin-SVD candidate is also gated at `rows >= 2*cols`, so the
`1 < ratio < 2` band hit the full SVD before. Each route's existing acceptance
gate — the right-inverse check (`‖pinv·A − I‖ ≤ tol`) for tall pinv, the
iterative-refinement convergence (`‖dx‖/‖x‖ < 1e-4`) for wide lstsq — measures
achieved accuracy and falls back to SVD on uncertainty, so widening the eligible
shape is safe.

## Isomorphism / behavior parity

- `>= 2*cols` (resp. `>= 2*rows`) cases unchanged. Only the band is newly
  routed, verified against the same full-SVD reference.
- tall pinv 320×256: `pinv_max_abs_diff = 8.45e-14`, `rank = 256`.
- wide lstsq 256×320: `lstsq_max_abs_diff = 1.79e-13`, `rank = 256`, routed vs
  reference minimum-norm match to 14 digits
  (`5.2203560162630636` vs `5.2203560162630724`).
- Public golden payload SHA-256 byte-identical:
  `1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225`.
- Full `fsci-linalg` release suite: 388 passed, 0 failed.

## Performance (rch, same-worker A/B vs prior full-SVD route)

- tall pinv 320×256: `reference 81.42 ms → routed 7.97 ms` = **10.21×**
- wide lstsq 256×320: `reference 117.64 ms → routed 29.97 ms` = **3.92×**

## Score

`10.21×` (tall pinv) and `3.92×` (wide lstsq), both clearing Score ≥ 2.0.
**Keep.** The full rectangular full-rank regime (any ratio > 1, min-dim ≥ 128)
is now off the full SVD across all four routes (tall/wide lstsq, tall pinv,
square pinv).
