# Keep: extend tall lstsq normal-equations route to the 1 < ratio < 2 band

Bead: `frankenscipy-m4avk` (sibling of `8l8r1.62` / `g8u9m` / `hyq08`)
Worker fleet: rch

## Lever

The tall normal-equations Cholesky route (`lstsq_full_rank_tall_cholesky`,
bead `8l8r1.62`) was gated at `rows >= 2*cols`. The public thin-SVD candidate
(`public_bidiag_thin_svd_candidate`) is **also** gated at `rows >= 2*cols`, so a
full-rank tall matrix in the `cols < rows < 2*cols` band (e.g. 320×256, ratio
1.25) hit *neither* fast route and fell to the full `safe_svd(matrix, true,
true)` — the most expensive path (it builds U and the n×n V).

Single-line relaxation: the guard now requires only `rows > cols` (strictly
tall, full column rank) keeping `cols >= FULL_RANK_TALL_PINV_MIN_COLS (128)`.
The existing iterative-refinement convergence gate (`‖dx‖/‖x‖ < 1e-4`) already
guarantees accuracy and falls back to the SVD route on any conditioning
uncertainty, so widening the eligible shape is safe.

## Isomorphism / behavior parity

- The `>= 2*cols` cases are unchanged (still accepted, same code). Only the
  `1 < ratio < 2` band is newly routed; it was previously full-SVD and is now
  verified against that same full-SVD reference to
  `lstsq_max_abs_diff = 1.78e-13` with `rank = cols` (full column rank).
- Public contract unchanged: `public_svd_lstsq_pinv_golden_payload` (10×5,
  cols=5 < 128) never enters the route. Golden SHA-256 byte-identical:
  `1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225`.
- Full `fsci-linalg` release suite: 388 passed, 0 failed.

## Performance (rch, 320×256 probe, same-worker A/B)

The probe's `reference_lstsq_ms` is exactly the pre-change route (full SVD).

- run 1: `reference 168.07 ms → routed 49.33 ms` = **3.41×**
- run 2: `reference  73.45 ms → routed 23.63 ms` = **3.11×**
- `lstsq_rank = 256`, `lstsq_max_abs_diff ≈ 1.8e-13` (stable)

## Score

`≈ 3.1–3.4×` across the previously-full-SVD band. Clears Score ≥ 2.0. **Keep.**
Follow-ups (separate commits): the same `>= 2*cols → > cols` relaxation for the
tall pinv Cholesky route and the wide min-norm route (`cols > rows`).
