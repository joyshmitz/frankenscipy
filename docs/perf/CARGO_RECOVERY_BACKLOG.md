# Cargo-recovery backlog (cc / MistyBirch) — prioritized playbook for when disk frees

Authored across the 2026-06-20 DISK-CRITICAL/LOW window (cargo build/bench/check ALL
paused). Run TOP-DOWN when cargo returns. Full reasoning per item in
docs/NEGATIVE_EVIDENCE.md (search the dated entries) and
docs/perf/make_interp_spline_banded_plan.md (Plans 1 & 2).

## P0 — verify the unverified disk-window commits (do FIRST)
9 byte-identical-by-construction commits (sort_unstable sweep + smoothing-spline GCV
band-restricts). Pre-reviewed compile-sound (DISK_WINDOW_VERIFY_QUEUE.md). Run:
`cargo check -p fsci-interpolate -p fsci-stats -p fsci-signal -p fsci-special -p fsci-cluster`
then `cargo test` per crate (interpolate expect 172/0; the 5 zscore/mad/sklearn stats
fails are pre-existing/unrelated). If green, delete the verify-queue doc.

## P1 — high-value, code already authored / well-scoped
1. **factor-once GCV trace** (make_smoothing_spline) — ✅ DONE (2026-06-20, cc): the
   column-independent SPD `lhs` is now factored ONCE via banded Cholesky (chol_banded)
   and the n trace RHS substituted (chol_subst) → O(n²). VERIFIED interpolate 173/0
   (scipy-parity holds; tolerance-parity). NB: the earlier LU-getrs paste-ready code was
   INCORRECT for a physical-row-swap Vec<Vec> (later swaps scatter stored L) — Cholesky
   (no pivoting, SPD lhs) avoids that entirely.
2. **bounded least_squares / curve_fit** — add a `bounds` option + TRF solver (scipy
   default). Common capability gap (non-negative/physical params). Verify vs scipy
   least_squares(method='trf') + curve_fit(bounds=).

## P2 — method/parity gaps (oracle-diff vs scipy, then fix)
3. **solve_bvp** — single shooting → 4th-order COLLOCATION (sparse banded + adaptive
   mesh). Robustness gap (shooting diverges on stiff BVPs). Verify on a stiff BVP.
4. **CloughTocher2D** — local per-vertex gradients → scipy's GLOBAL curvature-min solve.
   Parity gap on curved data (affine matches by construction). Oracle-diff interior pts.
5. **linprog** — leaving-var tie-break is row-order, not strict Bland → harden to
   smallest basis-variable index (degenerate-LP cycling guard). Verify degenerate + large
   LP vs scipy. (HiGHS-parity overall is a known perf wall.)

## P3 — minor capability gaps
6. **minimize methods** — present: BFGS/CG/L-BFGS-B/Nelder-Mead/Newton-CG/Powell/SLSQP/
   TNC (good, incl. constrained SLSQP). Missing vs scipy: COBYLA (derivative-free
   constrained), trust-constr (large-scale constrained). Add if needed; verify SLSQP
   parity while there.

## WALLS (documented, not fixable in safe Rust without a major effort)
- **hilbert / FFT mid-radix** — 2.5x vs pocketfft; the gap is the hand-tuned-C-SIMD FFT
  kernel (radix already Winograd; rfft already native). SIMD-across-r needs a native-SoA
  FFT rewrite (forbid(unsafe) blocks AoS Complex64 SIMD).
- **linprog vs HiGHS**, **RBF residual vs LAPACK**, **ConvexHull/Voronoi vs Qhull** —
  hand-tuned-C walls; fsci is correct + reasonable, just not C-SIMD-fast.

## Confirmed OPTIMAL (do not re-audit)
sort_unstable (72 sites), kendalltau (Knight), weightedtau (Fenwick), distances
(wasserstein/energy O(n log n), pdist SIMD), KDE/mvn/mvt (parallel), MGC (prefix-sum),
rank_max (sort-once), binned-statistic family (accumulate), savgol (coeffs-once),
RectBivariateSpline (compact-band cascade), RGI, make_interp_spline (compact-band),
RBF (flat-solve), DE/global-optimizers (callback lever), procrustes (SVD-align, parity+perf-tested), spatial pdist/KDTree/find_simplex/SphericalVoronoi.
Also confirmed (2026-06-20 audit): Interp1d/CubicSpline (O(n) tridiagonal Thomas
solve, all BCs), minimize methods present (BFGS/CG/L-BFGS-B/Nelder-Mead/Newton-CG/
Powell/SLSQP/TNC). Pchip (Fritsch-Carlson interior + scipy-exact endpoint).
