# Tolerance Policy

Per-packet `(rtol, atol)` policy for FrankenSciPy P2C conformance fixtures. Required by SPEC §4
(*fail-closed*: "compatibility exceptions require explicit allowlist entries and audit traces") and
by the reality-check honesty rule that no packet may carry a `parity_green` claim while its
tolerances remain undocumented or ad-hoc.

This document is the audit trail for every tolerance choice in
`crates/fsci-conformance/fixtures/FSCI-P2C-*.json`. New fixture cases MUST cite a tier or open
an exception under §5.

Source of truth for current values: the fixture JSONs themselves. This document fixes the
**policy** (which tier each operation is allowed to use, and why); the fixtures fix the **values**.
When the two disagree, fix the fixture, not the policy.

---

## 1. Tolerance tiers

Tiers are ordered tightest → loosest. Each tier states the rule, the algorithmic category it is
intended for, and the rationale.

| Tier | (rtol, atol)            | Use for                                                          | Rationale |
|------|-------------------------|------------------------------------------------------------------|-----------|
| T0   | `(0.0, 0.0)`            | Exact integer/categorical results, structural equality           | No floating-point operation; bit-exact agreement is the only correct answer |
| T1   | `(1e-15, 1e-15)`        | Machine-precision identities (FFT roundtrips, fftfreq, constants)| One round of double-precision arithmetic; agreement to ~5 ulp is achievable |
| T2   | `(1e-12, 1e-12)`        | Direct closed-form / O(n) numerics (linalg solve, polyval, mm)   | A handful of double-precision multiplies; agreement to ~5000 ulp |
| T3   | `(1e-10, 1e-10)`        | Iterative numerics with bounded condition number                 | Multi-step algorithms (Jacobi, eigh on well-conditioned matrices, quadrature on smooth integrands) |
| T4   | `(1e-08, 1e-08)`        | Series with truncation, conditional accuracy paths               | Special-function series, ODE integrators with default rtol, root-finding |
| T5   | `(1e-06, 1e-06)`        | Library-internal heuristics, bessel arguments near branch cuts   | Cases where SciPy itself documents non-determinism / heuristic switching |
| T6   | `(1e-04, 1e-04)` — `(1e-01, 1e-01)` | Stochastic / globally-converging optimizers, IIR design  | Outputs depend on RNG seeds, basin selection, or equiripple iteration count |
| Txn  | non-uniform `(rtol, atol)` per element | Vector-valued tolerances (P2C-001 validate_tol)        | Element-specific scale, e.g. log-magnitudes mixed with linear |
| Tnone| `null` / unspecified    | Comparison performed by structural equality (None as sentinel)   | Case is a label/shape/index check; no floating tolerance applies |

**Loosening above T4 always requires a §5 exception.**

---

## 2. Per-packet policy

For each packet the *baseline* column is the tier that the largest cluster of cases SHOULD use.
Operations listed under "Approved exceptions" are the only ones permitted to relax beyond
baseline, and the justification column states why.

### FSCI-P2C-001 validate_tol — 4 cases

The tolerance-validator's own self-test packet. Cases use vector-valued `rtol`/`atol` to exercise
the per-element comparison path. Tier: **Txn (vector tolerances)** by design. No baseline.

### FSCI-P2C-002 linalg_core — 63 cases

| Tier | Operations | Notes |
|------|------------|-------|
| **Baseline T2 (1e-12)** | solve, solve_triangular, det, inv, qr, cholesky, eig, eigh, svd, pinv | Direct LAPACK-equivalent factorization |
| Approved T3 (1e-10) | inv (1 case), pinv (3 cases), solve_banded (1 case) | Higher condition number test matrices; loosened to absorb back-substitution amplification |
| Approved T4 (1e-08) | solve (2 cases), solve_banded (1 case) | Banded structure with near-singular bands |
| Approved T4 (1e-09) | lstsq (4 cases) | Least-squares uses pseudoinverse path; SVD truncation amplifies error |
| Approved T5 (1e-06) | lstsq (1 case) | Rank-deficient case where SciPy's truncation rank may differ |
| **Exception** | pinv `atol=-1.0` (1 case) | Sentinel meaning "scipy default rcond"; consumed by harness, not arithmetic |

Justification: linalg routines map directly to BLAS/LAPACK primitives; agreement to T2 is the
norm. T4/T5 relaxations cluster around rank-deficient and ill-conditioned cases where the rank
threshold itself differs between implementations, not the arithmetic.

### FSCI-P2C-003 optimize_core — 53 cases

| Tier | Operations | Notes |
|------|------------|-------|
| Baseline T6 / case-by-case | All optimizers | No optimizer in this packet is bit-deterministic against SciPy without seed control |
| T2 (1e-12) | brute (3 cases) | Grid search returns deterministic argmin on a fixed grid |
| T4 (1e-08) | root (11 cases) | Newton/hybr default xtol; converges to 1e-8 |
| T6 (1e-04) | minimize (3 cases) | Default gradient-based xtol |
| T6 (1e-03 / 1e-02) | minimize (3), basinhopping (2) | Stochastic restarts, tolerance set to expected basin separation |
| T6 (1e-01 / 2.5e-01) | differential_evolution (6 cases) | Population-based, RNG-seeded |
| **Exception** | dual_annealing (1 case, T5 1e-06) | Test forces a deterministic short run; documented in case `case_id` suffix |

Justification: optimizers are the principal source of legitimate tolerance loosening. SPEC §4
allows it provided each loosened case carries either a fixed seed in args or a metamorphic
relation (e.g. "global min within K basins of true min").

### FSCI-P2C-004 sparse_ops — 36 cases

| Tier | Operations | Notes |
|------|------------|-------|
| **Baseline T3 (1e-10) / atol T2 (1e-12)** | add, scale, spmv, spsolve | Direct sparse linear ops; mixed tier rtol/atol pair lets relative match dominate while absolute floor catches near-zeros |
| Approved T5 (1e-05) | eigsh (3), svds (2) | ARPACK iterative — convergence sensitive to restart vectors |
| **Exception** | eigs (1 case, T6 0.1/0.3) | Non-symmetric Lanczos; tolerance documented as "indicative ordering only" |

### FSCI-P2C-005 fft_core — 93 cases

| Tier | Operations | Notes |
|------|------------|-------|
| **Baseline T2 rtol / T1 atol** | fft, fft2, fftn, ifft, ifft2, ifftn, rfft*, irfft*, hfft, ihfft | Standard pocketfft path; T1 atol catches the magnitude-zero bins |
| T1 (1e-15) | fftfreq, rfftfreq, fftshift, ifftshift | Pure index arithmetic; bit-equal up to rounding |
| T4 / T1 atol | dct, dst, irfft (some) | Type-II/III DCT uses double-length FFT internally; small accumulated phase error |
| **Exception** | `op == "br"` (26 cases) | Bit-reversal permutation; baseline rtol=1e-09 documented as "stride accumulation in radix-2 stage" |

### FSCI-P2C-006 special_core — 362 cases (largest packet)

This packet has the most diverse tolerance distribution; every cluster is tracked.

| Tier | Operations | Notes |
|------|------------|-------|
| T0 (exact) | comb, perm, factorial, factorial2 | Integer combinatorics |
| T0 / 1e-300 atol | rgamma (6), ndtr (1), erfcx, ellipkinc near-K boundary, fdtridfd, nrdtrisd, expm1 | Exact-zero or absolute-floor only |
| **T2 (1e-12)** | beta, betaln, boxcox*, cosdg, sindg, tandg, eval_*, log1p, logit, logaddexp*, multigammaln, owens_t, polygamma, ndtr, kolmogorov, sinc, softplus, sph_harm_y, struve_relatives, wright_bessel, xlog*, eval_*, roots_chebyt/u, roots_hermite, roots_laguerre, roots_legendre, ellipe, ellipk, ellipkm1, expit, exp10, exp2 | Standard special functions |
| T3 (1e-10) | digamma (some), ellipeinc, ellipj_*, ellipkinc, kei, ker, kl_div, kolmogi, kv, lambertw, lpmv, modstruve, rel_entr, ndtr, polygamma, shichi, sici, struve | Series with adaptive truncation |
| T4 (1e-08) | dawsn, digamma, exp1, expi, expn, hyp0f1, hyp1f1, hyp2f1, kvp, log_ndtr, hurwitz_zeta, j0, polygamma, spence, spherical_*, zeta, zetac | Series convergence dependent on argument; documented divergence in §5 below |
| T5 (1e-06) | iv (2), kv (2), erfinv, erfcinv, ndtri (1), spherical_jn, spherical_yn | Bessel near argument-order ratio extremes |
| T5 / 5e-08 | fresnel_c, fresnel_s | Cornu-spiral oscillation tail |
| T6 (1e-04) | jn (5), yn (2) | Integer-order Bessel via recurrence; SciPy switches branch around `n ≈ x` |
| T6 (5e-04) | j0, j1, jvp, yvp, jn near zero | Documented in scipy's own test suite as low-precision branch |
| T6 (8e-03) | y0 | Series/asymptotic crossover at argument ~12; tracked by frankenscipy-rd28n family |
| **Exception (must NOT regress)** | hyp1f1, hyp2f1 series silent-divergence | Currently T4; flagged by frankenscipy-rd28n / frankenscipy-9u8kz. Tightening blocked on CASP branch selection (frankenscipy-1i92b) |

Special-function tolerances are the largest source of legitimate looseness. Every relaxation
above T4 has a corresponding `rel_*` metamorphic relation (e.g. `rel_gamma_recurrence`,
`rel_jn_recurrence`, `rel_erf_erfc_identity`) that compensates by checking algebraic
identities at tighter tolerance.

### FSCI-P2C-007 arrayapi_core — 34 cases

All cases are structural / shape / dtype assertions. Tier: **Tnone** by construction.

### FSCI-P2C-008 runtime_casp — 15 cases

CASP calibrator/policy/solver decision paths. Tier: **Tnone** (tests select correct branch, not
a numeric value).

### FSCI-P2C-009 cluster_core — 44 cases

| Tier | Operations | Notes |
|------|------------|-------|
| **Baseline T3 (1e-10)** | linkage (most), inconsistent, vq, whiten, cophenet, silhouette_score, adjusted_rand_score | Hierarchical merging is order-stable when ties are broken deterministically |
| Approved T4 (1e-09) | linkage (14 cases) | Single/complete linkage with `method=ward` accumulates merge-distance error |
| T2 (1e-12) | num_obs_linkage | Exact integer count |
| Tnone | dbscan, fcluster, kmeans, mean_shift, leaves_list, is_*, is_valid_linkage | Cluster labels: structural equivalence (relabel-invariant) |

### FSCI-P2C-010 spatial_core — 57 cases

| Tier | Operations | Notes |
|------|------------|-------|
| **Baseline T3 rtol / T2 atol** | pdist, cdist, all named distance metrics, kdtree_query_*, mahalanobis, voronoi vertex | Distance accumulation O(d) deep; T2 atol catches near-zero distances |
| T0 | kdtree_count_neighbors, kdtree_query_pairs, kdtree_query_ball_tree | Integer counts / index sets |
| T3 (both) | chebyshev, cityblock, convex_hull, correlation, cosine, directed_hausdorff, euclidean, geometric_slerp, kdtree_query, halfspace_intersection, squareform_to_matrix | Closed-form distances |
| T5 (1e-06) | procrustes | Orthogonal Procrustes solves SVD; loosen for sign ambiguity |
| Tnone | halfspace_intersection (2 unbounded cases) | Existence-only check |

### FSCI-P2C-011 signal_core — 77 cases

| Tier | Operations | Notes |
|------|------------|-------|
| **T2 / T1 atol** | bartlett, blackman, bohman, boxcar, cosine, exponential, gaussian, general_cosine, general_hamming, hann, hamming, lanczos, parzen, tukey, hilbert, convolve, correlate, detrend | Window functions are deterministic; convolve/correlate are direct sum |
| T3 / T1 atol | barthann, blackmanharris, dpss, firwin, flattop, freqz, nuttall | Multi-term window or DFT-evaluated response |
| T4 (1e-08) | kaiser, kaiser_bessel_derived, taylor, savgol_filter | Bessel-internal in window weight |
| T5 (1e-05 / 1e-06) | firls, minimum_phase | Equiripple iteration count not pinned |
| **T6 (1e-04)** | bessel, butter, cheby1, cheby2, ellip | IIR analog→digital pole mapping; SciPy's bilinear pre-warp differs by ~5 ulp at coefficient level → ~1e-4 at filter response |
| T6 (5e-02 / 6e-02 atol) | firwin2, remez | Equiripple Parks-McClellan; relax to amplitude-error envelope |
| T6 (1e-07 / 1e-08) | chebwin | Chebyshev attenuation parameter sensitivity |

The IIR filter T6 floor is the single largest signal-domain relaxation. It is gated on
frankenscipy-1i92b (CASP for branch selection) and frankenscipy-b6z3m (special parity), both
of which are prerequisites for tightening below T4.

### FSCI-P2C-012 stats_core — 71 cases

| Tier | Operations | Notes |
|------|------------|-------|
| **Baseline T3 (1e-10)** | bootstrap_mean, describe, entropy, iqr, kurtosis, linregress, mannwhitneyu (asymptotic), moment, pearsonr, sem, skew, ttest_1samp, ttest_ind, variation, zscore, ks_2samp, spearmanr | Closed-form moment / order statistics |
| T2 (1e-12) | distribution_pdf, distribution_cdf, distribution_ppf (subset) | Direct evaluation of named-distribution density |
| T3 / T2 atol | distribution_pdf/cdf (most cases), ttest_ind (1 case) | Mixed-scale (tail values can underflow) |
| T4 / 1e-09 | distribution_cdf/pdf in tails, kendalltau, ttest_ind (1 case), mannwhitneyu | Series tail / continuity correction |
| T5 (1e-06) | distribution_ppf (2), shapiro | Inverse CDF root-find / order-statistic table |
| T5 (1e-06 / 1e-09) | kendalltau (2 cases) | Tie-correction term |

### FSCI-P2C-013 integrate_core — 54 cases

| Tier | Operations | Notes |
|------|------------|-------|
| **T2 (1e-12)** | trapezoid (smooth), fixed_quad, gauss_legendre, newton_cotes, simpson (smooth), cumulative_trapezoid (smooth) | Closed-form polynomial-exact rules |
| T3 (1e-10) | quad, quad_vec, dblquad, tplquad, simpson, cumulative_simpson, trapezoid (general), newton_cotes (general) | Adaptive Clenshaw-Curtis / QUADPACK epsabs |
| T4 (1e-08) | cubature (default), romb, fixed_quad (1 high-order case), simpson (1 case) | Higher-dim cubature truncation |
| T5 (1e-06) | odeint, solve_ivp (default rtol) | RK45 default tolerance |
| T6 (1e-04) | solve_ivp (3 cases), qmc_quad (4 cases) | Long-time integration / Sobol convergence rate |

### FSCI-P2C-014 interpolate_core — 24 cases

| Tier | Operations | Notes |
|------|------------|-------|
| **T2 (1e-12)** | bspline, cubic_spline (smooth), regular_grid_interpolator (smooth), interp1d (linear) | Closed-form piecewise polynomial |
| T3 (1e-09 / 1e-10) | cubic_spline (most cases) | Tridiagonal solve for natural BC accumulates ~5 ulp |
| T0 | interp1d (categorical/index cases), regular_grid_interpolator (1 case) | Nearest-neighbor / index lookup |

### FSCI-P2C-015 ndimage_core — 6 cases

| Tier | Operations | Notes |
|------|------------|-------|
| T0 | binary_dilation, binary_erosion | Boolean operations |
| T2 (1e-12) | distance_transform_edt, gaussian_filter (1 case) | Direct convolution |
| Tnone | gaussian_filter (1 case), label | Structural shape / connectivity |

This packet is small (only 6 cases) and is the explicit subject of the M3 milestone; expect
expansion under frankenscipy-4703g.

### FSCI-P2C-016 constants_core — 72 cases

| Tier | Operations | Notes |
|------|------------|-------|
| **T1 rtol / T0 atol (1e-15, 0.0)** | constant_value (51), deg2rad, ev_to_joules, joules_to_ev, kg_to_lb, rad2deg | Exact CODATA constants; rtol absorbs the unit-conversion multiply |
| T0 (0.0, 0.0) | constant_value (8 cases — bit-equal CODATA), lb_to_kg | Pinned by table identity |
| T2 (1e-12) | freq_to_wavelength, wavelength_to_freq, convert_temperature | One reciprocal / linear conversion |
| Tnone | constant_value (1 sentinel), convert_temperature (1 sentinel) | Existence/units check |

The constants packet has the cleanest tier structure: T0+T1 covers everything because no
arithmetic happens beyond a single multiply.

### FSCI-P2C-017 io_core — 13 cases

| Tier | Operations | Notes |
|------|------------|-------|
| **T2 (1e-12)** | loadmat, loadtxt, mmread, mmwrite, savemat, savetxt, wav_write | Roundtrip serialization; T2 absorbs the f64 ↔ ASCII conversion error |
| Tnone | loadmat (1), loadtxt (1) | Schema/shape check |

---

## 3. Tier selection algorithm

When adding a new fixture case, pick the tier as follows:

1. **Is the result a count, label, structural object, or boolean?** → T0 (or Tnone if comparison
   is structural).
2. **Is the computation a single round of double-precision arithmetic against a CODATA constant
   or an FFT permutation?** → T1.
3. **Is the computation a closed-form O(n) numeric (linalg solve, polyval, distance, window)?** →
   T2.
4. **Is the computation iterative but bounded (sparse spmv, quadrature on smooth, ndimage
   convolution)?** → T3.
5. **Does the computation contain a series whose truncation is governed by argument magnitude
   (special functions, ODE)?** → T4.
6. **Does the comparison cross a SciPy-internal heuristic switch (Bessel near branch cut, IIR
   coefficient mapping, root-finder near multiple roots)?** → T5.
7. **Is the algorithm stochastic (RNG-seeded), or does its convergence depend on iteration
   count we don't pin (DE, equiripple, Sobol)?** → T6 with mandatory `rationale` field in the
   case JSON.

Anything looser than T6 must open a §5 exception.

---

## 4. Audit trail

For every case where the chosen tolerance differs from this document's per-packet baseline:

- The `case_id` SHOULD encode a hint (e.g. `solve_singular_…`, `bessel_branch_cut_…`).
- The fixture entry MUST carry a per-case `rationale` field referencing either:
  - the bead that owns the deviation (e.g. `frankenscipy-rd28n`), or
  - the SciPy upstream test/reference that documents the algorithmic difference, or
  - a metamorphic relation (`rel_*` case_id) that re-asserts the property at a tighter tier.

Cases that lack any of those three references are linted by the conformance harness and surfaced
in `parity_report.json` under `tolerance_audit_misses`.

---

## 5. Exception process

Loosening above T4 (or above the per-packet baseline column in §2) requires:

1. Open a `bug` or `feature` bead in beads with the title prefix `[tolerance-exception]`.
2. The bead body must state: (a) the case_id, (b) the tighter tier we'd want, (c) the reason
   we cannot achieve it today, (d) the prerequisite work that would let us tighten (link to
   parent bead).
3. Add the case to the fixture with a `rationale` field whose value is the bead id.
4. The harness emits a warning at run-time for every active exception; CI greps for the count
   of `[tolerance-exception]` beads to ensure the number monotonically decreases.

Active exceptions at policy creation:

| case_id pattern | Tier | Bead | Reason |
|-----------------|------|------|--------|
| `hyp1f1_*` series | T4 | frankenscipy-9u8kz | Series silent-divergence for large `|z|`; needs CASP branch selection |
| `hyp2f1_*` series | T4 | frankenscipy-rd28n | Series silent-divergence for `|z| > 1`; needs CASP branch selection |
| `bessel_iir_*`, `butter_*`, `cheby[12]_*`, `ellip_*` | T6 1e-04 | frankenscipy-1i92b | IIR pole mapping divergence; needs CASP for fsci-special branch |
| `firwin2_*`, `remez_*` | T6 5e-02 atol | frankenscipy-b6z3m | Equiripple iteration count not pinned |
| `y0_arg_12_*` | T6 8e-03 | frankenscipy-1i92b | Series→asymptotic crossover at arg ~12 |
| `pinv_atol_neg_one` | T2 (atol=-1.0 sentinel) | none | Sentinel consumed by harness, not arithmetic |

---

## 6. Honesty notes

- **No packet is currently `parity_green` under this policy** if the policy is enforced strictly.
  P2C-001/-007/-008 are structural and pass trivially. P2C-002, -005, -016, -017 are within one
  generation of green pending T2/T3 baseline review. The remainder all have at least one case
  riding on a §5 exception.
- **The harness today does NOT reject undocumented exceptions.** That gate is bead
  frankenscipy-e9z9y (M4 hardening). Until it lands, "documented exceptions" are an honor
  system.
- **Vector-valued tolerances** (Txn) are intentionally not promoted to first-class tier — they
  exist solely to exercise the validator's per-element path in P2C-001, and they should not
  appear in production fixtures because they make the tier system non-uniform across cases.
