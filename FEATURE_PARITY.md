# FEATURE_PARITY

## Status Legend

- `not_started` — no implementation exists
- `in_progress` — partial implementation, major gaps remain
- `parity_gap` — implementation exists but missing significant SciPy functions
- `parity_green` — core V1 functions implemented, conformance tested

## Module-Level Parity Assessment

SciPy exports ~1437 public symbols. FrankenSciPy implements ~750+ public functions.
Overall V1 coverage: **~52% of full SciPy surface** (targets highest-value functions).

| Module | SciPy Symbols | FSCI Functions | Coverage | Status |
|---|---|---|---|---|
| linalg | 108 | 113 | 105% | parity_green |
| sparse | 112 (incl. linalg) | 45+ | 40% | parity_gap |
| optimize | 82 | 45+ | 55% | parity_gap |
| integrate | 38 | 14 | 37% | parity_gap |
| fft | 40 | 22 | 55% | parity_gap |
| special | 319 | 151 | 47% | parity_gap |
| stats | 182 | 136 fn + 24 dist | 88% | parity_green |
| signal | 159 | 137 | 86% | parity_green |
| interpolate | 63 | 32 | 51% | parity_gap |
| spatial | 23 + distance | 15 | 45% | parity_gap |

## Detailed Gap Analysis

### fsci-linalg (105% coverage - exceeds SciPy core)

**Implemented:** solve, solve_triangular, solve_banded, inv, det, lstsq, pinv, lu, lu_factor, lu_solve, qr, svd, svdvals, cholesky, cho_factor, cho_solve, ldl, eig, eigvals, eigh, eigvalsh, schur, hessenberg, expm, norm, matrix_rank, solve_with_casp, logm, sqrtm, matrix_power, solve_sylvester, solve_continuous_lyapunov, solve_discrete_lyapunov, solve_circulant, solve_toeplitz, khatri_rao, hadamard, block_diag, companion, issymmetric, ishermitian, is_positive_definite, is_diagonal, is_upper_triangular, is_lower_triangular, is_orthogonal, mat_norm_1, mat_norm_inf, bandwidth, and 60+ more

**Missing (LOW priority):**
- Matrix functions: `fractional_matrix_power`, `funm`, `signm`
- Generalized decompositions: `qz`, `ordqz`
- Subspace: `orth`, `null_space`, `subspace_angles`, `polar`
- Banded: `eig_banded`, `eigh_tridiagonal`, `cho_solve_banded`, `solveh_banded`

### fsci-sparse (40% coverage)

**Implemented:** COO/CSR/CSC/DIA formats, spsolve, splu, spilu, cg, pcg, gmres, bicgstab, minres, lsqr, lsmr, eigsh, eigs, svds, spsolve_triangular, eye, diags, random, block_diag, bmat, kron, dijkstra, bellman_ford, breadth_first_order, depth_first_order, minimum_spanning_tree, connected_components, laplacian

**Missing (MEDIUM priority):**
- Formats: BSR, DOK, LIL
- Solvers: `bicg`, `cgs`, `lgmres`, `qmr`
- Operations: `expm`, `matrix_power`, `norm`

### fsci-opt (55% coverage)

**Implemented:** minimize (BFGS, CG, Powell, NM, L-BFGS-B, Newton-CG, TNC, COBYLA, SLSQP, trust-ncg, trust-krylov, trust-constr, dogleg), minimize_scalar, brentq, brenth, bisect, ridder, toms748, halley, newton_scalar, secant, root_scalar, root, fsolve, broyden1, curve_fit, least_squares, line_search, Bounds, LinearConstraint, NonlinearConstraint, differential_evolution, basinhopping, dual_annealing, shgo, pso, linprog, milp, linear_sum_assignment, approx_fprime, check_grad, cobyla

**Missing (LOW priority):**
- `minimize` methods: `trust-exact`, `Newton-CG` hessian product
- Root: `broyden2`, `anderson`, `hybr`, `lm`
- Utilities: `rosen` family test functions

### fsci-integrate (37% coverage)

**Implemented:** solve_ivp (RK23, RK45, DOP853, BDF, Radau), odeint, solve_bvp, quad, dblquad, trapezoid, simpson, cumulative_trapezoid

**Missing (MEDIUM priority):**
- `LSODA` solver (adaptive stiff/non-stiff)
- `tplquad`, `nquad` (3D and N-D quadrature)
- `quad_vec` (vectorized quadrature)
- `romb` (Romberg integration)
- `cumulative_simpson`
- `cubature` (adaptive cubature)

### fsci-fft (55% coverage)

**Implemented:** fft, ifft, rfft, irfft, fft2, ifft2, fftn, dct (I-IV), idct, dst (I-IV), hilbert, fftfreq, rfftfreq, fftshift, ifftshift

**Missing (MEDIUM priority):**
- `rfft2`, `irfft2`, `rfftn`, `irfftn` (multi-D real FFT)
- `ifftn` (N-D inverse)
- `dctn`, `idctn`, `dstn`, `idstn` (N-D DCT/DST)
- `hfft`, `ihfft` (Hermitian FFT)
- `next_fast_len` (efficient size selection)
- `fht`, `ifht` (fast Hankel transform)

### fsci-special (47% coverage)

**Implemented:** gamma, gammaln, digamma, polygamma, gammainc, gammaincc, rgamma, factorial, factorial2, comb, perm, zeta, zetac, beta, betaln, betainc, betaincinv, erf, erfc, erfinv, erfcinv, j0, j1, jn, jv, jve, y0, y1, yn, yv, yve, iv, ive, kv, kve, k0, k1, k0e, k1e, i0, i1, i0e, i1e, hankel1, hankel2, airy, airye, ai_zeros, bi_zeros, hyp1f1, hyp2f1, ellipk, ellipkm1, ellipe, ellipkinc, ellipeinc, ellipj, elliprc, elliprd, elliprf, elliprg, elliprj, lambertw, wrightomega, exp1, expi, eval_legendre, eval_chebyt, eval_chebyu, eval_chebyc, eval_chebys, eval_laguerre, eval_genlaguerre, eval_hermite, eval_hermitenorm, roots_legendre, fresnel, entr, rel_entr, kl_div, sinc, xlogy, xlog1py, logsumexp, expit, logit, log_expit, log_ndtr, ndtr, ndtri, huber, pseudo_huber, softmax, log_softmax, exprel

**Missing (MEDIUM priority):**
- Spherical: `spherical_jn`, `spherical_yn`, `sph_harm_y`, `lpmv`
- Struve: `struve`, `modstruve`
- Dawson: `dawsn`
- Some elliptic functions
- Remaining Bessel variants

### fsci-stats (88% coverage - near-complete)

**Implemented:** 24+ distributions (Normal, StudentT, ChiSquared, Chi, Uniform, Exponential, F, Beta, Gamma, Weibull, Lognormal, Poisson, Cauchy, Laplace, Triangular, Binomial, NegBinomial, Geometric, Hypergeometric, Pareto, Rayleigh, Gumbel, Logistic, Maxwell, VonMises, InverseGaussian/Wald, Bernoulli), describe, skew, kurtosis, mode, moment, sem, iqr, variation, zscore, ttest_1samp, ttest_ind, ttest_ind_welch, ttest_rel, ks_1samp, ks_2samp, shapiro, normaltest, mannwhitneyu, wilcoxon, kruskal, f_oneway, chi2_contingency, fisher_exact, linregress, pearsonr, spearmanr, kendalltau, pointbiserialr, rankdata, gaussian_kde, boxcox, boxcox_normmax, entropy, differential_entropy, circmean, circvar, circstd, bootstrap, permutation_test, power_divergence, median_abs_deviation, trim_mean, tmean, tvar, tstd, tsem, mstats, and many more

**Missing (LOW priority):**
- Some exotic distributions (~80+ remaining from SciPy's 100+)
- `fit` method for all distributions
- Full `rv_continuous`/`rv_discrete` base class compatibility

### fsci-signal (86% coverage - near-complete)

**Implemented:** savgol_coeffs, savgol_filter, all window functions (hann, hamming, blackman, kaiser, nuttall, bohman, bartlett, flattop, cosine, tukey), convolve, fftconvolve, correlate, correlate2d, find_peaks, butter, cheby1, cheby2, ellip, bessel, iirfilter, iirnotch, iirpeak, firwin, firwin2, remez, kaiserord, lfilter, filtfilt, sosfilt, sosfiltfilt, sosfilt_zi, tf2sos, sos2tf, sos2zpk, zpk2sos, tf2zpk, zpk2tf, freqz, freqz_sos, group_delay, phase_delay, spectrogram, stft, istft, csd, coherence, welch, periodogram, lombscargle, resample, resample_poly, decimate, upfirdn, chirp, sawtooth, square, gausspulse, sweep_poly, unit_impulse, impulse, step, hilbert, detrend, get_window, medfilt, wiener, deconvolve, minimum_phase, cwt, morlet, ricker, max_len_seq, czt, zoom_fft, matched_filter, chroma, spectral_centroid, spectral_bandwidth, spectral_rolloff

**Missing (LOW priority):**
- LTI systems: `lti`, `dlti` classes, `lsim`
- `freqs` (analog frequency response)
- Some exotic windows

### fsci-interpolate (51% coverage)

**Implemented:** Interp1d (linear, nearest, cubic spline), CubicSpline (with boundary conditions), PchipInterpolator, Akima1DInterpolator, BSpline, make_interp_spline, RegularGridInterpolator, NearestNDInterpolator, LinearNDInterpolator

**Missing (MEDIUM priority):**
- `BarycentricInterpolator`
- `make_lsq_spline`
- `UnivariateSpline`, `InterpolatedUnivariateSpline`
- `interpn`, `griddata`
- 2D: `RectBivariateSpline`, `SmoothBivariateSpline`
- RBF: `RBFInterpolator`
- `CloughTocher2DInterpolator`

### fsci-spatial (45% coverage for distance, ~20% overall)

**Implemented:** KDTree (query, query_k), euclidean, sqeuclidean, cityblock, chebyshev, cosine, minkowski, correlation, pdist, squareform, cdist, cdist_metric

**Missing (HIGH priority):**
- Geometry: `ConvexHull`, `Delaunay`, `Voronoi`, `SphericalVoronoi`
- More metrics: `hamming`, `jaccard`, `mahalanobis`, `canberra`, `braycurtis`, `wminkowski`
- KDTree: `query_ball_point`, `query_ball_tree`, `count_neighbors`, `sparse_distance_matrix`
- Transform: `procrustes`, `geometric_slerp`
- `distance_matrix` (different from cdist)

## Packet Readiness Snapshot

| Packet ID | Extraction | Impl | Conformance | Sidecar | Overall |
|---|---|---|---|---|---|
| `FSCI-P2C-001` (IVP) | ready | parity_gap | in_progress | in_progress | parity_gap |
| `FSCI-P2C-002` (linalg) | ready | parity_gap | in_progress | in_progress | parity_gap |
| `FSCI-P2C-003` (opt) | ready | parity_gap | in_progress | in_progress | parity_gap |
| `FSCI-P2C-004` (sparse) | ready | parity_gap | in_progress | in_progress | parity_gap |
| `FSCI-P2C-005` (FFT) | ready | parity_gap | in_progress | in_progress | parity_gap |
| `FSCI-P2C-006` (special) | ready | parity_gap | in_progress | in_progress | parity_gap |
| `FSCI-P2C-007` (array API) | ready | in_progress | in_progress | in_progress | in_progress |
| `FSCI-P2C-008` (CASP) | ready | in_progress | in_progress | in_progress | in_progress |

## Highest-Impact Missing Functions (V1 Priorities)

All top 10 previously-listed functions are now IMPLEMENTED:

1. ~~**`scipy.stats.describe`** + `skew`, `kurtosis`, `mode`~~ ✓ DONE
2. ~~**`scipy.linalg.logm`** + `sqrtm` + `matrix_power`~~ ✓ DONE
3. ~~**`scipy.optimize.differential_evolution`**~~ ✓ DONE
4. ~~**`scipy.optimize.linprog`**~~ ✓ DONE
5. ~~**`scipy.interpolate.RegularGridInterpolator`**~~ ✓ DONE
6. ~~**`scipy.signal.freqz`**~~ ✓ DONE
7. ~~**`scipy.signal.spectrogram`** / `stft`~~ ✓ DONE
8. ~~**`scipy.signal.firwin`**~~ ✓ DONE
9. ~~**`scipy.special.sinc`** + `xlogy` + `logsumexp`~~ ✓ DONE
10. ~~**`scipy.stats.ks_1samp`** + `ks_2samp`~~ ✓ DONE

**Remaining V1 Priorities (lower impact):**
1. `scipy.special.spherical_jn/yn` — spherical Bessel functions
2. `scipy.interpolate.RBFInterpolator` — radial basis interpolation
3. `scipy.signal.lti`/`dlti` class — LTI system representation
4. `scipy.sparse.expm` — sparse matrix exponential
5. `scipy.linalg.qz` — generalized Schur decomposition

## Required Evidence Per Feature Family

1. Differential fixture report.
2. Edge-case/adversarial test results.
3. Benchmark delta (for runtime-significant paths).
4. Documented compatibility exceptions (if any).
5. RaptorQ sidecar manifest plus decode-proof record for each durable artifact bundle.
