# FEATURE_PARITY

## Status Legend

- `not_started` — no implementation exists
- `in_progress` — partial implementation, major gaps remain
- `parity_gap` — implementation exists but missing significant SciPy functions
- `parity_green` — core V1 functions implemented, conformance tested

## Module-Level Parity Assessment

SciPy exports ~1437 public symbols. FrankenSciPy implements ~200 public functions.
Overall V1 coverage: **~14% of full SciPy surface** (but targets highest-value functions).

| Module | SciPy Symbols | FSCI Functions | Coverage | Status |
|---|---|---|---|---|
| linalg | 108 | 28 | 26% | parity_gap |
| sparse | 112 (incl. linalg) | 30 | 27% | parity_gap |
| optimize | 82 | 17 | 21% | parity_gap |
| integrate | 38 | 14 | 37% | in_progress |
| fft | 40 | 22 | 55% | parity_gap |
| special | 319 | 50 | 16% | parity_gap |
| stats | 182 | 20 fn + 14 dist | 19% | parity_gap |
| signal | 159 | 14 | 9% | in_progress |
| interpolate | 63 | 3 | 5% | in_progress |
| spatial | 23 + distance | 15 | 45% | parity_gap |

## Detailed Gap Analysis

### fsci-linalg (26% coverage)

**Implemented:** solve, solve_triangular, solve_banded, inv, det, lstsq, pinv, lu, lu_factor, lu_solve, qr, svd, svdvals, cholesky, cho_factor, cho_solve, ldl, eig, eigvals, eigh, eigvalsh, schur, hessenberg, expm, norm, matrix_rank, solve_with_casp

**Missing (HIGH priority for V1):**
- Matrix functions: `logm`, `sqrtm`, `fractional_matrix_power`, `matrix_power`, `funm`, `signm`
- Generalized decompositions: `qz`, `ordqz`
- Subspace: `orth`, `null_space`, `subspace_angles`, `polar`
- Equation solvers: `solve_sylvester`, `solve_continuous_lyapunov`, `solve_discrete_lyapunov`
- Special matrices: `toeplitz`, `circulant`, `hilbert`, `hadamard`, `block_diag`, `companion`
- Banded: `eig_banded`, `eigh_tridiagonal`, `cho_solve_banded`, `solveh_banded`

### fsci-sparse (27% coverage)

**Implemented:** COO/CSR/CSC formats, spsolve, splu, spilu, cg, pcg, gmres, eigsh, spsolve_triangular, eye, diags, random, block_diag, bmat, kron

**Missing (HIGH priority):**
- Formats: BSR, DIA, DOK, LIL
- Solvers: `bicg`, `bicgstab`, `cgs`, `lgmres`, `minres`, `qmr`, `lsqr`, `lsmr`
- Eigenvalues: `eigs` (non-symmetric), `svds` (sparse SVD)
- Operations: `expm`, `matrix_power`, `norm`
- Graph algorithms: entire `csgraph` submodule (shortest_path, connected_components, etc.)

### fsci-opt (21% coverage)

**Implemented:** minimize (BFGS, CG, Powell, NM, L-BFGS-B, Newton-CG), minimize_scalar, brentq, brenth, bisect, ridder, root_scalar, curve_fit, least_squares, line_search

**Missing (HIGH priority):**
- Constrained: `Bounds`, `LinearConstraint`, `NonlinearConstraint` (needed for constrained minimize)
- Global: `differential_evolution`, `basinhopping`, `dual_annealing`, `shgo`
- Linear programming: `linprog`, `milp`
- Multivariate root: `root`, `fsolve`
- Assignment: `linear_sum_assignment`
- Utilities: `approx_fprime`, `check_grad`

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

### fsci-special (16% coverage)

**Implemented:** gamma, gammaln, digamma, polygamma, gammainc, gammaincc, rgamma, factorial, comb, perm, zeta, beta, betaln, betainc, erf, erfc, erfinv, erfcinv, j0, j1, jn, jv, y0, y1, yn, yv, iv, kv, hankel1, hankel2, airy, ai, bi, hyp1f1, hyp2f1, ellipk, ellipe, ellipkinc, ellipeinc, lambertw, exp1, expi

**Missing (HIGH priority):**
- Orthogonal polynomials: `eval_legendre`, `eval_chebyt`, `roots_legendre`, `roots_chebyt` (42 functions)
- Spherical: `spherical_jn`, `spherical_yn`, `sph_harm_y`, `lpmv`
- Struve: `struve`, `modstruve`
- Fresnel: `fresnel`
- Dawson: `dawsn`
- Info theory: `entr`, `rel_entr`, `kl_div`
- Convenience: `sinc`, `xlogy`, `logsumexp`, `expit`, `logit`
- Elliptic Jacobi: `ellipj`
- Statistical cdfs: `ndtr`, `ndtri`, `btdtr`, `btdtri`, `fdtr`, `gdtr`

### fsci-stats (19% coverage)

**Implemented:** 14 distributions (Normal, StudentT, ChiSquared, Uniform, Exponential, F, Beta, Gamma, Weibull, Lognormal, Poisson, Cauchy, Laplace, Triangular), ttest_1samp, ttest_ind, ttest_ind_welch, linregress, pearsonr, spearmanr, rankdata

**Missing (HIGH priority):**
- ~100+ distributions (Binomial, NegBinomial, Geometric, Pareto, Rayleigh, etc.)
- Summary stats: `describe`, `skew`, `kurtosis`, `mode`, `moment`, `sem`, `iqr`, `variation`
- Tests: `ks_1samp`, `ks_2samp`, `shapiro`, `normaltest`, `mannwhitneyu`, `wilcoxon`, `kruskal`, `f_oneway`, `chi2_contingency`, `fisher_exact`
- KDE: `gaussian_kde`
- Transforms: `boxcox`, `zscore`
- Correlation: `kendalltau`
- Entropy: `entropy`, `differential_entropy`
- Fitting: `fit`
- Base classes: `rv_continuous`, `rv_discrete`

### fsci-signal (9% coverage)

**Implemented:** savgol_coeffs, savgol_filter, hann, hamming, blackman, kaiser, convolve, fftconvolve, find_peaks, butter, lfilter, filtfilt, periodogram, welch

**Missing (HIGH priority):**
- IIR design: `cheby1`, `cheby2`, `ellip`, `bessel`, `iirfilter`, `iirdesign`
- FIR design: `firwin`, `remez`, `kaiserord`
- SOS filtering: `sosfilt`, `sosfiltfilt`, `tf2sos`, `sos2tf`
- Frequency response: `freqz`, `freqs`, `group_delay`
- Spectral: `spectrogram`, `stft`, `istft`, `csd`, `coherence`
- Resampling: `resample`, `resample_poly`, `decimate`
- Waveforms: `chirp`, `sawtooth`, `square`
- LTI systems: `lti`, `dlti`, `impulse`, `step`, `lsim`
- Other: `hilbert`, `detrend`, `get_window`, `medfilt`
- Filter conversion: `tf2zpk`, `zpk2tf`, `zpk2sos`, etc.

### fsci-interpolate (5% coverage)

**Implemented:** Interp1d (linear, nearest, cubic spline), PchipInterpolator

**Missing (HIGH priority):**
- `CubicSpline` (standalone class)
- `Akima1DInterpolator`
- `BarycentricInterpolator`
- `BSpline`, `make_interp_spline`, `make_lsq_spline`
- `UnivariateSpline`, `InterpolatedUnivariateSpline`
- N-D: `RegularGridInterpolator`, `interpn`, `griddata`
- 2D: `RectBivariateSpline`, `SmoothBivariateSpline`
- RBF: `RBFInterpolator`
- Scattered: `LinearNDInterpolator`, `NearestNDInterpolator`, `CloughTocher2DInterpolator`

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

These are the most commonly used SciPy functions that we lack:

1. **`scipy.stats.describe`** + `skew`, `kurtosis`, `mode` — basic stats everyone needs
2. **`scipy.linalg.logm`** + `sqrtm` + `matrix_power` — matrix functions
3. **`scipy.optimize.differential_evolution`** — global optimization workhorse
4. **`scipy.optimize.linprog`** — linear programming
5. **`scipy.interpolate.RegularGridInterpolator`** — N-D interpolation
6. **`scipy.signal.freqz`** — filter frequency response
7. **`scipy.signal.spectrogram`** / `stft` — time-frequency analysis
8. **`scipy.signal.firwin`** — FIR filter design
9. **`scipy.special.sinc`** + `xlogy` + `logsumexp` — convenience functions
10. **`scipy.stats.ks_1samp`** + `ks_2samp` — Kolmogorov-Smirnov tests

## Required Evidence Per Feature Family

1. Differential fixture report.
2. Edge-case/adversarial test results.
3. Benchmark delta (for runtime-significant paths).
4. Documented compatibility exceptions (if any).
5. RaptorQ sidecar manifest plus decode-proof record for each durable artifact bundle.
