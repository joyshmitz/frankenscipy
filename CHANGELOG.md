# Changelog

All notable changes to FrankenSciPy are documented in this file.

FrankenSciPy is a clean-room Rust reimplementation of SciPy with a
Condition-Aware Solver Portfolio (CASP) at its core. The project has no
formal releases yet; this changelog tracks landed capabilities by date
against the `main` branch.

Repository: <https://github.com/Dicklesworthstone/frankenscipy>
License: MIT with OpenAI/Anthropic Rider

---

## [Unreleased] -- HEAD on main (workspace version 0.1.0)

### Workspace crates

| Crate | Purpose |
|---|---|
| `fsci-linalg` | Dense/structured linear algebra with CASP solver selection |
| `fsci-sparse` | CSR/CSC/COO matrices, iterative solvers, graph algorithms |
| `fsci-integrate` | ODE (IVP + BVP), quadrature, cumulative rules |
| `fsci-interpolate` | Splines (cubic, PCHIP, Akima, BSpline), scattered-data griddata |
| `fsci-opt` | Minimizers (Nelder-Mead, BFGS, CG, Powell, L-BFGS-B), root-finders, constraints |
| `fsci-fft` | Cooley-Tukey, Bluestein, real/complex/nD FFT, DCT/IDCT, Hilbert |
| `fsci-signal` | Filtering, spectral estimation, SOS, IIR/FIR design, find_peaks |
| `fsci-spatial` | KDTree, pairwise distances (pdist) |
| `fsci-special` | Gamma, beta, erf, Bessel, Airy, hypergeometric, zeta, orthogonal polynomials |
| `fsci-stats` | Continuous + discrete distributions, hypothesis tests, descriptive stats, regression |
| `fsci-arrayapi` | Contract-first Array API backend with broadcast, indexing, reduction |
| `fsci-conformance` | Differential conformance harness, RaptorQ evidence packs, parity reports |
| `fsci-runtime` | CASP runtime, execution-path tracing, schema validation |

---

### 2026-03-21 -- Cross-module hardening and graph algorithms

**Sparse -- graph traversal**
- Add BFS and DFS graph traversal algorithms on sparse adjacency matrices.
  ([d08a4f7](https://github.com/Dicklesworthstone/frankenscipy/commit/d08a4f798fe23b9f63c4e577ddbfaebc058458ab))
- Add Bellman-Ford shortest-path solver.
  ([adb66db](https://github.com/Dicklesworthstone/frankenscipy/commit/adb66db7a32181cb45e599f09b13232f796a26b7))

**Special -- NaN propagation overhaul**
- Rewrite `gammaln` and add NaN propagation to all gamma convenience functions.
  ([4337552](https://github.com/Dicklesworthstone/frankenscipy/commit/4337552cb263c2e483f36278d96c6ed64823e06c))
- Add NaN propagation for entropy functions.
  ([1614362](https://github.com/Dicklesworthstone/frankenscipy/commit/1614362889cc570c255b6d9e063a8398c1261c77))

**Stats**
- Add survival function (`sf`) to Normal distribution.
  ([0b83dae](https://github.com/Dicklesworthstone/frankenscipy/commit/0b83daea069f06295cacb3fea6c9ee05170ab957))
- Add new statistical routines (entropy, additional edge-case coverage).
  ([1614362](https://github.com/Dicklesworthstone/frankenscipy/commit/1614362889cc570c255b6d9e063a8398c1261c77))

**FFT / Signal**
- Add DCT transform variants (DCT-II, DCT-III, DCT-IV) and expand signal filter coverage.
  ([d7ac09a](https://github.com/Dicklesworthstone/frankenscipy/commit/d7ac09a9bdec6f8bb0d0d0bde8c04fcdf5b2dd0b))

**Linalg**
- Fix Sylvester equation solver; validate `arange` inputs in Array API.
  ([52d0818](https://github.com/Dicklesworthstone/frankenscipy/commit/52d081846f940cb940067a8d096f80d562a93899))
- Fix `logm` Parlett recurrence and apply rustfmt.
  ([5c7b400](https://github.com/Dicklesworthstone/frankenscipy/commit/5c7b400a9ddcc78354e37186a9a8a446f5592702))

**Integrate**
- Rewrite `cumulative_simpson` for correct per-interval output.
  ([8acb62f](https://github.com/Dicklesworthstone/frankenscipy/commit/8acb62f1a266b3239123222d09a2cf2100b8c38a))
- Add Hermite interpolation variant and `IvpSolver` trait.
  ([2b6b211](https://github.com/Dicklesworthstone/frankenscipy/commit/2b6b2114028adab9f2dc4ed99bbced00c37f1ce3))

**Signal**
- Add `lfilter_zi` + `filtfilt` steady-state initialization, window parameter threading, new window functions, and erf API cleanup.
  ([5a9f78a](https://github.com/Dicklesworthstone/frankenscipy/commit/5a9f78ab51cab8b88801c740a981e3c2ec7d400c))

**Interpolate**
- Fix Akima slope computation edge cases.
  ([adb66db](https://github.com/Dicklesworthstone/frankenscipy/commit/adb66db7a32181cb45e599f09b13232f796a26b7))

---

### 2026-03-20 -- ODE event handling overhaul and conformance regeneration

**Integrate**
- Major overhaul of ODE event handling: terminal events, direction detection, dense output interpolation.
  ([3fc730e](https://github.com/Dicklesworthstone/frankenscipy/commit/3fc730e606529c4259f21522743b5c66cb1af8bb))
- Add ODE event detection with root-finding on dense output.
  ([60c5930](https://github.com/Dicklesworthstone/frankenscipy/commit/60c59301358783bd0c4d6aed6dabd30344685632))

**Linalg / Opt**
- Correct `logm` Parlett recurrence denominator.
- Improve Nelder-Mead simplex pivot selection and Powell direction update strategy.
  ([93fb998](https://github.com/Dicklesworthstone/frankenscipy/commit/93fb998d3e6e241461ecc63926a3188c82fe9700))

**Conformance**
- Fix RaptorQ sidecar generation to serialize from JSON instead of pre-serialization bytes.
  ([89fc93a](https://github.com/Dicklesworthstone/frankenscipy/commit/89fc93a2981e6ad18da1b81ddb0c298428f34f0b))
- Regenerate P2C-001, P2C-002, P2C-007 evidence artifacts with corrected checksums.
  ([b276e59](https://github.com/Dicklesworthstone/frankenscipy/commit/b276e59f0a32dfbea580845758b78d04742c9721))

**Multi-module improvements**
- Simplify interpolation internals, improve spatial distance edge cases, harden stats distribution tails.
  ([3fc730e](https://github.com/Dicklesworthstone/frankenscipy/commit/3fc730e606529c4259f21522743b5c66cb1af8bb))
- FFT scaling fixes across forward/inverse transforms.
  ([60c5930](https://github.com/Dicklesworthstone/frankenscipy/commit/60c59301358783bd0c4d6aed6dabd30344685632))

---

### 2026-03-19 -- N-dimensional FFT, elliptic filters, solver portfolio refactor

**FFT**
- Add n-dimensional real FFT transforms (`rfftn`, `irfftn`) and full normalization mode support.
  ([1256e63](https://github.com/Dicklesworthstone/frankenscipy/commit/1256e63003a5ff05444d15089c2ee4acb89ec09b))
- Numerical corrections across FFT phase/magnitude handling.
  ([09c901c](https://github.com/Dicklesworthstone/frankenscipy/commit/09c901cdeebfa435b389a37b88008fcdfb9f5518))

**Signal**
- Add elliptic (Cauer) IIR filter design.
  ([e070b0e](https://github.com/Dicklesworthstone/frankenscipy/commit/e070b0ec22e506acb1ca33ea2b5c3eafc2e018e2))
- Fix numerical stability in filter coefficient computation.
  ([d8180d9](https://github.com/Dicklesworthstone/frankenscipy/commit/d8180d9d5eb5c458f43fe80b70f0761a47f58908))

**Special**
- Reorganize imports, expand error types, and improve Bessel function accuracy.
  ([3177df2](https://github.com/Dicklesworthstone/frankenscipy/commit/3177df2748f2011390ff30309f77b4abe403c0dd))

**Sparse**
- Add new sparse solvers (GMRES, iterative refinement).
  ([e070b0e](https://github.com/Dicklesworthstone/frankenscipy/commit/e070b0ec22e506acb1ca33ea2b5c3eafc2e018e2))

**Linalg / Runtime**
- Refactor solver portfolio: backward error tracking, numerical stability improvements, CASP calibration refinements.
  ([a83ed1a](https://github.com/Dicklesworthstone/frankenscipy/commit/a83ed1a6f4e723a248cec7361bf8f46b6f4730fd))

---

### 2026-03-18 -- Statistics expansion, BSpline/Akima, FIR filter design

**Stats**
- Add non-parametric and ANOVA tests: `f_oneway`, `mannwhitneyu`, `wilcoxon`, `kruskal`, `ranksums`.
  ([5b48848](https://github.com/Dicklesworthstone/frankenscipy/commit/5b488482dc331a188f108d0ff677fe35c6fac15e))
- Add descriptive statistics, correlation, and regression functions (+488 lines).
  ([004a3e9](https://github.com/Dicklesworthstone/frankenscipy/commit/004a3e90926522c6b1d523c8957e403804ac7fe8))
- Add goodness-of-fit tests (Kolmogorov-Smirnov, chi-squared, Shapiro-Wilk).
  ([7a1169e](https://github.com/Dicklesworthstone/frankenscipy/commit/7a1169ed185dd17d8804e2fd8b4331dbbcb2a0e3))

**Interpolate**
- Add BSpline and Akima interpolation.
  ([7a1169e](https://github.com/Dicklesworthstone/frankenscipy/commit/7a1169ed185dd17d8804e2fd8b4331dbbcb2a0e3))
- Add NaN-safe interpolation and IIR filter family support.
  ([346cc46](https://github.com/Dicklesworthstone/frankenscipy/commit/346cc46d8c350dd0e643a1fd6bcb69c8dd442592))

**Signal**
- Add FIR filter design (firwin, firwin2, remez).
  ([7a1169e](https://github.com/Dicklesworthstone/frankenscipy/commit/7a1169ed185dd17d8804e2fd8b4331dbbcb2a0e3))
- Add IIR filter families (Butterworth, Chebyshev I/II, Bessel).
  ([346cc46](https://github.com/Dicklesworthstone/frankenscipy/commit/346cc46d8c350dd0e643a1fd6bcb69c8dd442592))

**Multi-module expansion**
- Large cross-module feature drop spanning stats, signal, linalg, optimization, interpolation, spatial, and special (+6451 lines).
  ([9026a01](https://github.com/Dicklesworthstone/frankenscipy/commit/9026a01daa0d8936f2afc21b93919ba870136d09))

**Conformance**
- Update test infrastructure, parity reports, and mutex handling for concurrent conformance runs.
  ([9a1c403](https://github.com/Dicklesworthstone/frankenscipy/commit/9a1c40338712c5c9e2c223e85c00e95ef23a3958))

---

### 2026-03-17 -- Discrete distributions, SOS filtering, frequency response

**Stats**
- Add discrete distributions: Binomial, Bernoulli, Geometric, NegativeBinomial, Hypergeometric.
  ([c4568b0](https://github.com/Dicklesworthstone/frankenscipy/commit/c4568b01a83cd57324e8b2aac952079c5d06a2af))
- Fix two bugs in Hypergeometric distribution (PMF normalization, support range).
  ([aa3283b](https://github.com/Dicklesworthstone/frankenscipy/commit/aa3283b6ed35cecc92ff443719ecc69480b83485))

**Signal**
- Add frequency response analysis: `freqz`, `freqs`, `group_delay`.
  ([935983e](https://github.com/Dicklesworthstone/frankenscipy/commit/935983e1f90fa37996f23b7db0820ab432b6273f))
- Add SOS filtering: `sosfilt`, `sosfiltfilt`, `sosfilt_zi`.
  ([b2815a7](https://github.com/Dicklesworthstone/frankenscipy/commit/b2815a71b44839cde7223b2094f762fb6c793828))
- Fix unreachable early-return path in `tf2zpk` for all-zero numerator.
  ([8bd95b1](https://github.com/Dicklesworthstone/frankenscipy/commit/8bd95b12e64e4bbbfa49602a0d1db3301215b058))

**Interpolate**
- Add `NearestNDInterpolator` and `griddata` for scattered-data interpolation.
  ([eb6279a](https://github.com/Dicklesworthstone/frankenscipy/commit/eb6279af010b266c503a32ece6ea60525d8207d0))

**Special**
- Add orthogonal polynomial evaluation module (Legendre, Hermite, Laguerre, Chebyshev, Jacobi).
  ([0210ac2](https://github.com/Dicklesworthstone/frankenscipy/commit/0210ac29f792a1fec599bc3370eb4bb4b6100a2a))

---

### 2026-03-16 -- Filter conversions, constraint types, cross-module features

**Signal**
- Add full filter representation conversions: `tf2zpk`, `zpk2tf`, `tf2sos`, `sos2tf`, `zpk2sos`, `sos2zpk`.
  ([fc958ff](https://github.com/Dicklesworthstone/frankenscipy/commit/fc958ff69b8dee7d31bd99b721d8a37e59af5eee))
- Extend spectral estimation with windowed periodogram.
  ([65b3ce2](https://github.com/Dicklesworthstone/frankenscipy/commit/65b3ce2f9520064f6629dcefa1a29314ef007a26))

**Optimize**
- Add `Bounds`, `LinearConstraint`, `NonlinearConstraint` types for constrained optimization.
  ([65302a6](https://github.com/Dicklesworthstone/frankenscipy/commit/65302a601dac0b7e8a18aa7c81d15f265dfbeb9e))

**Stats / Special / Linalg**
- Add summary statistics, special-function convenience wrappers, and matrix functions (sqrtm, logm, funm) (+600 lines).
  ([23c5ff3](https://github.com/Dicklesworthstone/frankenscipy/commit/23c5ff3e19a15fd7f88c5be85fd7a895b1c60a6c))

**Stats / Spatial / Signal**
- Add `linregress`, `pdist` (pairwise distances), `welch` spectral estimation (+968 lines).
  ([f957c7d](https://github.com/Dicklesworthstone/frankenscipy/commit/f957c7d5d22cdd73034d4b7052deaa9ef1883299))

**Bug fixes**
- Correct 5 bugs in filter conversions and constraint types.
  ([a03f80d](https://github.com/Dicklesworthstone/frankenscipy/commit/a03f80dcc998f0c0013c321095d23846e85c6393))

---

### 2026-03-15 -- Signal crate bootstrap, spatial crate, ODE solvers, curve fitting

**Signal (new crate: `fsci-signal`)**
- Bootstrap signal processing module with filtering and spectral analysis (+462 lines).
  ([e132d4c](https://github.com/Dicklesworthstone/frankenscipy/commit/e132d4cc0394c85152dfdaeb08cda457b25b406c))
- Add `find_peaks` peak detection.
  ([47b6a99](https://github.com/Dicklesworthstone/frankenscipy/commit/47b6a99e7b9ae9120d38d304a9913c0fbffe20f8))
- Add convolution and window functions (Hann, Hamming, Blackman, Kaiser, etc.).
  ([937daa3](https://github.com/Dicklesworthstone/frankenscipy/commit/937daa3b16611c8eb02e2e05eda7865d20ccbb43))

**Spatial (new crate: `fsci-spatial`)**
- Add KDTree for nearest-neighbor queries.
  ([47b6a99](https://github.com/Dicklesworthstone/frankenscipy/commit/47b6a99e7b9ae9120d38d304a9913c0fbffe20f8))

**FFT**
- Add Hilbert transform.
  ([79b6092](https://github.com/Dicklesworthstone/frankenscipy/commit/79b6092ca696bfd185cb5943cc22260753df00ea))
- Correct Hilbert transform step function for odd-length inputs.
  ([e5b3f0d](https://github.com/Dicklesworthstone/frankenscipy/commit/e5b3f0d0b15bfd40ba24c88355c2c524532e99e6))
- Add specialized real FFT for power-of-2 inputs (performance optimization).
  ([ac1f717](https://github.com/Dicklesworthstone/frankenscipy/commit/ac1f717ef96efaa7b69806a5e7b9d8f1ab715202))
- Add DCT and IDCT transforms.
  ([8a47f61](https://github.com/Dicklesworthstone/frankenscipy/commit/8a47f611571b42f574da016f661c0c1cf494e624))

**Stats**
- Add Gamma and Poisson distributions.
  ([937daa3](https://github.com/Dicklesworthstone/frankenscipy/commit/937daa3b16611c8eb02e2e05eda7865d20ccbb43))
- Add Weibull and Lognormal distributions.
  ([47b6a99](https://github.com/Dicklesworthstone/frankenscipy/commit/47b6a99e7b9ae9120d38d304a9913c0fbffe20f8))
- Add F and Beta distributions.
  ([8a47f61](https://github.com/Dicklesworthstone/frankenscipy/commit/8a47f611571b42f574da016f661c0c1cf494e624))
- Add trait-level `ppf` (percent point function) for all distributions.
  ([f99b570](https://github.com/Dicklesworthstone/frankenscipy/commit/f99b5709d168a41b0e754ca1a54962d55150cf0e))
- Add statistical distribution functions and hypothesis testing (t-test, chi-squared).
  ([42abbcc](https://github.com/Dicklesworthstone/frankenscipy/commit/42abbcc5d48cac81b0ff19b9c2a2b76a75515dbf))
- Add curve fitting (`curve_fit`).
  ([9ebb9c2](https://github.com/Dicklesworthstone/frankenscipy/commit/9ebb9c2d4aa02532ce93b82f7bb68f9869955dd1))

**Interpolate**
- Add PCHIP (Piecewise Cubic Hermite Interpolating Polynomial).
  ([f99b570](https://github.com/Dicklesworthstone/frankenscipy/commit/f99b5709d168a41b0e754ca1a54962d55150cf0e))

**Integrate**
- Wire Radau and BDF into `solve_ivp` dispatch.
  ([0a02d09](https://github.com/Dicklesworthstone/frankenscipy/commit/0a02d091594d12fe20d998482530abb852402328))
- Add `odeint` convenience wrapper and `spsolve_triangular`.
  ([79b6092](https://github.com/Dicklesworthstone/frankenscipy/commit/79b6092ca696bfd185cb5943cc22260753df00ea))
- Add `solve_bvp` boundary value problem solver.
  ([447545e](https://github.com/Dicklesworthstone/frankenscipy/commit/447545ebcb0deb4bd61aa535f93297c222dfb1aa))
- Correct not-a-knot spline solver and scale BVP finite differences.
  ([132fb76](https://github.com/Dicklesworthstone/frankenscipy/commit/132fb7689633232bb0d2fad99b5e6a427b31ab36))

**Opt**
- Add Newton-CG optimizer.
  ([8a47f61](https://github.com/Dicklesworthstone/frankenscipy/commit/8a47f611571b42f574da016f661c0c1cf494e624))
- Correct Newton-CG Jacobian evaluation count.
  ([6e9e2dd](https://github.com/Dicklesworthstone/frankenscipy/commit/6e9e2dd6689df6c8c4a02b49ac18b44bdd681df3))

**Special**
- Add elliptic functions (ellipk, ellipe, ellipj).
  ([9ebb9c2](https://github.com/Dicklesworthstone/frankenscipy/commit/9ebb9c2d4aa02532ce93b82f7bb68f9869955dd1))

**Sparse**
- Add sparse eigsh solver (implicitly restarted Lanczos).
  ([71317bd](https://github.com/Dicklesworthstone/frankenscipy/commit/71317bd82cd58eaf7e7c092b7b118296768c3f20))
- Add preconditioned CG and spline boundary conditions.
  ([7a26704](https://github.com/Dicklesworthstone/frankenscipy/commit/7a26704bda817efd56d73973ee84bff55feea4b6))

**Bug fixes**
- Correct 5 bugs found during code review across multiple crates.
  ([3411647](https://github.com/Dicklesworthstone/frankenscipy/commit/3411647f3f82ef05fb9c030fb131623ab72ed60f))

---

### 2026-03-14 -- Integration/FFT/stats crate bootstraps, L-BFGS-B, Runge-Kutta

**Stats (new crate: `fsci-stats`)**
- Bootstrap with Normal, Uniform, Chi-Squared, Student-t, and Exponential distributions.
  ([b140aeb](https://github.com/Dicklesworthstone/frankenscipy/commit/b140aeb14acd169421e412b6245067496fa82383),
   [6d378cb](https://github.com/Dicklesworthstone/frankenscipy/commit/6d378cb6e964fe586f6232d92aa42848ac7704ac))

**Interpolate (new crate: `fsci-interpolate`)**
- Bootstrap with cubic spline interpolation and `cumulative_trapezoid`.
  ([cdd9049](https://github.com/Dicklesworthstone/frankenscipy/commit/cdd90493839db40074672ecf70e57a27ae870955))

**FFT**
- Implement Cooley-Tukey radix-2 FFT and Bluestein's algorithm for arbitrary lengths.
  ([66a57b0](https://github.com/Dicklesworthstone/frankenscipy/commit/66a57b003711e8f77d47d3987c66bd0fdf8f90bc))

**Opt**
- Implement L-BFGS-B bounded optimization.
  ([2344b16](https://github.com/Dicklesworthstone/frankenscipy/commit/2344b162338f882388090ccdc38881ba41093383))

**Integrate**
- Add Runge-Kutta integration API (RK23, RK45).
  ([5730fd5](https://github.com/Dicklesworthstone/frankenscipy/commit/5730fd5804ea72e6caf90804e7197fea14acecfe))
- Implement ILU(0) preconditioner and DOP853 Butcher tableau.
  ([8abf432](https://github.com/Dicklesworthstone/frankenscipy/commit/8abf432f043f59d28306edec97038cf188d51908))
- Add adaptive Gauss-Kronrod quadrature module.
  ([85d67a6](https://github.com/Dicklesworthstone/frankenscipy/commit/85d67a695442c36a78aec2d2657e9c0f7abf0380))
- Add Nelder-Mead optimizer, BDF stiff solver, and quadrature rules.
  ([8c20565](https://github.com/Dicklesworthstone/frankenscipy/commit/8c20565d76684f4ff0711e23e9b33b19c88341fb))
- Correct DOP853 Butcher tableau coefficients.
  ([6d378cb](https://github.com/Dicklesworthstone/frankenscipy/commit/6d378cb6e964fe586f6232d92aa42848ac7704ac))

**Special**
- Add Riemann zeta function.
  ([f480b75](https://github.com/Dicklesworthstone/frankenscipy/commit/f480b752d6c70985a9fce1f0c97337f664be4047))
- Implement `hyp1f1`/`hyp2f1` series evaluation and matrix exponential (`expm`).
  ([32d737c](https://github.com/Dicklesworthstone/frankenscipy/commit/32d737c4890a1879305f90451012f3585201861e))
- Add Airy functions and combinatorial functions.
  ([8c20565](https://github.com/Dicklesworthstone/frankenscipy/commit/8c20565d76684f4ff0711e23e9b33b19c88341fb))

**Linalg**
- Add LDL decomposition.
  ([8c20565](https://github.com/Dicklesworthstone/frankenscipy/commit/8c20565d76684f4ff0711e23e9b33b19c88341fb))
- Expand decompositions (Schur, Hessenberg) and sparse matrix construction helpers.
  ([d5c7dff](https://github.com/Dicklesworthstone/frankenscipy/commit/d5c7dff39c3805c9dc971deba08fffb6ef5f72bf))

**Bug fixes**
- Fix Bluestein DFT sign convention, Airy function phase, BDF efficiency constants, L-BFGS-B dead code.
  ([01eb649](https://github.com/Dicklesworthstone/frankenscipy/commit/01eb64904284921c3a6789c21f376712c7c7e1a0))

---

### 2026-03-13 -- Conformance expansion and feature-parity tracking

**Conformance**
- Expand conformance test suite with new parity tracking for all landed modules.
  ([180e906](https://github.com/Dicklesworthstone/frankenscipy/commit/180e906deff82b02ebf602eb2dbae177975bc6fb))
- Add test fixtures for sparse operations, FFT core, and runtime CASP.
  ([72845e3](https://github.com/Dicklesworthstone/frankenscipy/commit/72845e30f09671ea8d104fa2c37eb358dbc6ce86))

---

### 2026-03-12

**Opt**
- Reject unsupported initial guesses in root-finders instead of silently selecting Ridder.
  ([93edeca](https://github.com/Dicklesworthstone/frankenscipy/commit/93edeca6032910b9fd273700e382007a04d67776))

---

### 2026-03-10

**Conformance / Integrate / FFT**
- Expand E2E test coverage with new benchmarks and IVP integration API tests.
  ([72f8aa1](https://github.com/Dicklesworthstone/frankenscipy/commit/72f8aa1921c3416f59ca8149f9c780d5d0c35fca))

---

### 2026-03-04 -- Conformance quality gates and CI pipeline

**Conformance**
- Add conformance quality gates, CI pipeline, benchmarks, and forensic analysis tooling.
  ([f1e61c0](https://github.com/Dicklesworthstone/frankenscipy/commit/f1e61c0b3b71b67dbd0492a1013569762dd251f5))
- Add E2E and evidence conformance test suites across all subsystems.
  ([0aac03a](https://github.com/Dicklesworthstone/frankenscipy/commit/0aac03a222226def07e8adc5f73d7c053eace2d5))
- Add E2E scenario tests for FFT backend routing (FSCI-P2C-005).
  ([89a2f5c](https://github.com/Dicklesworthstone/frankenscipy/commit/89a2f5cc35837fe31ec68968cd5b21d593d1815e))
- Add FFT differentiation tests and expand DCT transform coverage.
  ([493b3be](https://github.com/Dicklesworthstone/frankenscipy/commit/493b3bea7200da15bf1bdf4bb0fe5421de3906bf))
- Wire `fsci-fft` as workspace dependency to conformance crate.
  ([9a324a4](https://github.com/Dicklesworthstone/frankenscipy/commit/9a324a4b40feb17b66d8d421bdcaaf1cf47f99d0))

---

### 2026-03-03 -- Special functions, Array API backend, P2C-003/006/007 conformance packets

**Special**
- Implement gamma, beta, and error function families (`gamma`, `lgamma`, `digamma`, `beta`, `betaln`, `erf`, `erfc`, `erfinv`).
  ([d6467d2](https://github.com/Dicklesworthstone/frankenscipy/commit/d6467d2f1fad59d37d06fbbdc2eff1f9d9d3ce87))
- Implement integer-order Bessel functions (J0, J1, Jn, Y0, Y1, Yn), regularized incomplete gamma, and structured trace logging.
  ([c92b075](https://github.com/Dicklesworthstone/frankenscipy/commit/c92b0758de912855c6e67c7c542b3679e228538e))

**Array API**
- Implement `CoreArrayBackend` with broadcast, indexing, reduction, and type promotion.
  ([1b5700e](https://github.com/Dicklesworthstone/frankenscipy/commit/1b5700e83014c3bb809c0ae1137342d3a43f34e9))
- Add comprehensive unit tests and proptest suites for all Array API modules.
  ([a2eabd2](https://github.com/Dicklesworthstone/frankenscipy/commit/a2eabd2e75885954948f02173aa40fee6e735bde))

**Performance**
- Replace per-element unravel/ravel with incremental coordinate advancement in broadcast (Array API).
  ([2838ba6](https://github.com/Dicklesworthstone/frankenscipy/commit/2838ba6ae61cbec677c17700fd76fce002256b2c))
- Inline RMS norm to eliminate temporary vector allocations in step-size control.
  ([b867518](https://github.com/Dicklesworthstone/frankenscipy/commit/b86751887c7ac8b82a25231d5d1123beff01cb2f))
- Add criterion benchmark suite for core Array API operations.
  ([274f0ec](https://github.com/Dicklesworthstone/frankenscipy/commit/274f0ec8f4390d2545bbfc7e2e9044939e13aee6))

**Conformance**
- Add FSCI-P2C-003 optimization conformance packet with E2E scenarios.
  ([43bb37e](https://github.com/Dicklesworthstone/frankenscipy/commit/43bb37e4859dc1c560e764bc8c21657bd2493a7d))
- Wire Array API core operations into differential conformance harness (FSCI-P2C-007).
  ([4f539b6](https://github.com/Dicklesworthstone/frankenscipy/commit/4f539b67251f69123f319f5ef0f7d1bcfd48d7ca))
- Add special-function conformance harness (FSCI-P2C-006).
  ([1b5700e](https://github.com/Dicklesworthstone/frankenscipy/commit/1b5700e83014c3bb809c0ae1137342d3a43f34e9))
- Add P2C-007 evidence pack test and performance profile artifacts.
  ([f27349e](https://github.com/Dicklesworthstone/frankenscipy/commit/f27349ead58337ea5775c8bbfe916b7cf8946f7b),
   [50edb3d](https://github.com/Dicklesworthstone/frankenscipy/commit/50edb3d9b269f948248d1851c2c59ffb41ec6440))
- Add IVP solver performance profiling harness (P2C-001-H).
  ([8f09e95](https://github.com/Dicklesworthstone/frankenscipy/commit/8f09e95d99aa70731e302fb1734a001a14d16785))

---

### 2026-03-02 -- Optimizer expansion and contract-first scaffolding

**Opt**
- Implement BFGS, Conjugate Gradient (Polak-Ribiere+), and Powell minimizers.
- Implement `brentq`, `bisect`, and `ridder` root-finders.
  ([bf634ec](https://github.com/Dicklesworthstone/frankenscipy/commit/bf634ecc1e7da33af4806e5bf11d4523b2e2fddf))
- Add comprehensive minimizer and root-finder test suites with proptest.
  ([be93eab](https://github.com/Dicklesworthstone/frankenscipy/commit/be93eabe6e98d462e43b24c07d227f729b318293))

**Array API**
- Scaffold contract-first Array API foundation crate with module structure and type stubs.
  ([8e773d3](https://github.com/Dicklesworthstone/frankenscipy/commit/8e773d37b60c79536db72ed6283db5b0947ec779))

**Special**
- Scaffold contract-first special function modules with dispatch plans for all target functions.
  ([28a717f](https://github.com/Dicklesworthstone/frankenscipy/commit/28a717f125e7214d5792823afdea7018876e6a4e))

---

### 2026-02-25 -- FFT scaffold and dependency migration

**FFT**
- Scaffold contract-first FFT module with transforms, plan cache, and helper utilities.
  ([e490d08](https://github.com/Dicklesworthstone/frankenscipy/commit/e490d083f3eaf96582bd906caa5158277e47689f))

**Deps**
- Switch `asupersync` and `ftui` from local path dependencies to crates.io.
  ([9c1d6d8](https://github.com/Dicklesworthstone/frankenscipy/commit/9c1d6d805a5665e421fa1c5a171731aa3383d1b4))

---

### 2026-02-21 -- License and branding

- Add MIT license with OpenAI/Anthropic Rider.
  ([2888129](https://github.com/Dicklesworthstone/frankenscipy/commit/2888129fa7f3fe95183a84d487c06fc999490147))
- Add GitHub social preview image.
  ([47b8782](https://github.com/Dicklesworthstone/frankenscipy/commit/47b87822815fefe3a31521ba8eafce08efd3856d))

---

### 2026-02-15 -- Dependency version pinning

- Pin `asupersync` to v0.2.0 and refresh `Cargo.lock`.
  ([1118492](https://github.com/Dicklesworthstone/frankenscipy/commit/111849231ccfb87a27c55052f415c6b0b310fac8))

---

### 2026-02-14 -- Conformance harness, runtime architecture, sparse tests

**Conformance**
- Add E2E linear algebra conformance test suite.
  ([9ade3fb](https://github.com/Dicklesworthstone/frankenscipy/commit/9ade3fbb71a7879070cfbbdc2eda8799d41bd459))
- Add P2C-002 evidence/perf/RaptorQ test suites and expand benchmark coverage.
  ([b7d7614](https://github.com/Dicklesworthstone/frankenscipy/commit/b7d7614c7b9f7d79877129c37415f9a4953624cd))
- Add schema validation tests and runtime property tests.
  ([e6666dd](https://github.com/Dicklesworthstone/frankenscipy/commit/e6666dd05c019d74c74f84d16848e5b2f83531f4))

**Linalg / Runtime**
- Add `lstsq` driver dispatch with CASP-aware algorithm selection.
  ([9b921bd](https://github.com/Dicklesworthstone/frankenscipy/commit/9b921bd69c71748e24d8c9819964e7e1c744d5f7))
- Build differential conformance harness with artifact governance.
  ([9b921bd](https://github.com/Dicklesworthstone/frankenscipy/commit/9b921bd69c71748e24d8c9819964e7e1c744d5f7))

**Integrate**
- Expand integration solver with dense output and step-size control refinements.
  ([f78ada6](https://github.com/Dicklesworthstone/frankenscipy/commit/f78ada6f2596589e25cbacbb40ef41c3591f24e6))

**Sparse**
- Expand sparse matrix operations (CSR/CSC/COO construction, arithmetic, conversion).
  ([a870010](https://github.com/Dicklesworthstone/frankenscipy/commit/a87001061d76b849ac84fd530e84449b27d01921))
- Add comprehensive unit and property-based test coverage for sparse matrices.
  ([22a79cd](https://github.com/Dicklesworthstone/frankenscipy/commit/22a79cdeef6a61d356d912a20b430c998b8c2cad))

**Opt**
- Scaffold optimization module with line-search, minimize, and root dispatchers.
  ([8efb237](https://github.com/Dicklesworthstone/frankenscipy/commit/8efb237020460e5ab0e297fab490ee790c67027d))

---

### 2026-02-13 -- Initial commit

**Project foundation**
- Initial commit establishing clean-room Rust reimplementation of SciPy.
  ([55f1ee9](https://github.com/Dicklesworthstone/frankenscipy/commit/55f1ee94577f2b67e6242b93ac459998a47aa797))
- Workspace with 9 crates: `fsci-linalg`, `fsci-sparse`, `fsci-integrate`, `fsci-opt`, `fsci-fft`, `fsci-special`, `fsci-arrayapi`, `fsci-conformance`, `fsci-runtime`.
- Dense linear algebra: LU, QR, Cholesky, SVD, eigendecomposition, least-squares.
- CASP solver portfolio with conformal calibration for adaptive solver selection.
  ([c129019](https://github.com/Dicklesworthstone/frankenscipy/commit/c12901945d5ab9ee8313ddd9c1567a6cb374cdad))
- Conformance test harness with FSCI-P2C-001 (tolerance validation) and FSCI-P2C-002 (dense linalg) packets.
- RaptorQ + decode-proof artifact generation.
- Interactive `ftui` dashboard binary for artifact navigation.
- Python oracle capture script for SciPy-vs-Rust comparison.
- Criterion benchmark harness for linalg operations.
  ([0a22fb4](https://github.com/Dicklesworthstone/frankenscipy/commit/0a22fb41d5585d0eb72c77caf58087474f8b3852))
- Project charter, comprehensive spec, porting plan, and architecture docs.

---

## Conformance Packets

| Packet ID | Domain | Introduced |
|---|---|---|
| FSCI-P2C-001 | Tolerance validation | 2026-02-13 |
| FSCI-P2C-002 | Dense linear algebra | 2026-02-13 |
| FSCI-P2C-003 | Optimization | 2026-03-03 |
| FSCI-P2C-004 | Sparse operations | 2026-03-13 |
| FSCI-P2C-005 | FFT core | 2026-03-04 |
| FSCI-P2C-006 | Special functions | 2026-03-03 |
| FSCI-P2C-007 | Array API | 2026-03-03 |
| FSCI-P2C-008 | Runtime CASP | 2026-03-13 |

---

## Statistics

- **Total commits**: 151 (as of 2026-03-21)
- **Rust source**: ~134,000 lines across 13 workspace crates
- **Crate creation timeline**: 9 crates at initial commit (2026-02-13), `fsci-interpolate` + `fsci-stats` (2026-03-14), `fsci-signal` + `fsci-spatial` (2026-03-15)
- **No tagged releases yet** -- workspace version is `0.1.0`
