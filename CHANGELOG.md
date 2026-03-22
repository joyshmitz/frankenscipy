# Changelog

All notable changes to FrankenSciPy are documented in this file.

FrankenSciPy is a clean-room Rust reimplementation of SciPy with a
Condition-Aware Solver Portfolio (CASP) at its core. The project has no
formal releases yet; this changelog tracks landed capabilities organized
by domain against the `main` branch.

Repository: <https://github.com/Dicklesworthstone/frankenscipy>
License: MIT with OpenAI/Anthropic Rider

---

## [Unreleased] -- HEAD on main (workspace version 0.1.0)

### Workspace crates (13 crates, ~66,000 lines of Rust source + ~25,000 lines of tests/benchmarks)

| Crate | Lines | Purpose |
|---|---|---|
| `fsci-linalg` | ~5,400 | Dense/structured linear algebra with CASP solver selection |
| `fsci-sparse` | ~6,300 | CSR/CSC/COO matrices, iterative solvers, graph algorithms |
| `fsci-integrate` | ~4,900 | ODE (IVP + BVP), quadrature, cumulative rules |
| `fsci-interpolate` | ~3,200 | Splines (cubic, PCHIP, Akima, BSpline), scattered-data griddata |
| `fsci-opt` | ~8,400 | Minimizers (Nelder-Mead, BFGS, CG, Powell, L-BFGS-B, Newton-CG), root-finders, constraints |
| `fsci-fft` | ~2,700 | Cooley-Tukey, Bluestein, real/complex/nD FFT, DCT/IDCT, Hilbert |
| `fsci-signal` | ~6,100 | Filtering, spectral estimation, SOS, IIR/FIR design, find_peaks |
| `fsci-spatial` | ~1,400 | KDTree, pairwise distances (pdist) |
| `fsci-special` | ~9,500 | Gamma, beta, erf, Bessel, Airy, hypergeometric, zeta, elliptic, orthogonal polynomials |
| `fsci-stats` | ~6,000 | Continuous + discrete distributions, hypothesis tests, descriptive stats, regression |
| `fsci-arrayapi` | ~2,500 | Contract-first Array API backend with broadcast, indexing, reduction |
| `fsci-conformance` | ~8,800 | Differential conformance harness, RaptorQ evidence packs, parity reports |
| `fsci-runtime` | ~1,300 | CASP runtime, execution-path tracing, schema validation |

---

## Linear Algebra (`fsci-linalg`)

Landed 2026-02-13. Core dense linear algebra with the Condition-Aware Solver Portfolio (CASP) that drives the project's identity.

### Decompositions and solvers

- LU, QR, Cholesky, SVD, eigendecomposition, and least-squares solvers at initial commit.
  ([55f1ee9](https://github.com/Dicklesworthstone/frankenscipy/commit/55f1ee94577f2b67e6242b93ac459998a47aa797))
- CASP solver portfolio with conformal calibration for adaptive solver selection.
  ([c129019](https://github.com/Dicklesworthstone/frankenscipy/commit/c12901945d5ab9ee8313ddd9c1567a6cb374cdad))
- `lstsq` driver dispatch with CASP-aware algorithm selection and differential conformance harness.
  ([9b921bd](https://github.com/Dicklesworthstone/frankenscipy/commit/9b921bd69c71748e24d8c9819964e7e1c744d5f7))
- LDL decomposition.
  ([8c20565](https://github.com/Dicklesworthstone/frankenscipy/commit/8c20565d76684f4ff0711e23e9b33b19c88341fb))
- Schur and Hessenberg decompositions; sparse matrix construction helpers.
  ([d5c7dff](https://github.com/Dicklesworthstone/frankenscipy/commit/d5c7dff39c3805c9dc971deba08fffb6ef5f72bf))

### Matrix functions

- Matrix exponential (`expm`) via Pade approximation.
  ([32d737c](https://github.com/Dicklesworthstone/frankenscipy/commit/32d737c4890a1879305f90451012f3585201861e))
- Matrix square root (`sqrtm`), matrix logarithm (`logm`), general matrix function (`funm`).
  ([23c5ff3](https://github.com/Dicklesworthstone/frankenscipy/commit/23c5ff3e19a15fd7f88c5be85fd7a895b1c60a6c))
- Fix Sylvester equation solver; validate `arange` inputs in Array API.
  ([52d0818](https://github.com/Dicklesworthstone/frankenscipy/commit/52d081846f940cb940067a8d096f80d562a93899))

### Solver portfolio improvements

- Refactor solver portfolio: backward error tracking, numerical stability improvements, CASP calibration refinements.
  ([a83ed1a](https://github.com/Dicklesworthstone/frankenscipy/commit/a83ed1a6f4e723a248cec7361bf8f46b6f4730fd))
- Fix `logm` Parlett recurrence denominator, improve simplex pivot selection.
  ([93fb998](https://github.com/Dicklesworthstone/frankenscipy/commit/93fb998d3e6e241461ecc63926a3188c82fe9700))
- Fix `logm` Parlett recurrence (second pass) and apply rustfmt.
  ([5c7b400](https://github.com/Dicklesworthstone/frankenscipy/commit/5c7b400a9ddcc78354e37186a9a8a446f5592702))
- Linalg improvements and parity report updates.
  ([359ab72](https://github.com/Dicklesworthstone/frankenscipy/commit/359ab725a12bcd4dadc8398c3087f8a3102a88e1))

### Benchmarks

- Criterion benchmark harness for linear algebra operations.
  ([0a22fb4](https://github.com/Dicklesworthstone/frankenscipy/commit/0a22fb41d5585d0eb72c77caf58087474f8b3852))

---

## Sparse Matrices (`fsci-sparse`)

Scaffolded 2026-02-13; expanded significantly through 2026-03-21.

### Matrix formats and arithmetic

- CSR/CSC/COO construction, arithmetic, format conversion.
  ([a870010](https://github.com/Dicklesworthstone/frankenscipy/commit/a87001061d76b849ac84fd530e84449b27d01921))
- Comprehensive unit and property-based test coverage.
  ([22a79cd](https://github.com/Dicklesworthstone/frankenscipy/commit/22a79cdeef6a61d356d912a20b430c998b8c2cad))

### Iterative solvers

- Expand sparse solver algorithms (CG, BiCGSTAB).
  ([5730fd5](https://github.com/Dicklesworthstone/frankenscipy/commit/5730fd5804ea72e6caf90804e7197fea14acecfe))
- ILU(0) preconditioner.
  ([8abf432](https://github.com/Dicklesworthstone/frankenscipy/commit/8abf432f043f59d28306edec97038cf188d51908))
- Preconditioned CG.
  ([7a26704](https://github.com/Dicklesworthstone/frankenscipy/commit/7a26704bda817efd56d73973ee84bff55feea4b6))
- `spsolve_triangular` for triangular sparse systems.
  ([79b6092](https://github.com/Dicklesworthstone/frankenscipy/commit/79b6092ca696bfd185cb5943cc22260753df00ea))
- GMRES and iterative refinement solvers.
  ([e070b0e](https://github.com/Dicklesworthstone/frankenscipy/commit/e070b0ec22e506acb1ca33ea2b5c3eafc2e018e2))

### Eigensolvers

- Sparse `eigsh` solver (implicitly restarted Lanczos).
  ([71317bd](https://github.com/Dicklesworthstone/frankenscipy/commit/71317bd82cd58eaf7e7c092b7b118296768c3f20))

### Graph algorithms

- Bellman-Ford shortest-path solver.
  ([adb66db](https://github.com/Dicklesworthstone/frankenscipy/commit/adb66db7a32181cb45e599f09b13232f796a26b7))
- BFS and DFS graph traversal on sparse adjacency matrices.
  ([d08a4f7](https://github.com/Dicklesworthstone/frankenscipy/commit/d08a4f798fe23b9f63c4e577ddbfaebc058458ab))

---

## Integration (`fsci-integrate`)

Landed 2026-02-13 with basic IVP infrastructure; expanded through 2026-03-21.

### Initial value problems (IVP)

- Runge-Kutta integration API (RK23, RK45).
  ([5730fd5](https://github.com/Dicklesworthstone/frankenscipy/commit/5730fd5804ea72e6caf90804e7197fea14acecfe))
- DOP853 Butcher tableau.
  ([8abf432](https://github.com/Dicklesworthstone/frankenscipy/commit/8abf432f043f59d28306edec97038cf188d51908))
- BDF stiff solver.
  ([8c20565](https://github.com/Dicklesworthstone/frankenscipy/commit/8c20565d76684f4ff0711e23e9b33b19c88341fb))
- Wire Radau and BDF into `solve_ivp` dispatch.
  ([0a02d09](https://github.com/Dicklesworthstone/frankenscipy/commit/0a02d091594d12fe20d998482530abb852402328))
- `odeint` convenience wrapper.
  ([79b6092](https://github.com/Dicklesworthstone/frankenscipy/commit/79b6092ca696bfd185cb5943cc22260753df00ea))
- Dense output and step-size control refinements.
  ([f78ada6](https://github.com/Dicklesworthstone/frankenscipy/commit/f78ada6f2596589e25cbacbb40ef41c3591f24e6))
- `IvpSolver` trait and Hermite interpolation variant.
  ([2b6b211](https://github.com/Dicklesworthstone/frankenscipy/commit/2b6b2114028adab9f2dc4ed99bbced00c37f1ce3))

### ODE event handling

- ODE event detection with root-finding on dense output.
  ([60c5930](https://github.com/Dicklesworthstone/frankenscipy/commit/60c59301358783bd0c4d6aed6dabd30344685632))
- Major overhaul: terminal events, direction detection, dense output interpolation.
  ([3fc730e](https://github.com/Dicklesworthstone/frankenscipy/commit/3fc730e606529c4259f21522743b5c66cb1af8bb))

### Boundary value problems (BVP)

- `solve_bvp` boundary value problem solver.
  ([447545e](https://github.com/Dicklesworthstone/frankenscipy/commit/447545ebcb0deb4bd61aa535f93297c222dfb1aa))
- Correct not-a-knot spline solver and scale BVP finite differences.
  ([132fb76](https://github.com/Dicklesworthstone/frankenscipy/commit/132fb7689633232bb0d2fad99b5e6a427b31ab36))

### Quadrature

- Adaptive Gauss-Kronrod quadrature module.
  ([85d67a6](https://github.com/Dicklesworthstone/frankenscipy/commit/85d67a695442c36a78aec2d2657e9c0f7abf0380))
- Additional quadrature rules.
  ([8c20565](https://github.com/Dicklesworthstone/frankenscipy/commit/8c20565d76684f4ff0711e23e9b33b19c88341fb))
- `cumulative_trapezoid`.
  ([cdd9049](https://github.com/Dicklesworthstone/frankenscipy/commit/cdd90493839db40074672ecf70e57a27ae870955))
- Rewrite `cumulative_simpson` for correct per-interval output.
  ([8acb62f](https://github.com/Dicklesworthstone/frankenscipy/commit/8acb62f1a266b3239123222d09a2cf2100b8c38a))

### Bug fixes

- Correct DOP853 Butcher tableau coefficients.
  ([6d378cb](https://github.com/Dicklesworthstone/frankenscipy/commit/6d378cb6e964fe586f6232d92aa42848ac7704ac))
- Fix BDF efficiency constants.
  ([01eb649](https://github.com/Dicklesworthstone/frankenscipy/commit/01eb64904284921c3a6789c21f376712c7c7e1a0))

### Performance

- Inline RMS norm to eliminate temporary vector allocations in step-size control.
  ([b867518](https://github.com/Dicklesworthstone/frankenscipy/commit/b86751887c7ac8b82a25231d5d1123beff01cb2f))

---

## Interpolation (`fsci-interpolate`)

New crate bootstrapped 2026-03-14.

### Splines

- Cubic spline interpolation at crate bootstrap.
  ([cdd9049](https://github.com/Dicklesworthstone/frankenscipy/commit/cdd90493839db40074672ecf70e57a27ae870955))
- PCHIP (Piecewise Cubic Hermite Interpolating Polynomial).
  ([f99b570](https://github.com/Dicklesworthstone/frankenscipy/commit/f99b5709d168a41b0e754ca1a54962d55150cf0e))
- BSpline and Akima interpolation.
  ([7a1169e](https://github.com/Dicklesworthstone/frankenscipy/commit/7a1169ed185dd17d8804e2fd8b4331dbbcb2a0e3))
- Spline boundary conditions (clamped, not-a-knot, natural).
  ([7a26704](https://github.com/Dicklesworthstone/frankenscipy/commit/7a26704bda817efd56d73973ee84bff55feea4b6))
- Fix Akima slope computation edge cases.
  ([adb66db](https://github.com/Dicklesworthstone/frankenscipy/commit/adb66db7a32181cb45e599f09b13232f796a26b7))

### Scattered data

- `NearestNDInterpolator` and `griddata` for scattered-data interpolation.
  ([eb6279a](https://github.com/Dicklesworthstone/frankenscipy/commit/eb6279af010b266c503a32ece6ea60525d8207d0))

### NaN handling

- NaN-safe interpolation.
  ([346cc46](https://github.com/Dicklesworthstone/frankenscipy/commit/346cc46d8c350dd0e643a1fd6bcb69c8dd442592))

---

## Optimization (`fsci-opt`)

Scaffolded 2026-02-14; core algorithms landed 2026-03-02.

### Minimizers

- BFGS, Conjugate Gradient (Polak-Ribiere+), and Powell minimizers.
  ([bf634ec](https://github.com/Dicklesworthstone/frankenscipy/commit/bf634ecc1e7da33af4806e5bf11d4523b2e2fddf))
- L-BFGS-B bounded optimization.
  ([2344b16](https://github.com/Dicklesworthstone/frankenscipy/commit/2344b162338f882388090ccdc38881ba41093383))
- Nelder-Mead optimizer.
  ([8c20565](https://github.com/Dicklesworthstone/frankenscipy/commit/8c20565d76684f4ff0711e23e9b33b19c88341fb))
- Newton-CG optimizer.
  ([8a47f61](https://github.com/Dicklesworthstone/frankenscipy/commit/8a47f611571b42f574da016f661c0c1cf494e624))
- Expand minimization algorithms (trust-region variants).
  ([d612bd6](https://github.com/Dicklesworthstone/frankenscipy/commit/d612bd6bb8ff4a433cd56aa8af18f3f56c256d8d))
- Enhance Powell direction update strategy.
  ([93fb998](https://github.com/Dicklesworthstone/frankenscipy/commit/93fb998d3e6e241461ecc63926a3188c82fe9700))
- Fix Powell bracketing.
  ([8acb62f](https://github.com/Dicklesworthstone/frankenscipy/commit/8acb62f1a266b3239123222d09a2cf2100b8c38a))

### Root-finders

- `brentq`, `bisect`, and `ridder` root-finders.
  ([bf634ec](https://github.com/Dicklesworthstone/frankenscipy/commit/bf634ecc1e7da33af4806e5bf11d4523b2e2fddf))
- Reject unsupported initial guesses instead of silently selecting Ridder.
  ([93edeca](https://github.com/Dicklesworthstone/frankenscipy/commit/93edeca6032910b9fd273700e382007a04d67776))

### Curve fitting

- `curve_fit` nonlinear least-squares curve fitting.
  ([9ebb9c2](https://github.com/Dicklesworthstone/frankenscipy/commit/9ebb9c2d4aa02532ce93b82f7bb68f9869955dd1))

### Constraint types

- `Bounds`, `LinearConstraint`, `NonlinearConstraint` types for constrained optimization.
  ([65302a6](https://github.com/Dicklesworthstone/frankenscipy/commit/65302a601dac0b7e8a18aa7c81d15f265dfbeb9e))

### Line search

- Line-search, minimize, and root dispatch scaffolding.
  ([8efb237](https://github.com/Dicklesworthstone/frankenscipy/commit/8efb237020460e5ab0e297fab490ee790c67027d))
- Expand quadrature, sparse linalg, and line search algorithms.
  ([a9010d8](https://github.com/Dicklesworthstone/frankenscipy/commit/a9010d8b657cde4196c8a8351ce21367878425c3))

### Bug fixes

- Correct Newton-CG Jacobian evaluation count.
  ([6e9e2dd](https://github.com/Dicklesworthstone/frankenscipy/commit/6e9e2dd6689df6c8c4a02b49ac18b44bdd681df3))
- Fix numerical stability in optimizer internals.
  ([d8180d9](https://github.com/Dicklesworthstone/frankenscipy/commit/d8180d9d5eb5c458f43fe80b70f0761a47f58908))
- Remove L-BFGS-B dead code.
  ([01eb649](https://github.com/Dicklesworthstone/frankenscipy/commit/01eb64904284921c3a6789c21f376712c7c7e1a0))
- Correct 5 bugs in filter conversions and constraint types.
  ([a03f80d](https://github.com/Dicklesworthstone/frankenscipy/commit/a03f80dcc998f0c0013c321095d23846e85c6393))

### Tests

- Comprehensive minimizer and root-finder test suites with proptest.
  ([be93eab](https://github.com/Dicklesworthstone/frankenscipy/commit/be93eabe6e98d462e43b24c07d227f729b318293))

---

## FFT (`fsci-fft`)

Scaffolded 2026-02-25; core algorithms landed 2026-03-14.

### Core transforms

- Scaffold contract-first FFT module with transforms, plan cache, and helpers.
  ([e490d08](https://github.com/Dicklesworthstone/frankenscipy/commit/e490d083f3eaf96582bd906caa5158277e47689f))
- Cooley-Tukey radix-2 FFT and Bluestein's algorithm for arbitrary lengths.
  ([66a57b0](https://github.com/Dicklesworthstone/frankenscipy/commit/66a57b003711e8f77d47d3987c66bd0fdf8f90bc))
- DCT and IDCT transforms.
  ([8a47f61](https://github.com/Dicklesworthstone/frankenscipy/commit/8a47f611571b42f574da016f661c0c1cf494e624))
- DCT transform variants (DCT-II, DCT-III, DCT-IV).
  ([d7ac09a](https://github.com/Dicklesworthstone/frankenscipy/commit/d7ac09a9bdec6f8bb0d0d0bde8c04fcdf5b2dd0b))

### Multi-dimensional transforms

- N-dimensional real FFT transforms (`rfftn`, `irfftn`) with full normalization mode support.
  ([1256e63](https://github.com/Dicklesworthstone/frankenscipy/commit/1256e63003a5ff05444d15089c2ee4acb89ec09b))

### Analytic signal

- Hilbert transform.
  ([79b6092](https://github.com/Dicklesworthstone/frankenscipy/commit/79b6092ca696bfd185cb5943cc22260753df00ea))

### Performance

- Specialized real FFT for power-of-2 inputs.
  ([ac1f717](https://github.com/Dicklesworthstone/frankenscipy/commit/ac1f717ef96efaa7b69806a5e7b9d8f1ab715202))

### Bug fixes

- Fix Bluestein DFT sign convention.
  ([01eb649](https://github.com/Dicklesworthstone/frankenscipy/commit/01eb64904284921c3a6789c21f376712c7c7e1a0))
- Correct Hilbert transform step function for odd-length inputs.
  ([e5b3f0d](https://github.com/Dicklesworthstone/frankenscipy/commit/e5b3f0d0b15bfd40ba24c88355c2c524532e99e6))
- FFT scaling fixes across forward/inverse transforms.
  ([60c5930](https://github.com/Dicklesworthstone/frankenscipy/commit/60c59301358783bd0c4d6aed6dabd30344685632))
- Numerical corrections across FFT phase/magnitude handling.
  ([09c901c](https://github.com/Dicklesworthstone/frankenscipy/commit/09c901cdeebfa435b389a37b88008fcdfb9f5518))

---

## Signal Processing (`fsci-signal`)

New crate bootstrapped 2026-03-15.

### Filtering

- Signal processing module with filtering and spectral analysis (+462 lines).
  ([e132d4c](https://github.com/Dicklesworthstone/frankenscipy/commit/e132d4cc0394c85152dfdaeb08cda457b25b406c))
- Convolution and window functions (Hann, Hamming, Blackman, Kaiser, etc.).
  ([937daa3](https://github.com/Dicklesworthstone/frankenscipy/commit/937daa3b16611c8eb02e2e05eda7865d20ccbb43))
- `lfilter_zi` and `filtfilt` steady-state initialization, window parameter threading, new window functions.
  ([5a9f78a](https://github.com/Dicklesworthstone/frankenscipy/commit/5a9f78ab51cab8b88801c740a981e3c2ec7d400c))

### SOS (second-order sections)

- `sosfilt`, `sosfiltfilt`, `sosfilt_zi`.
  ([b2815a7](https://github.com/Dicklesworthstone/frankenscipy/commit/b2815a71b44839cde7223b2094f762fb6c793828))

### Filter design

- FIR filter design (`firwin`, `firwin2`, `remez`).
  ([7a1169e](https://github.com/Dicklesworthstone/frankenscipy/commit/7a1169ed185dd17d8804e2fd8b4331dbbcb2a0e3))
- IIR filter families: Butterworth, Chebyshev I/II, Bessel.
  ([346cc46](https://github.com/Dicklesworthstone/frankenscipy/commit/346cc46d8c350dd0e643a1fd6bcb69c8dd442592))
- Elliptic (Cauer) IIR filter design.
  ([e070b0e](https://github.com/Dicklesworthstone/frankenscipy/commit/e070b0ec22e506acb1ca33ea2b5c3eafc2e018e2))

### Filter representation conversions

- `tf2zpk`, `zpk2tf`, `tf2sos`, `sos2tf`, `zpk2sos`, `sos2zpk`.
  ([fc958ff](https://github.com/Dicklesworthstone/frankenscipy/commit/fc958ff69b8dee7d31bd99b721d8a37e59af5eee))

### Frequency response analysis

- `freqz`, `freqs`, `group_delay`.
  ([935983e](https://github.com/Dicklesworthstone/frankenscipy/commit/935983e1f90fa37996f23b7db0820ab432b6273f))

### Spectral estimation

- Welch's method for power spectral density estimation.
  ([f957c7d](https://github.com/Dicklesworthstone/frankenscipy/commit/f957c7d5d22cdd73034d4b7052deaa9ef1883299))
- Windowed periodogram.
  ([65b3ce2](https://github.com/Dicklesworthstone/frankenscipy/commit/65b3ce2f9520064f6629dcefa1a29314ef007a26))

### Peak detection

- `find_peaks` peak detection.
  ([47b6a99](https://github.com/Dicklesworthstone/frankenscipy/commit/47b6a99e7b9ae9120d38d304a9913c0fbffe20f8))

### Bug fixes

- Fix unreachable early-return path in `tf2zpk` for all-zero numerator.
  ([8bd95b1](https://github.com/Dicklesworthstone/frankenscipy/commit/8bd95b12e64e4bbbfa49602a0d1db3301215b058))
- Clean up messy derivation comments in `group_delay`.
  ([f2325ad](https://github.com/Dicklesworthstone/frankenscipy/commit/f2325ad84d276b78ea01cf9e49bab0d8f147086b))
- Fix numerical stability in filter coefficient computation.
  ([d8180d9](https://github.com/Dicklesworthstone/frankenscipy/commit/d8180d9d5eb5c458f43fe80b70f0761a47f58908))
- Correct 5 bugs in filter conversions.
  ([a03f80d](https://github.com/Dicklesworthstone/frankenscipy/commit/a03f80dcc998f0c0013c321095d23846e85c6393))

---

## Spatial Algorithms (`fsci-spatial`)

New crate bootstrapped 2026-03-15.

### Nearest-neighbor search

- KDTree for nearest-neighbor queries.
  ([47b6a99](https://github.com/Dicklesworthstone/frankenscipy/commit/47b6a99e7b9ae9120d38d304a9913c0fbffe20f8))

### Distance computation

- `pdist` pairwise distance computation.
  ([f957c7d](https://github.com/Dicklesworthstone/frankenscipy/commit/f957c7d5d22cdd73034d4b7052deaa9ef1883299))

---

## Special Functions (`fsci-special`)

Scaffolded 2026-02-13; contract-first dispatch plans added 2026-03-02; implementations landed 2026-03-03 onward.

### Gamma family

- `gamma`, `lgamma`, `digamma`, `beta`, `betaln`.
  ([d6467d2](https://github.com/Dicklesworthstone/frankenscipy/commit/d6467d2f1fad59d37d06fbbdc2eff1f9d9d3ce87))
- Regularized incomplete gamma function.
  ([c92b075](https://github.com/Dicklesworthstone/frankenscipy/commit/c92b0758de912855c6e67c7c542b3679e228538e))
- Rewrite `gammaln` and add NaN propagation to all gamma convenience functions.
  ([4337552](https://github.com/Dicklesworthstone/frankenscipy/commit/4337552cb263c2e483f36278d96c6ed64823e06c))
- Combinatorial functions.
  ([8c20565](https://github.com/Dicklesworthstone/frankenscipy/commit/8c20565d76684f4ff0711e23e9b33b19c88341fb))

### Error functions

- `erf`, `erfc`, `erfinv`.
  ([d6467d2](https://github.com/Dicklesworthstone/frankenscipy/commit/d6467d2f1fad59d37d06fbbdc2eff1f9d9d3ce87))
- erf API cleanup.
  ([5a9f78a](https://github.com/Dicklesworthstone/frankenscipy/commit/5a9f78ab51cab8b88801c740a981e3c2ec7d400c))

### Bessel functions

- Integer-order Bessel functions: J0, J1, Jn, Y0, Y1, Yn.
  ([c92b075](https://github.com/Dicklesworthstone/frankenscipy/commit/c92b0758de912855c6e67c7c542b3679e228538e))
- Improve Bessel function accuracy and reorganize imports.
  ([3177df2](https://github.com/Dicklesworthstone/frankenscipy/commit/3177df2748f2011390ff30309f77b4abe403c0dd))

### Airy functions

- Airy functions (`airy`, `ai`, `bi`).
  ([8c20565](https://github.com/Dicklesworthstone/frankenscipy/commit/8c20565d76684f4ff0711e23e9b33b19c88341fb))
- Fix Airy function phase computation.
  ([01eb649](https://github.com/Dicklesworthstone/frankenscipy/commit/01eb64904284921c3a6789c21f376712c7c7e1a0))

### Hypergeometric functions

- `hyp1f1` and `hyp2f1` series evaluation.
  ([32d737c](https://github.com/Dicklesworthstone/frankenscipy/commit/32d737c4890a1879305f90451012f3585201861e))

### Elliptic functions

- Complete elliptic integrals (`ellipk`, `ellipe`) and Jacobi elliptic functions (`ellipj`).
  ([9ebb9c2](https://github.com/Dicklesworthstone/frankenscipy/commit/9ebb9c2d4aa02532ce93b82f7bb68f9869955dd1))

### Zeta function

- Riemann zeta function.
  ([f480b75](https://github.com/Dicklesworthstone/frankenscipy/commit/f480b752d6c70985a9fce1f0c97337f664be4047))

### Orthogonal polynomials

- Legendre, Hermite, Laguerre, Chebyshev, Jacobi polynomial evaluation.
  ([0210ac2](https://github.com/Dicklesworthstone/frankenscipy/commit/0210ac29f792a1fec599bc3370eb4bb4b6100a2a))

### NaN propagation

- NaN propagation for entropy functions.
  ([1614362](https://github.com/Dicklesworthstone/frankenscipy/commit/1614362889cc570c255b6d9e063a8398c1261c77))
- NaN propagation overhaul for all convenience functions.
  ([4337552](https://github.com/Dicklesworthstone/frankenscipy/commit/4337552cb263c2e483f36278d96c6ed64823e06c))

---

## Statistics (`fsci-stats`)

New crate bootstrapped 2026-03-14.

### Continuous distributions

- Normal, Uniform, Chi-Squared, Student-t at crate bootstrap.
  ([b140aeb](https://github.com/Dicklesworthstone/frankenscipy/commit/b140aeb14acd169421e412b6245067496fa82383))
- Exponential distribution.
  ([6d378cb](https://github.com/Dicklesworthstone/frankenscipy/commit/6d378cb6e964fe586f6232d92aa42848ac7704ac))
- Gamma and Poisson distributions.
  ([937daa3](https://github.com/Dicklesworthstone/frankenscipy/commit/937daa3b16611c8eb02e2e05eda7865d20ccbb43))
- Weibull and Lognormal distributions.
  ([47b6a99](https://github.com/Dicklesworthstone/frankenscipy/commit/47b6a99e7b9ae9120d38d304a9913c0fbffe20f8))
- F and Beta distributions.
  ([8a47f61](https://github.com/Dicklesworthstone/frankenscipy/commit/8a47f611571b42f574da016f661c0c1cf494e624))
- Trait-level `ppf` (percent point function) for all distributions.
  ([f99b570](https://github.com/Dicklesworthstone/frankenscipy/commit/f99b5709d168a41b0e754ca1a54962d55150cf0e))
- Survival function (`sf`) for Normal distribution.
  ([0b83dae](https://github.com/Dicklesworthstone/frankenscipy/commit/0b83daea069f06295cacb3fea6c9ee05170ab957))

### Discrete distributions

- Binomial, Bernoulli, Geometric, NegativeBinomial, Hypergeometric.
  ([c4568b0](https://github.com/Dicklesworthstone/frankenscipy/commit/c4568b01a83cd57324e8b2aac952079c5d06a2af))
- Fix two bugs in Hypergeometric distribution (PMF normalization, support range).
  ([aa3283b](https://github.com/Dicklesworthstone/frankenscipy/commit/aa3283b6ed35cecc92ff443719ecc69480b83485))

### Hypothesis testing

- t-test, chi-squared tests.
  ([42abbcc](https://github.com/Dicklesworthstone/frankenscipy/commit/42abbcc5d48cac81b0ff19b9c2a2b76a75515dbf))
- Non-parametric and ANOVA tests: `f_oneway`, `mannwhitneyu`, `wilcoxon`, `kruskal`, `ranksums`.
  ([5b48848](https://github.com/Dicklesworthstone/frankenscipy/commit/5b488482dc331a188f108d0ff677fe35c6fac15e))
- Goodness-of-fit tests (Kolmogorov-Smirnov, chi-squared, Shapiro-Wilk).
  ([7a1169e](https://github.com/Dicklesworthstone/frankenscipy/commit/7a1169ed185dd17d8804e2fd8b4331dbbcb2a0e3))

### Descriptive statistics and regression

- Descriptive statistics, correlation, and regression functions (+488 lines).
  ([004a3e9](https://github.com/Dicklesworthstone/frankenscipy/commit/004a3e90926522c6b1d523c8957e403804ac7fe8))
- `linregress` linear regression.
  ([f957c7d](https://github.com/Dicklesworthstone/frankenscipy/commit/f957c7d5d22cdd73034d4b7052deaa9ef1883299))
- Summary statistics and special-function convenience wrappers.
  ([23c5ff3](https://github.com/Dicklesworthstone/frankenscipy/commit/23c5ff3e19a15fd7f88c5be85fd7a895b1c60a6c))
- New statistical routines (entropy, additional edge-case coverage).
  ([1614362](https://github.com/Dicklesworthstone/frankenscipy/commit/1614362889cc570c255b6d9e063a8398c1261c77))

---

## Array API (`fsci-arrayapi`)

Scaffolded 2026-03-02; implemented 2026-03-03.

### Core backend

- Scaffold contract-first Array API foundation crate with module structure and type stubs.
  ([8e773d3](https://github.com/Dicklesworthstone/frankenscipy/commit/8e773d37b60c79536db72ed6283db5b0947ec779))
- `CoreArrayBackend` with broadcast, indexing, reduction, and type promotion.
  ([1b5700e](https://github.com/Dicklesworthstone/frankenscipy/commit/1b5700e83014c3bb809c0ae1137342d3a43f34e9))
- Validate `arange` inputs.
  ([52d0818](https://github.com/Dicklesworthstone/frankenscipy/commit/52d081846f940cb940067a8d096f80d562a93899))

### Performance

- Replace per-element unravel/ravel with incremental coordinate advancement in broadcast.
  ([2838ba6](https://github.com/Dicklesworthstone/frankenscipy/commit/2838ba6ae61cbec677c17700fd76fce002256b2c))

### Tests and benchmarks

- Comprehensive unit tests and proptest suites for all Array API modules.
  ([a2eabd2](https://github.com/Dicklesworthstone/frankenscipy/commit/a2eabd2e75885954948f02173aa40fee6e735bde))
- Criterion benchmark suite for core Array API operations.
  ([274f0ec](https://github.com/Dicklesworthstone/frankenscipy/commit/274f0ec8f4390d2545bbfc7e2e9044939e13aee6))

---

## Conformance Infrastructure (`fsci-conformance`)

Landed 2026-02-13 with FSCI-P2C-001 and FSCI-P2C-002; expanded continuously.

### Conformance packets

| Packet ID | Domain | Introduced | Key commit |
|---|---|---|---|
| FSCI-P2C-001 | Tolerance validation | 2026-02-13 | [55f1ee9](https://github.com/Dicklesworthstone/frankenscipy/commit/55f1ee94577f2b67e6242b93ac459998a47aa797) |
| FSCI-P2C-002 | Dense linear algebra | 2026-02-13 | [55f1ee9](https://github.com/Dicklesworthstone/frankenscipy/commit/55f1ee94577f2b67e6242b93ac459998a47aa797) |
| FSCI-P2C-003 | Optimization | 2026-03-03 | [43bb37e](https://github.com/Dicklesworthstone/frankenscipy/commit/43bb37e4859dc1c560e764bc8c21657bd2493a7d) |
| FSCI-P2C-004 | Sparse operations | 2026-03-13 | [72845e3](https://github.com/Dicklesworthstone/frankenscipy/commit/72845e30f09671ea8d104fa2c37eb358dbc6ce86) |
| FSCI-P2C-005 | FFT core | 2026-03-04 | [89a2f5c](https://github.com/Dicklesworthstone/frankenscipy/commit/89a2f5cc35837fe31ec68968cd5b21d593d1815e) |
| FSCI-P2C-006 | Special functions | 2026-03-03 | [1b5700e](https://github.com/Dicklesworthstone/frankenscipy/commit/1b5700e83014c3bb809c0ae1137342d3a43f34e9) |
| FSCI-P2C-007 | Array API | 2026-03-03 | [4f539b6](https://github.com/Dicklesworthstone/frankenscipy/commit/4f539b67251f69123f319f5ef0f7d1bcfd48d7ca) |
| FSCI-P2C-008 | Runtime CASP | 2026-03-13 | [72845e3](https://github.com/Dicklesworthstone/frankenscipy/commit/72845e30f09671ea8d104fa2c37eb358dbc6ce86) |

### Harness and infrastructure

- Differential conformance harness with artifact governance.
  ([9b921bd](https://github.com/Dicklesworthstone/frankenscipy/commit/9b921bd69c71748e24d8c9819964e7e1c744d5f7))
- RaptorQ + decode-proof artifact generation.
  ([55f1ee9](https://github.com/Dicklesworthstone/frankenscipy/commit/55f1ee94577f2b67e6242b93ac459998a47aa797))
- Fix RaptorQ sidecar generation to serialize from JSON instead of pre-serialization bytes.
  ([89fc93a](https://github.com/Dicklesworthstone/frankenscipy/commit/89fc93a2981e6ad18da1b81ddb0c298428f34f0b))
- Schema validation tests and runtime property tests.
  ([e6666dd](https://github.com/Dicklesworthstone/frankenscipy/commit/e6666dd05c019d74c74f84d16848e5b2f83531f4))

### E2E test suites

- E2E linear algebra conformance test suite.
  ([9ade3fb](https://github.com/Dicklesworthstone/frankenscipy/commit/9ade3fbb71a7879070cfbbdc2eda8799d41bd459))
- E2E scenario tests for FFT backend routing (FSCI-P2C-005).
  ([89a2f5c](https://github.com/Dicklesworthstone/frankenscipy/commit/89a2f5cc35837fe31ec68968cd5b21d593d1815e))
- E2E and evidence conformance test suites across all subsystems.
  ([0aac03a](https://github.com/Dicklesworthstone/frankenscipy/commit/0aac03a222226def07e8adc5f73d7c053eace2d5))
- P2C-002 evidence/perf/RaptorQ test suites and expanded benchmark coverage.
  ([b7d7614](https://github.com/Dicklesworthstone/frankenscipy/commit/b7d7614c7b9f7d79877129c37415f9a4953624cd))
- IVP solver performance profiling harness (P2C-001-H).
  ([8f09e95](https://github.com/Dicklesworthstone/frankenscipy/commit/8f09e95d99aa70731e302fb1734a001a14d16785))
- FFT differentiation tests and expanded DCT transform coverage.
  ([493b3be](https://github.com/Dicklesworthstone/frankenscipy/commit/493b3bea7200da15bf1bdf4bb0fe5421de3906bf))

### Quality gates and CI

- Conformance quality gates, CI pipeline, benchmarks, and forensic analysis tooling.
  ([f1e61c0](https://github.com/Dicklesworthstone/frankenscipy/commit/f1e61c0b3b71b67dbd0492a1013569762dd251f5))
- Expand E2E test coverage with new benchmarks and IVP integration API tests.
  ([72f8aa1](https://github.com/Dicklesworthstone/frankenscipy/commit/72f8aa1921c3416f59ca8149f9c780d5d0c35fca))
- Update test infrastructure, parity reports, and mutex handling for concurrent conformance runs.
  ([9a1c403](https://github.com/Dicklesworthstone/frankenscipy/commit/9a1c40338712c5c9e2c223e85c00e95ef23a3958))
- Regenerate P2C-001, P2C-002, P2C-007 evidence artifacts with corrected checksums.
  ([b276e59](https://github.com/Dicklesworthstone/frankenscipy/commit/b276e59f0a32dfbea580845758b78d04742c9721))

### Python oracle

- Python oracle capture script for SciPy-vs-Rust comparison.
  ([55f1ee9](https://github.com/Dicklesworthstone/frankenscipy/commit/55f1ee94577f2b67e6242b93ac459998a47aa797))

### Interactive dashboard

- `ftui` dashboard binary for artifact navigation.
  ([55f1ee9](https://github.com/Dicklesworthstone/frankenscipy/commit/55f1ee94577f2b67e6242b93ac459998a47aa797))

---

## Runtime and CASP (`fsci-runtime`)

Landed 2026-02-13.

- CASP runtime with execution-path tracing and schema validation.
  ([55f1ee9](https://github.com/Dicklesworthstone/frankenscipy/commit/55f1ee94577f2b67e6242b93ac459998a47aa797))
- Runtime property tests.
  ([e6666dd](https://github.com/Dicklesworthstone/frankenscipy/commit/e6666dd05c019d74c74f84d16848e5b2f83531f4))
- Expand runtime architecture with execution path tracing documentation.
  ([f78ada6](https://github.com/Dicklesworthstone/frankenscipy/commit/f78ada6f2596589e25cbacbb40ef41c3591f24e6))

---

## Cross-cutting numerical corrections

These commits span multiple modules and fix numerical issues that do not belong to a single domain.

- Improve numerical correctness across FFT, special functions, stats, and optimizer.
  ([1d9133d](https://github.com/Dicklesworthstone/frankenscipy/commit/1d9133d715ede43d9cf8d863fe1ceb9ae41eb531))
- Numerical corrections across FFT, signal, optimizer, and special functions.
  ([09c901c](https://github.com/Dicklesworthstone/frankenscipy/commit/09c901cdeebfa435b389a37b88008fcdfb9f5518))
- Correct 5 bugs found during code review across multiple crates.
  ([3411647](https://github.com/Dicklesworthstone/frankenscipy/commit/3411647f3f82ef05fb9c030fb131623ab72ed60f))
- Apply cargo fmt and resolve clippy warnings across workspace.
  ([335509f](https://github.com/Dicklesworthstone/frankenscipy/commit/335509ff4a3dd8abb7b0d44ab7440cec45dfc86a))
- ODE event handling overhaul, interpolation simplification, and multi-module improvements.
  ([3fc730e](https://github.com/Dicklesworthstone/frankenscipy/commit/3fc730e606529c4259f21522743b5c66cb1af8bb))
- Multi-module expansions across integrate, FFT, signal, sparse, spatial, and stats.
  ([60c5930](https://github.com/Dicklesworthstone/frankenscipy/commit/60c59301358783bd0c4d6aed6dabd30344685632))
- Large cross-module feature drop spanning stats, signal, linalg, optimization, interpolation, spatial, and special.
  ([9026a01](https://github.com/Dicklesworthstone/frankenscipy/commit/9026a01daa0d8936f2afc21b93919ba870136d09))

---

## Project infrastructure and dependencies

### License

- MIT license with OpenAI/Anthropic Rider.
  ([2888129](https://github.com/Dicklesworthstone/frankenscipy/commit/2888129fa7f3fe95183a84d487c06fc999490147))

### Branding

- GitHub social preview image (1280x640).
  ([47b8782](https://github.com/Dicklesworthstone/frankenscipy/commit/47b87822815fefe3a31521ba8eafce08efd3856d))

### Dependency management

- Switch `asupersync` and `ftui` from local path dependencies to crates.io.
  ([b7643af](https://github.com/Dicklesworthstone/frankenscipy/commit/b7643af85e6d317192fe99e7c7662e082d314f55),
   [9c1d6d8](https://github.com/Dicklesworthstone/frankenscipy/commit/9c1d6d805a5665e421fa1c5a171731aa3383d1b4))
- Pin `asupersync` to v0.2.0.
  ([1118492](https://github.com/Dicklesworthstone/frankenscipy/commit/111849231ccfb87a27c55052f415c6b0b310fac8))

### Documentation

- Project charter, comprehensive spec, porting plan, and architecture docs at initial commit.
  ([55f1ee9](https://github.com/Dicklesworthstone/frankenscipy/commit/55f1ee94577f2b67e6242b93ac459998a47aa797))
- Add cass (Cross-Agent Session Search) tool reference to AGENTS.md.
  ([c95e977](https://github.com/Dicklesworthstone/frankenscipy/commit/c95e9777607b448a0928aa0e4477b14375d1d1b7))

---

## Statistics

- **Total commits**: 152 (as of 2026-03-21)
- **Rust source**: ~66,000 lines across 13 workspace crates
- **Tests and benchmarks**: ~25,000 additional lines
- **First commit**: 2026-02-13 ([55f1ee9](https://github.com/Dicklesworthstone/frankenscipy/commit/55f1ee94577f2b67e6242b93ac459998a47aa797))
- **Crate creation timeline**:
  - 2026-02-13: 9 crates at initial commit (`fsci-linalg`, `fsci-sparse`, `fsci-integrate`, `fsci-opt`, `fsci-fft`, `fsci-special`, `fsci-arrayapi`, `fsci-conformance`, `fsci-runtime`)
  - 2026-03-14: `fsci-interpolate`, `fsci-stats`
  - 2026-03-15: `fsci-signal`, `fsci-spatial`
- **No tagged releases yet** -- workspace version is `0.1.0`
