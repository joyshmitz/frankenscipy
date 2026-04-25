# Conformance Coverage Matrix

> **Score ≥ 0.95 for MUST-level functions = conformant.**
> Coverage below this threshold on any family ships with a visible gap
> in DISCREPANCIES.md, never as an unknown gap.

This file tracks what is *tested* vs what is *implemented* vs what
exists in upstream SciPy. The "oracle-backed" column counts only
cases whose expected values were produced by running the Python
scipy oracle (scipy_<family>_oracle.py) — not by hand-populating the
fixture.

## Summary

| Family | SciPy surface | fsci impl | Fixture cases | Oracle-backed |
|--------|-------------:|----------:|-------------:|-------------:|
| linalg     | ~180 | ~60 | 53 | 53 (stats pattern) |
| stats      | ~400 | ~80 | 30 | 30 |
| special    | ~120 | ~45 | 257 | 257 |
| fft        | ~40  | ~25 | 56 | 56 |
| optimize   | ~60  | ~25 | 49 | 49 |
| sparse     | ~200 | ~40 | 27 | 27 |
| integrate  | ~30  | ~15 | 28 | 28 |
| cluster    | ~25  | ~12 | 20 | 20 |
| spatial    | ~70  | ~25 | 26 | 26 |
| signal     | ~200 | ~40 | 12 | 12 |
| arrayapi   | N/A  | ~60 | 34 | 34 |
| constants  | ~140 | ~50 | 24 | 24 |
| interpolate| ~40  | ~15 | 13 | 13 |
| ndimage    | ~100 | ~30 | 6  | 6 |
| io         | ~15  | ~5  | 7  | 7 |

_Numbers are best-effort rough counts. Refine by running the coverage
generator: `cargo run -p fsci-conformance --bin coverage_report` (TBD)._

## Per-family detail

### linalg (P2C-002)

- **Oracle script:** `python_oracle/scipy_linalg_oracle.py`
- **Fixture:** `fixtures/FSCI-P2C-002_linalg_core.json`
- Covers: solve / inv / det / lstsq / pinv / solve_triangular /
  solve_banded / qr / svd / cholesky / eig / eigh
- **Remaining gaps:** decomposition fixture cases currently compare stable
  scalar/matrix surfaces only (QR R, SVD singular values, Cholesky factor,
  eig/eigh eigenvalues). Sign-normalized full eigenvector and singular-vector
  comparisons are still unmapped.

### optimize (P2C-003)

- **Oracle script:** `python_oracle/scipy_optimize_oracle.py`
- **Fixture:** `fixtures/FSCI-P2C-003_optimize_core.json`
- Covers: local minimizers over Rosenbrock/Ackley/Rastrigin benchmark
  objectives, root scalar methods, seeded global optimizers
  (`differential_evolution`, `basinhopping`, `dual_annealing`, `brute`),
  metamorphic invariants, and adversarial fail-closed cases.
- **E2E harness:** `tests/e2e_optimize.rs`
- **Named parity gate:** `p2c003_optimize_global_local_parity`

### special (P2C-006)

- **Oracle script:** `python_oracle/scipy_special_oracle.py`
- **Fixture:** `fixtures/FSCI-P2C-006_special_core.json`
- Covers: gamma / beta / erf family / hyp family / Bessel family.
- **Notes:** 7 Hardened-mode cases exercise pole / domain / overflow
  rejection. See DISCREPANCIES for error-kind mapping.

### constants (P2C-016)

- **Oracle script:** `python_oracle/scipy_constants_oracle.py`
- **Fixture:** `fixtures/FSCI-P2C-016_constants_core.json`
- **E2E harness:** `tests/e2e_constants.rs`
- Covers: SI-exact constants, derived constants, temperature conversion,
  energy conversion, wavelength/frequency conversion, angle conversion,
  mass conversion, and fail-closed unknown-name/unknown-scale paths.

### interpolate (P2C-014)

- **Oracle script:** `python_oracle/scipy_interpolate_oracle.py`
- **Fixture:** `fixtures/FSCI-P2C-014_interpolate_core.json`
- **E2E harness:** `tests/e2e_interpolate.rs`
- Covers: `interp1d` linear/nearest evaluation, fill-value behavior,
  reject paths, `RegularGridInterpolator` linear/nearest evaluation,
  `CubicSpline`, and `BSpline`.

### io (P2C-017)

- **Oracle script:** `python_oracle/scipy_io_oracle.py`
- **Fixture:** `fixtures/FSCI-P2C-017_io_core.json`
- **E2E harness:** `tests/e2e_io.rs`
- Covers: Matrix Market dense and symmetric coordinate reads, Matrix Market
  dense round-trips, `loadtxt`/`savetxt` numeric text round-trips, WAV
  16-bit PCM metadata/value round-trip, and ragged-row fail-closed behavior.

### Remaining families

Each of the newer oracles added under br-di9p (cluster / spatial /
signal / integrate / arrayapi / constants) covers the same functions
the fixture case set exercises — extending either surface requires
adding to both the oracle dispatcher and the fixture.

## What's NOT tested

- Hardened-mode coverage is 7.3% across all fixtures (53/729 cases).
  The former zero-coverage families tracked by frankenscipy-xo4o and
  frankenscipy-ncst now each have at least 5 Hardened cases; remaining
  work is raising per-family Hardened floors beyond the seed coverage.
- Stochastic optimize methods (`differential_evolution`, `basinhopping`,
  `dual_annealing`, `brute`) have seeded fixture coverage in P2C-003.
  Stochastic clustering methods (kmeans / dbscan) remain tracked separately.
- io now has a seed P2C fixture packet, but the surface remains narrow.
  Extend it before reporting broad parity percentages for that family.
- ndimage has a P2C packet and SciPy oracle capture lane for core
  filtering, morphology, labeling, and distance-transform smoke
  coverage.

## Update procedure

When adding a new family or expanding coverage:

1. Update the Summary table — count by running
   `jq '.cases | length' fixtures/FSCI-*.json`.
2. Add per-family detail if meaningful.
3. Cross-link to the bead that drove the addition.
4. Commit with `chore(conformance): COVERAGE.md ...`.
