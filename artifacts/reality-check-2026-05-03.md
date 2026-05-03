# Reality Check — 2026-05-03

**Vision (from README + COMPREHENSIVE_SPEC_FOR_FRANKENSCIPY_V1):** clean-room Rust reimplementation of SciPy with strict-vs-hardened mode separation, Condition-Aware Solver Portfolio (CASP), RaptorQ-backed durability, and conformance-test parity for scoped V1 surface (linalg / sparse / opt / stats / signal families).

**Today's working surface (counted from `crates/fsci-*/src/`):**

| Crate            | Public fns | Notes                                                       |
|------------------|-----------:|-------------------------------------------------------------|
| fsci-special     | 349        | gamma/Bessel/elliptic/orthopoly/Kelvin/Struve/zeta/lambertw |
| fsci-stats       | 211 (+24 distributions) | pearson/spearman/kruskal + 24 dists + qmc gaps  |
| fsci-signal      | 172        | windows, filters, wavelets, IIR/FIR, hilbert                |
| fsci-linalg      | 137        | dense + matrix functions + structured (toeplitz/hadamard)   |
| fsci-sparse      | 99         | formats + 13 iterative solvers + graph algorithms           |
| fsci-ndimage     | 89         | filters + morphology + interpolation                        |
| fsci-opt         | 68         | scalar + multivariate + global + constraint solvers         |
| fsci-spatial     | 63         | KDTree, ConvexHull, Voronoi, Rotation, distances            |
| fsci-interpolate | 58         | Interp1d, Pchip, Akima, RBF, BSpline, Krogh                 |
| fsci-fft         | 52         | fft/rfft/dct/dst/fht with multi-D and normalisation         |
| fsci-integrate   | 47         | quad family, solve_ivp, BVP, Monte Carlo                    |
| fsci-cluster     | 32         | kmeans/dbscan/hierarchical/silhouette                       |
| fsci-io          | 19         | mm, wav, mat, txt, csv, json, npy_text                      |
| fsci-arrayapi    | (backend)  | Array API spec compatibility shim                           |
| fsci-conformance | (binary)   | RaptorQ sidecar artifacts FSCI-P2C-001 .. P2C-018           |
| fsci-constants   | (data)     | physical constants                                          |
| fsci-runtime     | (mode)     | Strict / Hardened runtime mode plumbing                     |

**Metamorphic test coverage:** ~660 oracle-free relations across 12 crates (added in the past session); all green on rch.

**Bead state at scan time:** 0 open / 1 in-progress (LOW priority naming nit) / closed everything else. The implementation backlog is empty — gaps are now strategic, not tactical.

---

## Vision Checklist

| #  | Goal                                                              | Source                          | Status         | Evidence |
|----|-------------------------------------------------------------------|---------------------------------|----------------|----------|
| 1  | Scoped linalg parity (V1)                                          | README §V1 Scope                | WORKING        | 137 fns; FEATURE_PARITY=114% |
| 2  | Scoped sparse parity (V1)                                          | README §V1 Scope                | PARTIAL        | 99 fns; FEATURE_PARITY=45% |
| 3  | Scoped opt parity (V1)                                             | README §V1 Scope                | PARTIAL        | 68 fns; FEATURE_PARITY=55% |
| 4  | Scoped stats parity (V1)                                           | README §V1 Scope                | WORKING        | 211 fns + 24 dists; FEATURE_PARITY=88% |
| 5  | Scoped signal parity (V1)                                          | README §V1 Scope                | WORKING        | 172 fns; FEATURE_PARITY=86% |
| 6  | Conformance parity reports (Tier B)                                | SPEC §8                         | PARTIAL        | 18 P2C packets exist; FEATURE_PARITY shows all `parity_gap` |
| 7  | RaptorQ sidecars on long-lived artifacts                           | SPEC §9                         | PARTIAL        | sidecars on conformance only; no benchmark/migration/release sidecars |
| 8  | CASP — runtime algorithm selection                                 | README "core identity constraint" | PARTIAL      | Only `solve_with_casp`/`inv_with_casp`/`lstsq_with_casp`/`pinv_with_casp`. Not extended to opt/sparse/special/stats |
| 9  | Strict / Hardened mode separation                                  | SPEC §4                         | WORKING        | RuntimeMode plumbed across all crates |
| 10 | Adversarial fixtures + fuzz suites at high-risk entry points       | SPEC §5                         | WORKING        | 53 fuzz harnesses in `fuzz/fuzz_targets/` |
| 11 | Calibrated decision evidence ledger                                | SPEC §6                         | PARTIAL        | `audit::sync_audit_ledger` exists; loss-matrix/posterior model not surfaced |
| 12 | Profile-first optimization with proof                              | SPEC §7                         | UNPROVEN       | Criterion benches present (`benches/`); no published baseline / regression gates |
| 13 | scipy.integrate, scipy.interpolate, scipy.fft, scipy.spatial parity | implicit (broader port)        | WORKING        | Coverages 78–90% per FEATURE_PARITY |
| 14 | scipy.ndimage, scipy.cluster                                       | implicit                        | WORKING        | Both crates exist; well-tested; broader parity unmeasured |
| 15 | scipy.io                                                           | implicit                        | PARTIAL        | mm/wav/mat/csv only — netcdf/fortran/harwell-boeing/idl/arff missing |
| 16 | scipy.special parity                                               | implicit                        | PARTIAL        | 349 fns; FEATURE_PARITY=47% |
| 17 | scipy.datasets (test datasets)                                     | upstream module                 | NO_BEAD        | No fsci-datasets crate |
| 18 | scipy.differentiate (numerical differentiation)                    | upstream module (1.15+)         | NO_BEAD        | No top-level numerical diff API |
| 19 | scipy.odr (orthogonal distance regression)                         | upstream module                 | NO_BEAD        | No fsci-odr crate / no equivalent fns |
| 20 | SciPy-present CI lane for real oracle capture                      | README "Next Steps #2"          | NOT_STARTED    | Only baked oracle JSONs; no live capture pipeline |
| 21 | Benchmark baselines + profile artifacts                            | README "Next Steps #3"          | NOT_STARTED    | Bench infra exists; no published numbers / regression gates |
| 22 | Dashboard mismatch drill-down + numeric deltas                     | README "Next Steps #4"          | NOT_STARTED    | `ftui` exists; numeric-delta drill-down absent |
| 23 | M3 milestone (scope expansion exit criteria)                       | SPEC §10                        | PARTIAL        | "no unresolved critical compatibility defects" — 18 P2C packets all `parity_gap` |
| 24 | M4 hardening exit criteria (zero conformance drift)                | SPEC §10                        | NOT_STARTED    | Drift detection scaffold present; gates not enforced |

---

## Significant Gaps

### A. Entire scipy submodules missing (NO_BEAD)

1. **scipy.datasets** — small built-in datasets (`ascent`, `face`, `electrocardiogram`). Used pervasively in upstream tests. No `fsci-datasets` crate exists.
2. **scipy.differentiate** — modern (≥1.15) numerical differentiation API: `derivative`, `jacobian`, `hessian`. Distinct from the per-crate `numerical_gradient` shims. No top-level fsci surface.
3. **scipy.odr** — orthogonal distance regression. Upstream has `Model`, `Data`, `RealData`, `ODR`, `Output`. No fsci equivalent.

### B. Within-crate parity gaps documented in FEATURE_PARITY

4. **fsci-sparse 45%** — 50+ fns implemented but ~55% of upstream `scipy.sparse` surface still missing: `sparray`/`spmatrix` distinction, broader specialized ops, missing decompositions (`splu` for non-square), CSR / BSR conversions edge cases.
5. **fsci-opt 55%** — `minimize` methods `trust-exact` and Newton-CG hessian product. Constrained `differential_evolution` integration. `hybr` root finder.
6. **fsci-special 47%** — `scipy.special` exposes 319 symbols; 151 implemented. Missing: many incomplete Bessel zeros, full `hyp2f1` transformations (also tracked as bug `frankenscipy-rd28n`), `cython_special` cdef API, exotic Mathieu / spheroidal / parabolic cylinder functions, less-used elliptic integrals, `softplus`, `spence` (Spence's dilogarithm), `wofz` complete.

### C. CASP not extended beyond linalg

7. **CASP for opt** — runtime selector across solver portfolio (BFGS / L-BFGS-B / Newton-CG / trust-region) based on problem conditioning + constraint structure.
8. **CASP for sparse iterative solvers** — currently caller picks `cg` / `gmres` / `lgmres` etc by hand. CASP should pick by symmetry + spectrum diagnostics + preconditioner availability.
9. **CASP for special function evaluation** — series vs asymptotic vs continued-fraction selection driven by argument magnitude + precision target. Currently each special function hardcodes one branch (root cause of `frankenscipy-rd28n` and `frankenscipy-9u8kz`).

### D. Operational gaps

10. **No live SciPy oracle capture pipeline** — README Next Steps #2. Currently parity reports use baked-in JSON snapshots; no automated re-capture against upstream `scipy` import.
11. **Benchmark baselines + regression gates** — README Next Steps #3. Criterion benches exist but `target/criterion/` baselines are not committed/published; no CI gate.
12. **RaptorQ scope** — sidecars only on `crates/fsci-conformance/fixtures/artifacts/FSCI-P2C-*`. Not extended to: benchmark baselines, migration manifests, release-grade state, decision-ledger snapshots.
13. **Calibrated decision-evidence ledger** — `audit::sync_audit_ledger` exists but emits opaque entries; SPEC §6 requires explicit state space + evidence signals + loss matrix + posterior model + action rule. Currently reduced to "log when fallback triggered."

### E. Test-coverage soft spots

14. **Metamorphic suite gaps** — fsci-arrayapi / fsci-constants / fsci-runtime / fsci-conformance / fsci-io have no metamorphic suite. The 12 numerical crates have ~660 relations but coverage of CASP behaviour itself (the crown-jewel innovation) is implicit at best.
15. **Mismatch drill-down in `ftui`** — README Next Steps #4. Dashboard shows pass/fail; does not surface numeric deltas / oracle-vs-target side-by-side / sidecar provenance.

### F. Conformance milestone gap

16. **All 18 P2C packets at `parity_gap`** — implementations exist; conformance assertion under explicit tolerance policy is not green for any packet. M3 exit criterion ("expanded parity reports green") is unmet.

---

## Open Defects (already filed, included for completeness)

- `frankenscipy-rd28n` [HIGH] hyp2f1 silently returns divergent partial sum for `|z| > 1` in Strict mode.
- `frankenscipy-9u8kz` [HIGH] hyp1f1 silently returns partial sum on non-convergence.
- `frankenscipy-cbgqx` [MEDIUM] zeta_scalar 1e-4 imprecision vs π²/6 (naive truncated sum).
- `frankenscipy-o8x6j` [LOW] magnitude_response returns dB but name implies linear magnitude.

---

## Honesty Notes

- FEATURE_PARITY claims "fsci-fft 80% (22 fns)" but actual public surface is 52 fns plus DCT/DST variants — the claimed coverage is a low-water mark, not high-water. Same pattern for fsci-opt (claimed 45+, actual 68) and fsci-stats (claimed 136 fns, actual 211).
- "WORKING" status (e.g., scoped linalg) means the function exists, has tests, and survives metamorphic relations — *not* that it has been bitwise-checked against scipy.linalg under a concrete tolerance policy. That comparison is the unfinished P2C packet conformance work.
- The Condition-Aware Solver Portfolio (the "crown-jewel innovation" in SPEC §0) is implemented for exactly four linalg routines. Calling CASP "the core identity constraint" while only 4/750+ functions exercise it is a vision-vs-reality gap that should be confronted.
- Performance-and-proof discipline (SPEC §7) requires a profile→optimization→correctness-evidence loop. We have benches, we have clippy/test gates, but the published baseline-and-regression-gate pipeline does not exist yet.
