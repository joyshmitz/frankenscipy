# FSCI-P2C-004 Sparse Implementation Sequence

## Scope
- Packet: `FSCI-P2C-004`
- Crate: `fsci-sparse`
- Contracts source: `crates/fsci-conformance/fixtures/artifacts/P2C-004/contracts/contract_table.json`
- Threat source: `crates/fsci-conformance/fixtures/artifacts/P2C-004/threats/threat_matrix.json`

## Module Boundaries
- `formats.rs`
  - `CsrMatrix`, `CscMatrix`, `CooMatrix`
  - invariant-enforcing constructors for compressed/triplet forms
  - canonical metadata (`sorted_indices`, `deduplicated`)
  - nalgebra sparse bridge (`NalgebraBridge`)
- `ops.rs`
  - format conversion trait (`FormatConvertible`)
  - deterministic duplicate merge and canonical compression paths
- `construct.rs`
  - constructor surface (`eye`, `diags`, `random`)
  - explicit offset/shape validation for diagonal constructors
- `linalg.rs`
  - solver/factorization API contracts (`spsolve`, `splu`, `spilu`)
  - backend/policy option types and hardened prechecks

## Sequence (Risk-First)
1. `P2C-004-D1`: finalize format invariants and malformed-metadata rejection behavior.
2. `P2C-004-D2`: implement deterministic COO<->CSR/CSC conversion kernels and canonicalization controls.
3. `P2C-004-D3`: wire constructor APIs (`eye`, `diags`, `random`) to conversion primitives with strict/hardened guardrails.
4. `P2C-004-D4`: implement backend abstraction layer for `spsolve`, `splu`, `spilu` (SuperLU/UMFPACK policy routing).
5. `P2C-004-D5`: integrate CASP runtime signals from `fsci-runtime` into sparse solve routing and evidence emission.
6. `P2C-004-D6`: add conformance fixtures + differential oracle hooks for sparse packet operations.
7. `P2C-004-D7`: complete unit/property/adversarial test suites and threat-specific regression fixtures.
8. `P2C-004-D8`: baseline sparse performance (p50/p95/p99 + memory), then apply one optimization lever per iteration.

## Invariant Checklist (Construction-Time)
- `len(data) == len(indices)` for compressed formats.
- `len(indptr) == major_dim + 1`.
- `indptr[0] == 0` and `indptr[last] == nnz`.
- `indptr` monotone non-decreasing.
- every minor index in bounds.
- COO triplet lengths aligned (`data`, `row`, `col`).
- duplicate semantics deterministic under canonicalization.

## Integration Seams
- nalgebra sparse seam:
  - `to_nalgebra_cs` bridge for all formats.
  - `from_nalgebra_cs` bridge for roundtrip validation and backend interop.
- runtime policy seam:
  - `RuntimeMode` in solve/factorization options.
  - hardened fail-closed checks before backend dispatch.

## Evidence Plan
- Differential conformance report for every newly implemented function family.
- Threat regression artifacts for all `P2C-004` threat IDs.
- Benchmark delta artifacts for each optimization pass.
- Risk-note updates when compatibility envelope or backend policy changes.
