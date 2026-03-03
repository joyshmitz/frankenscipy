# fsci-arrayapi Implementation Sequence (P2C-007-D)

This file is the packet-D execution map for `fsci-arrayapi`.

## Module Boundaries

1. `backend.rs`
Defines the `ArrayApiBackend` trait surface consumed by all module layers:
- conversion (`asarray`)
- creation (`zeros/ones/empty/full/arange/linspace`)
- indexing (`getitem`)
- shape and dtype introspection (`shape_of`, `dtype_of`)
- broadcast and cast operations (`broadcast_to`, `astype`, `result_type`)

2. `creation.rs`
Contract-aligned wrappers and request structs for:
- `zeros`, `ones`, `empty`, `full`, `arange`, `linspace`
- shape overflow checks and request-shape normalization seam

3. `indexing.rs`
Boundary contract for:
- basic slicing
- advanced integer indexing
- boolean-mask indexing

4. `broadcast.rs`
Shape engine and normalization seam for:
- `broadcast_shapes`
- `promote_and_broadcast`

5. `dtype.rs`
Promotion and mode-aware defaults:
- `result_type`
- `default_float_dtype`

6. `integration.rs`
Cross-crate integration seams plus explicit nalgebra `DMatrix` handoff points.

7. `types.rs` + `error.rs`
Shared domain types, indexing primitives, shape metadata, and error taxonomy used by all modules.

## Contract Alignment Inputs

- Anchor ledger: `crates/fsci-conformance/fixtures/artifacts/P2C-007/anchor/behavior_ledger.json`
- Contract table: `crates/fsci-conformance/fixtures/artifacts/P2C-007/contracts/contract_table.json`
- Threat model: `crates/fsci-conformance/fixtures/artifacts/P2C-007/threats/threat_matrix.json`

## Strict vs Hardened Seams

- Strict mode:
  - preserve SciPy-observable namespace, indexing, copy/order, and promotion outcomes.
  - no behavior-altering repairs.

- Hardened mode:
  - preflight validation for malformed shapes, indices, and unsafe promotion classes.
  - fail closed on ambiguous namespace/protocol inputs.
  - keep strict outward results for valid inputs.

## nalgebra DMatrix Integration Points

1. `fsci-linalg` solve/decompose boundary:
   - map validated contiguous 2D float arrays to `nalgebra::DMatrix<f64>`.
2. `fsci-opt` Jacobian/Hessian workspaces:
   - preserve broadcasted parameter shapes before `DMatrix` materialization.
3. runtime diagnostics:
   - preserve shape/dtype provenance across conversions for deterministic audit logs.

## D -> D2 Handoff Checklist

1. Provide first concrete backend implementing `ArrayApiBackend`.
2. Replace placeholder wrapper paths with backend-specific coercion and indexing kernels.
3. Add module-level tests that bind every P2C-007 contract row to executable checks.
4. Wire structured logs for malformed-input and fail-closed policy triggers.
5. Run packet quality gates before opening `bd-3jh.18.5` and `bd-3jh.18.10`.
