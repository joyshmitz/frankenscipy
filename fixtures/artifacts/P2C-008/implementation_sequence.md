# P2C-008 Implementation Sequence: CASP Runtime Module Refinement

## Module Boundary Map

```
fsci-runtime/src/
├── lib.rs          # Crate root: re-exports, CASP portfolio, calibrator, test helpers
├── mode.rs         # RuntimeMode { Strict, Hardened }
├── signals.rs      # DecisionSignals, SignalSequence (adversarial replay)
├── evidence.rs     # PolicyEvidenceLedger, DecisionEvidenceEntry
└── policy.rs       # PolicyController, RiskState, PolicyAction, PolicyDecision,
                    #   logit model, softmax, loss matrices, action selection
```

## Public API Preservation

All existing public types remain accessible via `fsci_runtime::*`:

| Symbol | Source module | Re-exported from `lib.rs` |
|--------|-------------|---------------------------|
| `RuntimeMode` | `mode` | Yes |
| `DecisionSignals` | `signals` | Yes |
| `SignalSequence` | `signals` | Yes (new) |
| `DecisionEvidenceEntry` | `evidence` | Yes |
| `PolicyEvidenceLedger` | `evidence` | Yes |
| `PolicyController` | `policy` | Yes |
| `PolicyDecision` | `policy` | Yes |
| `PolicyAction` | `policy` | Yes |
| `RiskState` | `policy` | Yes |
| `MatrixConditionState` | `lib` | Direct |
| `SolverAction` | `lib` | Direct |
| `SolverEvidenceEntry` | `lib` | Direct |
| `SolverPortfolio` | `lib` | Direct |
| `ConformalCalibrator` | `lib` | Direct |

Zero breaking changes. All downstream crates (fsci-integrate, fsci-linalg, fsci-sparse, fsci-conformance) compile without modification.

## New Types Added

### `SignalSequence` (signals.rs)
- Ordered replay vector for adversarial/property testing
- Supports THREAT-003 (evidence poisoning) and THREAT-004 (NaN injection) test scenarios
- Fields: `id: String`, `signals: Vec<DecisionSignals>`, `expected_actions: Option<Vec<String>>`
- Methods: `new()`, `push()`, `len()`, `is_empty()`, `iter()`

### `DecisionSignals::is_finite()` (signals.rs)
- Returns `true` if all three signal components are finite
- Enables THREAT-004 NaN validation without breaking existing construction

## Implementation Order

### Phase 1: Module Extraction (completed)
1. Extract `RuntimeMode` to `mode.rs`
2. Extract `DecisionSignals` to `signals.rs`, add `SignalSequence` and `is_finite()`
3. Extract `PolicyEvidenceLedger` + `DecisionEvidenceEntry` to `evidence.rs`
4. Extract `PolicyController` + risk model to `policy.rs`
5. Update `lib.rs` with module declarations and re-exports
6. Verify: `cargo check`, `cargo test`, `cargo clippy` pass
7. Verify: all downstream crates compile

### Phase 2: Conformance Harness Wiring (bd-3jh.19.10)
1. Wire behavioral anchors from `behavior_ledger.json` into test fixtures
2. Create conformance harness entries for each contract in `contract_table.json`
3. Implement golden-value tests for each behavioral anchor pathway (N1, N2, E1, E2, A1-A4)

### Phase 3: Unit + Property Tests (bd-3jh.19.5)
1. Property tests for posterior normalization invariant
2. Property tests for deterministic decision invariant
3. Adversarial replay tests using `SignalSequence`
4. NaN/Inf signal rejection tests using `is_finite()`
5. Loss matrix tie-breaking property tests
6. Calibrator bounded-window property tests

### Phase 4: Hardened Mode Enhancements (future)
1. Signal finiteness validation in Hardened mode `decide()`
2. Ledger integrity hash chain
3. VecDeque migration for SolverPortfolio evidence
4. Calibrator anomaly detection

## Dependency Graph

```
bd-3jh.19.4 (this bead: module boundaries)
    ├── bd-3jh.19.10 (conformance harness corpus wiring)
    └── bd-3jh.19.5 (unit + property tests)
```
