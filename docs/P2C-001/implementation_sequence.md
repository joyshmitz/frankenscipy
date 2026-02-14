# P2C-001 Implementation Sequence

## Module Tree

```text
fsci-integrate
└── src
    ├── lib.rs              # public module graph + re-exports
    ├── validation.rs       # validate_tol / validate_first_step / validate_max_step
    ├── step_size.rs        # select_initial_step skeleton + request model
    ├── solver.rs           # OdeSolver trait + step lifecycle outcome types
    └── api.rs              # solve_ivp options/result surface
```

## Contract-To-Signature Mapping

| Contract entry | Rust signature (single owner) | Module |
| --- | --- | --- |
| `fsci_integrate::validate_tol` | `pub fn validate_tol(rtol: ToleranceValue, atol: ToleranceValue, n: usize, mode: RuntimeMode) -> Result<ValidatedTolerance, IntegrateValidationError>` | `validation.rs` |
| `fsci_integrate::validate_first_step` | `pub fn validate_first_step(first_step: f64, t0: f64, t_bound: f64) -> Result<f64, IntegrateValidationError>` | `validation.rs` |
| `fsci_integrate::validate_max_step` | `pub fn validate_max_step(max_step: f64) -> Result<f64, IntegrateValidationError>` | `validation.rs` |
| `fsci_integrate::select_initial_step` | `pub fn select_initial_step<F>(fun: &mut F, request: &InitialStepRequest<'_>) -> Result<f64, IntegrateValidationError> where F: FnMut(f64, &[f64]) -> Vec<f64>` | `step_size.rs` |
| `fsci_integrate::OdeSolver::step` | `fn step(&mut self) -> Result<StepOutcome, StepFailure>` | `solver.rs` |
| `fsci_integrate::solve_ivp` | `pub fn solve_ivp<F>(fun: &mut F, options: &SolveIvpOptions<'_>) -> Result<SolveIvpResult, IntegrateValidationError> where F: FnMut(f64, &[f64]) -> Vec<f64>` | `api.rs` |

## Dependency Discipline

1. `validation.rs` is leaf logic and has no internal crate dependencies.
2. `step_size.rs` depends on validation error/tolerance types for shared contracts.
3. `solver.rs` is trait-only boundary and does not depend on API orchestration.
4. `api.rs` consumes validation contracts but does not depend on concrete solver implementations.
5. `lib.rs` is a re-export layer only.

This keeps module dependencies acyclic.

## Implementation Order (Risk-Minimizing)

1. Lock validation parity surface (`validate_tol`, `validate_first_step`, `validate_max_step`) with existing tests.
2. Implement `select_initial_step` internals in `step_size.rs` with one deterministic call-site for RHS evaluation.
3. Add concrete solver structs implementing `OdeSolver` (`RK23`, `RK45`, `DOP853`, `Radau`, `BDF`, `LSODA`).
4. Implement `solve_ivp` dispatch and stepping loop in `api.rs`.
5. Add event and dense-output plumbing to `solve_ivp` while preserving strict-mode parity.
6. Add hardened-mode guardrails and policy logs using `RuntimeMode`.
7. Extend conformance/property/E2E coverage with packet-level fixture artifacts.

## fsci-runtime Integration Points

- `RuntimeMode` is threaded through `ValidatedTolerance`, `InitialStepRequest`, and `SolveIvpOptions`.
- Strict mode remains compatibility-first; hardened mode is reserved for bounded defensive behavior.
- Current skeleton marks non-trivial routines (`select_initial_step`, `solve_ivp`) as explicit `NotYetImplemented` errors to avoid silent behavior drift before conformance proofs land.
