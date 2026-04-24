# Conformance Fixtures

This folder stores normalized oracle-vs-target fixtures for fsci-conformance.

- `smoke_case.json`: minimal bootstrap fixture ensuring harness wiring works.
- `FSCI-P2C-001_validate_tol.json`: packet-level conformance cases for
  `scipy.integrate._ivp.common.validate_tol` parity.
- `FSCI-P2C-002_linalg_core.json`: first-wave dense linalg conformance cases for:
  - `solve`
  - `solve_triangular`
  - `solve_banded`
  - `inv`
  - `det`
  - `lstsq`
  - `pinv`
- `FSCI-P2C-003_optimize_core.json`: optimize/root differential + metamorphic +
  adversarial conformance cases for:
  - `minimize` (`BFGS`, `CG`, `Powell`) on Rosenbrock/Ackley/Rastrigin + invariance quadratics
  - `root_scalar` (`brentq`, `brenth`, `bisect`, `ridder`) with scalar root contracts
- `FSCI-P2C-014_interpolate_core.json`: interpolate differential + edge +
  adversarial conformance cases for:
  - `interp1d` (`linear`, `nearest`) vector evaluation, fill-value, and reject paths
  - `RegularGridInterpolator` (`linear`, `nearest`) grid evaluation and fill-value behavior
  - `CubicSpline` and `BSpline` spline evaluation packets
- `FSCI-P2C-016_constants_core.json`: constants/core helper evidence fixture for:
  - exact fundamental values
  - name/alias lookup via `value` and `find`
  - temperature, energy, wavelength/frequency, angle, and mass conversion invariants
- `artifacts/`: generated parity reports, RaptorQ sidecars, and decode-proof logs.

Expected per-packet durable artifacts:

- `parity_report.json`
- `parity_report.raptorq.json`
- `parity_report.decode_proof.json`

Optional oracle artifacts:

- `oracle_capture.json` when SciPy capture succeeds.
- `oracle_capture.error.txt` when oracle capture is configured as optional and fails.

## E2E Scenarios

Packet-aware E2E scenarios are discovered from:

- `artifacts/*/e2e/scenarios/*.json`

Each run writes forensic logs to:

- `artifacts/FSCI-P2C-*/e2e/runs/{run_id}/{scenario_id}/events.jsonl`
- `artifacts/FSCI-P2C-*/e2e/runs/{run_id}/{scenario_id}/summary.json`
- `artifacts/P2C-003/e2e/runs/*.json` (optimize E2E forensic bundles emitted by
  `tests/e2e_optimize.rs`)
- `artifacts/FSCI-P2C-014/e2e/runs/standalone/{scenario_id}/summary.json`
  (interpolate forensic bundles emitted by `tests/e2e_interpolate.rs`)

Run the orchestrator:

```bash
cargo run -p fsci-conformance --bin e2e_orchestrator -- --packet FSCI-P2C-001
```
