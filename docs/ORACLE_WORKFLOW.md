# Oracle Regeneration Workflow

This guide documents how to regenerate SciPy oracle captures, update fixtures, and verify
provenance. It is the canonical workflow for the "SciPy-present" CI lane described in
README Next Steps.

## Prerequisites

- A working Python 3 installation with NumPy + SciPy.
- Legacy SciPy oracle checkout at:
  - `/dp/frankenscipy/legacy_scipy_code/scipy`
- Reference environment (per `docs/TEST_CONVENTIONS.md`):
  - `.venv-py314`, SciPy 1.17.0, Python 3.14.2.

Suggested local setup (optional but recommended):

```bash
python3 -m venv .venv-py314
source .venv-py314/bin/activate
pip install --upgrade pip wheel
pip install numpy scipy
```

## Preferred Path: Rust Harness Capture (P2C-002)

The linalg harness owns the canonical capture pipeline and provenance tracking.

```bash
cargo test -p fsci-conformance --test evidence_p2c002 oracle_capture_p2c002_linalg -- --nocapture
```

This test:
- calls `capture_linalg_oracle(...)`
- writes `oracle_capture.json` to `crates/fsci-conformance/fixtures/artifacts/FSCI-P2C-002/`
- records provenance hashes and runtime info

If SciPy is missing, the test prints a skip note and exits without failure.

## Direct Script Capture (Manual)

If you need a one-off capture without invoking the Rust harness:

```bash
python3 crates/fsci-conformance/python_oracle/scipy_linalg_oracle.py \
  --fixture crates/fsci-conformance/fixtures/FSCI-P2C-002_linalg_core.json \
  --output crates/fsci-conformance/fixtures/artifacts/FSCI-P2C-002/oracle_capture.json \
  --oracle-root /dp/frankenscipy/legacy_scipy_code/scipy
```

Other oracle scripts (manual capture only, for now):
- `python_oracle/scipy_fft_oracle.py`
- `python_oracle/scipy_optimize_oracle.py`
- `python_oracle/scipy_special_oracle.py`

## Regenerate Fixtures from Oracle Capture

Use the fixture regeneration tool to update expected values and add provenance:

```bash
cargo run -p fsci-conformance --bin fixture_regen -- \
  --fixture crates/fsci-conformance/fixtures/FSCI-P2C-002_linalg_core.json \
  --oracle crates/fsci-conformance/fixtures/artifacts/FSCI-P2C-002/oracle_capture.json
```

Use `--dry-run` to preview changes without writing the fixture.

## Verify Provenance

After capture + regen, confirm:

- `oracle_capture.json` contains:
  - `runtime.python_version`, `runtime.numpy_version`, `runtime.scipy_version`
  - `provenance.fixture_input_blake3`, `provenance.oracle_output_blake3`, `provenance.capture_blake3`
- The fixture contains top-level `oracle_provenance` and per-case provenance metadata.

Quick checks:

```bash
rg -n "oracle_provenance" crates/fsci-conformance/fixtures/FSCI-P2C-002_linalg_core.json
rg -n "provenance" crates/fsci-conformance/fixtures/artifacts/FSCI-P2C-002/oracle_capture.json
```

## Optional Failure Artifact

If the harness runs with `required=false` (default) and SciPy is unavailable,
`oracle_capture.error.txt` is written to the packet artifact directory. This is
expected and should be preserved for auditability.

## CI Lane: SciPy-Present Oracle Capture

Recommended CI steps:

1. Set up Python with SciPy + NumPy.
2. Run the targeted oracle capture test:
   - `cargo test -p fsci-conformance --test evidence_p2c002 oracle_capture_p2c002_linalg -- --nocapture`
3. Upload `crates/fsci-conformance/fixtures/artifacts/FSCI-P2C-002/oracle_capture.json` as a CI artifact.
4. Optionally run `fixture_regen --dry-run` and fail CI if it reports drift.

This lane is additive; the rest of CI should continue to run in environments
without SciPy and rely on the explicit fallback behavior documented in README.
