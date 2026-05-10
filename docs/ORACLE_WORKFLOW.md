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

The primary CI lane is `g3-live-scipy-oracle` in `.github/workflows/ci.yml`.
It installs NumPy + SciPy, runs the Rust capture binary in required-oracle mode,
and uploads `target/live-oracle-capture.json`.

```bash
cargo run -p fsci-conformance --bin live_oracle_capture -- \
  --python python \
  --output target/live-oracle-capture.json
```

Required-oracle mode is intentional: do not pass `--allow-missing-oracle` in the
SciPy-present lane. The command fails if SciPy cannot be imported, if no packets
are captured, or if any packet exceeds the zero-drift thresholds. The broader
`g3-conformance` job may still run an optional fallback capture for non-SciPy
environments, but that fallback is not the release signal for live oracle parity.

For a targeted local smoke test, use the linalg capture test:

```bash
cargo test -p fsci-conformance --test evidence_p2c002 oracle_capture_p2c002_linalg -- --nocapture
```

Optionally run `fixture_regen --dry-run` against a specific capture and fail CI
if it reports fixture drift.
