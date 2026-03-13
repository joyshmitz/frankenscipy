# FEATURE_PARITY

## Status Legend

- `not_started`
- `in_progress`
- `parity_green`
- `parity_gap`

## Feature Family Matrix

| Feature Family | Status | Notes |
|---|---|---|
| IVP tolerance validation and step guards (`FSCI-P2C-001`) | in_progress | clean-room `validate_tol`, `validate_first_step`, `validate_max_step` ported in `fsci-integrate`; parity artifacts generated |
| Dense linear algebra (`FSCI-P2C-002`) | in_progress | clean-room `solve`, `solve_triangular`, `solve_banded`, `inv`, `det`, `lstsq`, `pinv` landed in `fsci-linalg`; packet harness + artifacts active |
| Root and optimize core (`FSCI-P2C-003`) | in_progress | BFGS, Powell, CG minimize + Brentq/Brenth/Bisect/Ridder root-finding in `fsci-opt`; 28-case fixture corpus, 8 E2E scenarios, differential tests, contract table, threat matrix, parity report with RaptorQ sidecar landed |
| Sparse array baseline (`FSCI-P2C-004`) | in_progress | CSR/CSC/COO formats, spsolve, arithmetic ops in `fsci-sparse`; 15-case fixture corpus (spmv all formats, format roundtrip, add, scale, spsolve error, dimension mismatch), parity report with RaptorQ sidecar landed |
| FFT backend routing (`FSCI-P2C-005`) | in_progress | FFT/IFFT/RFFT/IRFFT/FFT2 transforms + fftfreq/rfftfreq/fftshift/ifftshift in `fsci-fft`; 15-case fixture corpus, parity report with RaptorQ sidecar landed |
| Special-function backend/error policy (`FSCI-P2C-006`) | in_progress | Bessel, gamma, beta, error functions in `fsci-special`; fixture corpus and E2E tests active |
| Array API compatibility glue (`FSCI-P2C-007`) | in_progress | Backend negotiation, broadcasting, indexing in `fsci-arrayapi`; property tests and benchmarks active |
| Differential harness + artifact pipeline (`FSCI-P2C-008`) | in_progress | packet fixtures, parity report generation, RaptorQ sidecar + decode-proof metadata, optional SciPy oracle capture, `ftui` dashboard, and CASP runtime conformance (15 cases: policy decisions, solver selection, calibrator drift) landed in `fsci-conformance` |

## Packet Readiness Snapshot

| Packet ID | Extraction | Impl | Conformance Fixtures | Sidecar Artifacts | Overall |
|---|---|---|---|---|---|
| `FSCI-P2C-001` | ready | in_progress | in_progress | in_progress | in_progress |
| `FSCI-P2C-002` | ready | in_progress | in_progress | in_progress | in_progress |
| `FSCI-P2C-003` | ready | in_progress | in_progress | in_progress | in_progress |
| `FSCI-P2C-004` | ready | in_progress | in_progress | in_progress | in_progress |
| `FSCI-P2C-005` | ready | in_progress | in_progress | in_progress | in_progress |
| `FSCI-P2C-006` | ready | in_progress | in_progress | in_progress | in_progress |
| `FSCI-P2C-007` | ready | in_progress | in_progress | in_progress | in_progress |
| `FSCI-P2C-008` | ready | in_progress | in_progress | in_progress | in_progress |

## Required Evidence Per Feature Family

1. Differential fixture report.
2. Edge-case/adversarial test results.
3. Benchmark delta (for runtime-significant paths).
4. Documented compatibility exceptions (if any).
5. RaptorQ sidecar manifest plus decode-proof record for each durable artifact bundle.
