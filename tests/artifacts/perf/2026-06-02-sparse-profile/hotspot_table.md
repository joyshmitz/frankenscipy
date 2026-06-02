# fsci-sparse Profile Handoff

Scenario: `fsci-sparse` Criterion sparse benchmark matrix on deterministic random CSR inputs.

Command:

```text
RCH_FORCE_REMOTE=1 CARGO_TARGET_DIR=/data/tmp/cargo-target-frankenscipy-olivesnow-sparse-profile rch exec -- cargo bench -p fsci-sparse --bench sparse_bench --locked -- --warm-up-time 1 --measurement-time 2 --sample-size 10
```

Environment:
- RCH worker: `vmi1227854`
- Git input SHA: `d60607fe`
- Artifact: `tests/artifacts/perf/2026-06-02-sparse-profile/sparse_bench_rch.txt`
- Exit marker: `RCH_SPARSE_PROFILE_EXIT:0`

## Ranked Hotspots

| Rank | Location | Metric | Value | Category | Evidence |
|------|----------|--------|-------|----------|----------|
| 1 | `CscMatrix::to_csr` via `FormatConvertible` | Criterion mean, 10000x10000 d0 | 9.1720 ms | CPU/alloc | `sparse_bench_rch.txt:534-539` |
| 2 | `CsrMatrix::to_csc` via `FormatConvertible` | Criterion mean, 10000x10000 d0 | 7.9314 ms | CPU/alloc | `sparse_bench_rch.txt:526-531` |
| 3 | `CooMatrix::to_csr` construction | Criterion mean, 10000x10000 d0 | 3.2086 ms | CPU/alloc | `sparse_bench_rch.txt:467-472` |
| 4 | `add_csr` | Criterion mean, 10000x10000 d0 | 1.9436 ms | CPU | `sparse_bench_rch.txt:573-578` |
| 5 | `diags` tridiagonal construction | Criterion mean, n=10000 | 1.5433 ms | CPU/alloc | `sparse_bench_rch.txt:626-631` |

## Hypothesis Ledger

| Hypothesis | Verdict | Evidence |
|------------|---------|----------|
| Compressed CSR/CSC conversion is dominated by routing through COO and triplet sorting. | supports | `ops.rs` implements `CsrMatrix::to_csc` as `self.to_coo()?.to_csc()` and `CscMatrix::to_csr` as `self.to_coo()?.to_csr()`. These are the top two rows in the sparse profile. |
| Sparse matrix-vector multiply is the first target. | rejects | `sparse_spmv/10000x10000_d0_nnz100000` mean is 196.64 us, far below the conversion rows. |
| Sparse arithmetic is the first target. | rejects | `sparse_arithmetic/10000x10000_d0_add` mean is 1.9436 ms, below both conversion rows. |

## Opportunity Matrix

| Candidate Lever | Impact | Confidence | Effort | Score |
|-----------------|--------|------------|--------|-------|
| Replace CSR/CSC conversion through COO with direct compressed transpose for canonical matrices. | 5 | 4 | 3 | 6.67 |
| Reuse conversion work buffers across calls. | 3 | 2 | 4 | 1.50 |
| Tune `spmv_csr` row loop. | 1 | 3 | 2 | 1.50 |

Selected lever: direct compressed transpose for CSR/CSC conversion. The score is above the required 2.0 threshold and targets the top two profile rows.

## Repeat Loop State

Skill: `extreme-software-optimization`

Passes:
1. Profile and target selection: complete, sparse format conversion chosen from Criterion evidence.
2. Baseline and golden outputs: pending.
3. One-lever implementation: pending.
4. Behavior proof and validation: pending.
5. Re-benchmark and re-profile handoff: pending.
