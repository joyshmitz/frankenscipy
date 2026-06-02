# Pass 5 - Final Re-profile Handoff

Date: 2026-06-01T21:40:18-0400
Agent: OliveSnow
Target bead: `frankenscipy-perf-linalg-lu-scalar-2y3wp`
Git before pass: `b04b0709`

## Mission

Re-run the final remote benchmark/profile for the current dense LU solve path,
verify the retained golden output, and prepare closure evidence for the active
LU-scalar performance bead. This subagent did not close the bead, commit, push,
or edit Rust/Cargo code.

## Final Remote Benchmark

Command:

```bash
RCH_FORCE_REMOTE=1 rch exec -- hyperfine \
  --setup 'env CARGO_TARGET_DIR=/tmp/rch_target_fsci_linalg_pass5_final_bench_20260602a RUSTFLAGS="-C force-frame-pointers=yes" cargo build -p fsci-linalg --profile release-perf --bin perf_solve' \
  --warmup 3 \
  --runs 10 \
  --export-json tests/artifacts/perf/2026-06-01-linalg-solve/pass5_final_benchmark_rch.json \
  '/tmp/rch_target_fsci_linalg_pass5_final_bench_20260602a/release-perf/perf_solve solve 1000 1 42' \
  '/tmp/rch_target_fsci_linalg_pass5_final_bench_20260602a/release-perf/perf_solve lu_factor 1000 1 42' \
  '/tmp/rch_target_fsci_linalg_pass5_final_bench_20260602a/release-perf/perf_solve lu_solve 1000 1 42'
```

Results:

| mode | mean | stddev | median | min | max | user | system |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `solve 1000 1 42` | 117.4 ms | 6.7 ms | 115.4 ms | 110.5 ms | 129.8 ms | 90.2 ms | 26.6 ms |
| `lu_factor 1000 1 42` | 95.8 ms | 4.8 ms | 97.1 ms | 88.8 ms | 104.6 ms | 81.8 ms | 13.4 ms |
| `lu_solve 1000 1 42` | 118.2 ms | 5.7 ms | 118.1 ms | 110.2 ms | 126.0 ms | 95.2 ms | 22.5 ms |

`lu_factor` remains 81.6% of the end-to-end `solve` mean and is still the
dominant measured stage in this scenario.

## Final Profile Signal

Raw profile artifact: `pass5_final_gdb_profile.txt`.
Corrected sample-block counts: `pass5_final_profile_counts.txt`.

| signal | samples |
| --- | ---: |
| total samples | 25 |
| `gauss_step` | 18 |
| `array_axcpy` / `axcpy_uninit` / `blas_uninit` | 17 |
| `matrixmultiply` / GEMM | 0 |
| CASP dispatch / `SolverPortfolio::select_action` | 0 |
| matrix copy setup / `from_rows` / `from_row_slice` | 0 |

Representative stacks still run through:

```text
fsci_linalg::solve
fsci_linalg::solve_with_portfolio_internal
fsci_linalg::condition_diagnostics_with_assumption
nalgebra::DMatrix::lu
nalgebra::LU::new
nalgebra::linalg::lu::gauss_step
nalgebra::base::blas_uninit::axcpy_uninit
nalgebra::base::blas_uninit::array_axcpy
```

The remaining hotspot is nalgebra scalar LU (`gauss_step` / `array_axcpy`), not
CASP dispatch, matrix copy setup, or a matrixmultiply/GEMM path.

## Golden Proof

Command:

```bash
RCH_FORCE_REMOTE=1 rch exec -- bash -lc 'target=/tmp/rch_target_fsci_linalg_pass5_final_golden_20260602a; env CARGO_TARGET_DIR="$target" RUSTFLAGS="-C force-frame-pointers=yes" cargo build -p fsci-linalg --profile release-perf --bin perf_solve; "$target/release-perf/perf_solve" golden > tests/artifacts/perf/2026-06-01-linalg-solve/pass5_final_golden.txt; sha256sum tests/artifacts/perf/2026-06-01-linalg-solve/pass5_final_golden.txt > tests/artifacts/perf/2026-06-01-linalg-solve/pass5_final_golden.sha256'
```

Result:

```text
5809995418488c93cc66dc6f2dc01a0d5fd8e2d8faab6f9a7c44241e99025bdd  tests/artifacts/perf/2026-06-01-linalg-solve/pass5_final_golden.txt
```

The final golden matches the retained campaign hash exactly. Because no code or
configuration changed in this pass, ordering, tie-breaking, floating-point
operation order, RNG, warnings, certificates, and `backward_error` remain
unchanged by construction.

## Decision

Recommended bead disposition: close
`frankenscipy-perf-linalg-lu-scalar-2y3wp` as rejected/no keepable one-lever
optimization after five profile-backed passes.

Rationale:

- Pass 1 rejected dependency/config-only backend feature changes.
- Pass 2 rejected residual clone removal because observable `backward_error`
  bits changed.
- Pass 3 rejected LAPACK/BLAS replacement because provider/CASP/golden
  integration risk made it too large for a one-lever change.
- Pass 4 rejected a blocked safe-Rust kernel because a faster harness prototype
  changed `x` and `backward_error` bits.
- Pass 5 reconfirms the hotspot remains nalgebra scalar LU, but no keepable
  one-lever optimization remains under the bit-identical behavior contract.

No production code changes were made for this bead beyond the earlier separate
matrix-copy optimization already handled outside this active LU-scalar bead.

Recommended next perf state: after closure verification, return to
`br ready --json` for the next profile-backed `[perf]` bead. If this hotspot is
reopened, it should be a larger design bead for an LU backend contract change
covering CASP/rcond, certificates, warnings, and golden behavior policy.

## Validation

Lightweight artifact validation for this pass:

```bash
jq empty tests/artifacts/perf/2026-06-01-linalg-solve/pass5_final_benchmark_rch.json
git diff --check -- tests/artifacts/perf/2026-06-01-linalg-solve/pass5_final_reprofile_handoff.md .skill-loop-progress.md
```

Result: both commands passed. `sha256sum -c` also passed for
`pass5_final_golden.sha256`.
