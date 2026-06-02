# Pass 4 - Blocked Kernel Feasibility

Date: 2026-06-01T21:30:29-0400
Agent: OliveSnow
Target bead: `frankenscipy-perf-linalg-lu-scalar-2y3wp`
Git before pass: `975cbed4`

## Mission

Evaluate whether a crate-local safe Rust blocked/panel DGETRF-style path is a
valid one-lever candidate for the profiled dense LU scalar hotspot.

## Fresh Remote Baseline

Command:

```bash
RCH_FORCE_REMOTE=1 rch exec -- hyperfine --setup 'env CARGO_TARGET_DIR=/tmp/rch_target_fsci_linalg_pass4_olivesnow_base RUSTFLAGS="-C force-frame-pointers=yes" cargo build -p fsci-linalg --profile release-perf --bin perf_solve' --warmup 3 --runs 10 --export-json tests/artifacts/perf/2026-06-01-linalg-solve/pass4_blocked_kernel_olivesnow_baseline_rch.json '/tmp/rch_target_fsci_linalg_pass4_olivesnow_base/release-perf/perf_solve solve 1000 1 42' '/tmp/rch_target_fsci_linalg_pass4_olivesnow_base/release-perf/perf_solve lu_factor 1000 1 42'
```

Results:

| mode | mean | stddev | median | min | max | user | system |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `solve 1000 1 42` | 124.8 ms | 5.0 ms | 124.6 ms | 117.0 ms | 132.9 ms | 96.8 ms | 27.6 ms |
| `lu_factor 1000 1 42` | 96.3 ms | 6.3 ms | 93.3 ms | 89.6 ms | 109.0 ms | 82.6 ms | 13.3 ms |

`lu_factor` remains about 77.2% of end-to-end `solve`.

## Fresh Profile Check

Command:

```bash
out=tests/artifacts/perf/2026-06-01-linalg-solve/pass4_blocked_kernel_olivesnow_profile.txt; bin=/tmp/rch_target_fsci_linalg_pass4_olivesnow_base/release-perf/perf_solve; "$bin" solve 1000 80 42 >/tmp/pass4_olivesnow_profile_run.out 2>&1 & pid=$!; { printf 'command: %s solve 1000 80 42\n' "$bin"; printf 'sampler: gdb -batch -p <pid> -ex bt, 20 samples\n'; for i in $(seq 1 20); do printf '===SAMPLE %s===\n' "$i"; gdb -batch -p "$pid" -ex bt 2>&1 | sed -n '1,24p'; sleep 0.25; done; wait "$pid"; printf '===RUN OUTPUT===\n'; cat /tmp/pass4_olivesnow_profile_run.out; } > "$out"
```

Result:

- The fresh sampler still hits nalgebra generic LU scalar work:
  `gauss_step -> axpy -> array_axcpy` in the sampled stacks.
- `rg -c "gauss_step|array_axcpy|blas_uninit" pass4_blocked_kernel_olivesnow_profile.txt`
  returned `54`.
- Representative stack locations:
  - `nalgebra-0.34.2/src/base/blas_uninit.rs:49`
  - `nalgebra-0.34.2/src/linalg/lu.rs:355`
  - `crates/fsci-linalg/src/lib.rs:1132`
  - `crates/fsci-linalg/src/lib.rs:1574`

The profile condition for considering a blocked kernel is met.

## Feasibility Analysis

The production DirectLU path does not use LU factorization as an isolated
numeric black box. The same `nalgebra::LU` object created in
`condition_diagnostics_with_assumption` feeds:

- `fast_rcond_from_lu` and the `ConditionReport.rcond_estimate`
- `SolverPortfolio::select_action` ordering through expected-loss sorting
- the `DirectLU` solve in `dispatch_solve_action`
- `rcond_warning`
- `compute_backward_error` against the retained pristine `DMatrix`
- strict and hardened validation behavior around non-finite input, shape, and
  condition rejection
- public `lu_factor` / `lu_solve` compatibility if the replacement is made
  generally instead of only for `solve`

A real blocked/panel DGETRF replacement therefore has to replace factorization,
pivot representation, triangular solves, rcond estimation, warnings, and the
observable backward-error/certificate contract together. That is not a
single-lever production change in this codebase.

A harness-only prototype was not implemented because it would not retire the
main production risk: bit-identical integration with the existing CASP/rcond
and backward-error surface. A true blocked DGETRF update would also normally
batch trailing updates as panel GEMM, which changes floating-point grouping
relative to nalgebra's current per-pivot rank-1 update. Under this campaign's
golden contract, that makes behavior drift likely unless the "blocked" kernel
degenerates to nalgebra-equivalent unblocked ordering, which would not address
the profiled scalar overhead.

## Golden Proof

Command:

```bash
/tmp/rch_target_fsci_linalg_pass4_olivesnow_base/release-perf/perf_solve golden > tests/artifacts/perf/2026-06-01-linalg-solve/golden/golden_blocked_pass4_olivesnow_before.txt
sha256sum tests/artifacts/perf/2026-06-01-linalg-solve/golden/golden_blocked_pass4_olivesnow_before.txt tests/artifacts/perf/2026-06-01-linalg-solve/golden/golden_blas_lapack_pass3_before.txt
```

Result:

```text
5809995418488c93cc66dc6f2dc01a0d5fd8e2d8faab6f9a7c44241e99025bdd  tests/artifacts/perf/2026-06-01-linalg-solve/golden/golden_blocked_pass4_olivesnow_before.txt
5809995418488c93cc66dc6f2dc01a0d5fd8e2d8faab6f9a7c44241e99025bdd  tests/artifacts/perf/2026-06-01-linalg-solve/golden/golden_blas_lapack_pass3_before.txt
```

Because no Rust code or Cargo configuration was changed, production ordering,
tie-breaking, floating-point operations, RNG behavior, warning behavior,
certificates, and backward-error bits are unchanged by construction.

## Opportunity Score

| Candidate | Impact | Confidence | Effort | Score |
| --- | ---: | ---: | ---: | ---: |
| Production safe-Rust blocked/panel LU replacement | 4 | 1 | 5 | 0.8 |
| Harness-only blocked LU probe | 1 | 1 | 2 | 0.5 |

Both candidates are below the required `2.0` threshold.

## Decision

Rejected with zero production code change.

Reason: no viable safe one-lever implementation. The profile still identifies
nalgebra scalar LU overhead, but a production blocked kernel has too large an
integration surface and too low a bit-identical behavior confidence for this
pass. A harness-only prototype would measure an isolated kernel but would not
prove the existing golden contract for `solve`, `lu_factor`, `lu_solve`,
`rcond`, warnings, certificates, and `backward_error`.

Pass 5 should re-profile and hand off the bead rather than attempting another
backend/kernel swap under the current bit-identical contract.

## Subagent Harness Probe Addendum

An independent subagent probe temporarily added a harness-only safe-Rust
row-major blocked LU mode to `perf_solve.rs`, then restored the file manually.
No Rust code diff remains.

Additional baseline, using the restored production binary:

| mode | mean | stddev | median | min | max | user | system |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `solve 1000 1 42` | 118.4 ms | 2.5 ms | 118.8 ms | 114.8 ms | 122.4 ms | 91.2 ms | 26.9 ms |
| `lu_factor 1000 1 42` | 96.9 ms | 8.1 ms | 94.0 ms | 87.1 ms | 112.3 ms | 83.0 ms | 13.6 ms |
| `lu_solve 1000 1 42` | 116.3 ms | 5.8 ms | 115.1 ms | 109.4 ms | 127.0 ms | 90.7 ms | 25.3 ms |

Additional gdb sampling over `perf_solve lu_factor 1000 80 42` captured 20
samples:

- `gauss_step`: `17` matched lines
- `array_axcpy|axcpy_uninit`: `37` matched lines
- `core::clone|clone::impls`: `14` matched lines
- `matrixmultiply|dgemm|sgemm|gemm|mat_mul`: `0` matched lines
- `from_row_slice_generic|dmatrix_from_rows`: `3` matched lines

The temporary blocked harness measured faster in isolation:

| candidate mode | mean | stddev | median | min | max | user | system |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `blocked_lu_factor 1000 1 42` | 80.7 ms | 4.7 ms | 80.4 ms | 73.7 ms | 88.2 ms | 70.4 ms | 10.0 ms |
| `blocked_lu_solve 1000 1 42` | 80.7 ms | 5.0 ms | 79.6 ms | 74.8 ms | 89.4 ms | 71.3 ms | 9.0 ms |

Golden status:

```text
5809995418488c93cc66dc6f2dc01a0d5fd8e2d8faab6f9a7c44241e99025bdd  golden_blocked_pass4_before.txt
86b605b4e528a7ab9ab120cba7d171d32efdd9470b5a0b77f1cd2fcd7c7724f3  golden_blocked_pass4_candidate.txt
```

The first diff already changes `x` bits and `backward_error` bits on small
golden cases, so confidence for a production replacement remains zero under the
bit-identical contract. This reinforces the rejection: the kernel idea has a
speed signal, but it is not a keepable one-lever change for the current
observable CASP/rcond/backward-error surface.

Additional artifacts:

- `pass4_blocked_kernel_baseline_rch.json`
- `pass4_blocked_kernel_stage_rch.json`
- `pass4_blocked_kernel_gdb_samples.txt`
- `pass4_blocked_kernel_candidate_stage_rch.json`
- `golden/golden_blocked_pass4_before.txt`
- `golden/golden_blocked_pass4_candidate.txt`
