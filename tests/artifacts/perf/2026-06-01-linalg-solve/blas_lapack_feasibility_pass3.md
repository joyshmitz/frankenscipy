# BLAS/LAPACK Feasibility - Pass 3

**Bead:** `frankenscipy-perf-linalg-lu-scalar-2y3wp`
**Skill loop:** `extreme-software-optimization`, pass 3 of 5
**Verdict:** rejected; no production Rust code or Cargo configuration kept.

## Target

Evaluate whether a linked LAPACK `getrf`/`getrs` path can replace the profiled
`nalgebra::DMatrix::lu()` DirectLU hotspot for the deterministic
`perf_solve solve 1000 1 42` scenario without weakening pivoting, fallback,
warning, certificate, or tolerance behavior.

## Baseline

Command:

```bash
RCH_FORCE_REMOTE=1 rch exec -- hyperfine --setup 'env CARGO_TARGET_DIR=/tmp/rch_target_fsci_linalg_lapack_pass3_base RUSTFLAGS="-C force-frame-pointers=yes" cargo build -p fsci-linalg --profile release-perf --bin perf_solve' --warmup 3 --runs 10 --export-json tests/artifacts/perf/2026-06-01-linalg-solve/pass3_blas_lapack_baseline_rch.json '/tmp/rch_target_fsci_linalg_lapack_pass3_base/release-perf/perf_solve solve 1000 1 42'
```

Result for `solve 1000 1 42`:

| metric | value |
|--------|-------|
| mean +- sigma | 130.4 ms +- 8.2 ms |
| median | 132.6 ms |
| min / max | 118.0 / 142.8 ms |
| user / system | 97.8 / 31.8 ms |

Stage command:

```bash
RCH_FORCE_REMOTE=1 rch exec -- hyperfine --setup 'env CARGO_TARGET_DIR=/tmp/rch_target_fsci_linalg_lapack_pass3_stage RUSTFLAGS="-C force-frame-pointers=yes" cargo build -p fsci-linalg --profile release-perf --bin perf_solve' --warmup 2 --runs 10 --export-json tests/artifacts/perf/2026-06-01-linalg-solve/pass3_blas_lapack_stage_rch.json '/tmp/rch_target_fsci_linalg_lapack_pass3_stage/release-perf/perf_solve lu_factor 1000 1 42' '/tmp/rch_target_fsci_linalg_lapack_pass3_stage/release-perf/perf_solve lu_solve 1000 1 42'
```

| mode | mean +- sigma | median | user / system |
|------|---------------|--------|---------------|
| `lu_factor` | 95.7 ms +- 6.2 ms | 95.7 ms | 82.4 / 12.9 ms |
| `lu_solve` | 117.4 ms +- 8.6 ms | 114.6 ms | 92.4 / 24.7 ms |

## Candidate Probe

Temporary harness-only shape:

- add `nalgebra-lapack = { version = "0.27.0", default-features = false, features = ["lapack-custom"] }`;
- add `perf_solve` modes for `lapack_lu_factor`, `lapack_lu_solve`, and `lapack_golden`;
- keep production `solve`, `lu_factor`, `LuFactorResult`, `ConditionDiagnosticsWork`, and CASP dispatch unchanged.

The plain custom linker form failed on the rch worker because only versioned
runtime libraries were visible:

```bash
RCH_FORCE_REMOTE=1 rch exec -- env CARGO_TARGET_DIR=/tmp/rch_target_fsci_linalg_lapack_pass3_candidate RUSTFLAGS='-C force-frame-pointers=yes -C link-arg=-llapack -C link-arg=-lblas' cargo build -p fsci-linalg --profile release-perf --bin perf_solve
```

Failure:

```text
rust-lld: error: unable to find library -llapack
rust-lld: error: unable to find library -lblas
```

The absolute versioned system-library form compiled:

```bash
RCH_FORCE_REMOTE=1 rch exec -- env CARGO_TARGET_DIR=/tmp/rch_target_fsci_linalg_lapack_pass3_candidate_abs RUSTFLAGS='-C force-frame-pointers=yes -C link-arg=/lib/x86_64-linux-gnu/liblapack.so.3 -C link-arg=/lib/x86_64-linux-gnu/libblas.so.3' cargo build -p fsci-linalg --profile release-perf --bin perf_solve
```

Candidate benchmark:

```bash
RCH_FORCE_REMOTE=1 rch exec -- hyperfine --setup 'env CARGO_TARGET_DIR=/tmp/rch_target_fsci_linalg_lapack_pass3_candidate_abs RUSTFLAGS="-C force-frame-pointers=yes -C link-arg=/lib/x86_64-linux-gnu/liblapack.so.3 -C link-arg=/lib/x86_64-linux-gnu/libblas.so.3" cargo build -p fsci-linalg --profile release-perf --bin perf_solve' --warmup 2 --runs 10 --export-json tests/artifacts/perf/2026-06-01-linalg-solve/pass3_blas_lapack_candidate_stage_rch.json '/tmp/rch_target_fsci_linalg_lapack_pass3_candidate_abs/release-perf/perf_solve lapack_lu_factor 1000 1 42' '/tmp/rch_target_fsci_linalg_lapack_pass3_candidate_abs/release-perf/perf_solve lapack_lu_solve 1000 1 42'
```

| mode | baseline | candidate | delta |
|------|----------|-----------|-------|
| factorization | 95.7 ms +- 6.2 ms | 86.3 ms +- 4.2 ms | 9.8% faster |
| factor + solve probe | 117.4 ms +- 8.6 ms | 90.4 ms +- 2.4 ms | 23.0% faster |

This proves a getrf/getrs-style path is measurable in isolation. It does not
prove a safe production substitution because production DirectLU also feeds
CASP condition diagnostics, warning thresholds, fallback ordering, and
`SolveResult.backward_error`.

## Golden Output

Baseline golden command:

```bash
RCH_FORCE_REMOTE=1 rch exec -- bash -lc 'env CARGO_TARGET_DIR=/tmp/rch_target_fsci_linalg_lapack_pass3_golden RUSTFLAGS="-C force-frame-pointers=yes" cargo build -p fsci-linalg --profile release-perf --bin perf_solve >/dev/null && /tmp/rch_target_fsci_linalg_lapack_pass3_golden/release-perf/perf_solve golden > tests/artifacts/perf/2026-06-01-linalg-solve/golden/golden_blas_lapack_pass3_before.txt && sha256sum tests/artifacts/perf/2026-06-01-linalg-solve/golden/golden_blas_lapack_pass3_before.txt'
```

Candidate golden command:

```bash
RCH_FORCE_REMOTE=1 rch exec -- bash -lc '/tmp/rch_target_fsci_linalg_lapack_pass3_candidate_abs/release-perf/perf_solve lapack_golden > tests/artifacts/perf/2026-06-01-linalg-solve/golden/golden_blas_lapack_pass3_candidate.txt && sha256sum tests/artifacts/perf/2026-06-01-linalg-solve/golden/golden_blas_lapack_pass3_before.txt tests/artifacts/perf/2026-06-01-linalg-solve/golden/golden_blas_lapack_pass3_candidate.txt'
```

Checksums:

```text
5809995418488c93cc66dc6f2dc01a0d5fd8e2d8faab6f9a7c44241e99025bdd  golden/golden_blas_lapack_pass3_before.txt
bb17fe3b039a9f8e19cb96db8e6cdf1c74977b5a9933dff5b9b4bff55ef22443  golden/golden_blas_lapack_pass3_candidate.txt
```

The candidate harness emitted only the default-orientation DirectLU cases. That
subset matched the baseline bit-for-bit:

```bash
awk '$0 ~ / t=0 / { print }' tests/artifacts/perf/2026-06-01-linalg-solve/golden/golden_blas_lapack_pass3_before.txt | sha256sum
sha256sum tests/artifacts/perf/2026-06-01-linalg-solve/golden/golden_blas_lapack_pass3_candidate.txt
```

Both commands produced:

```text
bb17fe3b039a9f8e19cb96db8e6cdf1c74977b5a9933dff5b9b4bff55ef22443
```

So the measured default target scenario did not show output-bit drift in the
small golden set, but the full existing golden contract was not proven for a
production replacement.

## Feasibility Notes

Official documentation checked:

- nalgebra documents its in-crate `LU` as partial row pivoting and exposes
  `.solve()` / `.solve_mut()` for linear systems:
  https://www.nalgebra.rs/docs/user_guide/decompositions_and_lapack/
- `nalgebra-lapack` documents a safe `LU::new` / `.solve()` API for LU with
  partial pivoting, backed by LAPACK routines:
  https://docs.rs/nalgebra-lapack/latest/nalgebra_lapack/struct.LU.html
- `nalgebra-lapack` backend selection requires exactly one `lapack-*` feature;
  selecting a non-default backend requires disabling default features:
  https://docs.rs/nalgebra-lapack/latest/src/nalgebra_lapack/lib.rs.html
- `nalgebra-lapack` feature metadata maps `lapack-openblas` to
  `lapack-src/openblas` and `blas-src/openblas`:
  https://docs.rs/crate/nalgebra-lapack/latest/features

Options checked:

| option | finding | decision |
|--------|---------|----------|
| `nalgebra-lapack 0.28.0` | depends on `nalgebra ^0.35`; current crate uses `nalgebra 0.34.2` | reject for this pass; would introduce a version upgrade or duplicate matrix stack |
| `nalgebra-lapack 0.27.0 + lapack-custom` | shares `nalgebra ^0.34`; compiles only with absolute `/lib/.../liblapack.so.3` and `/lib/.../libblas.so.3` link args on rch | reject for production; brittle operational dependency |
| `lapack-openblas` | no OpenBLAS library is visible from `ldconfig`/`pkg-config`; feature would require adding/building provider dependencies | reject for this one-lever pass |
| default `lapack-netlib` | official docs call it practical but not typically the best performing backend; it would add a bundled Fortran LAPACK/BLAS provider | reject for this pass; lower performance confidence than the measured system path |

Production integration blocker:

- `ConditionDiagnosticsWork` stores `lu_cache: Option<nalgebra::LU<f64, Dyn, Dyn>>`.
- `LuFactorResult` stores `lu_internal: nalgebra::LU<f64, Dyn, Dyn>`.
- `fast_rcond_from_lu` and warning/certificate behavior are built around that
  nalgebra LU representation.
- A real LAPACK replacement would need either a second LAPACK-backed rcond path
  or a dual-factorization design. That is more than one optimization lever and
  could silently change `rcond_warning`, fallback selection, certificates, or
  hardened validation behavior.

## Isomorphism Proof

Retained change: none.

- Ordering preserved: yes by construction; production dispatch code unchanged.
- Tie-breaking unchanged: yes by construction; expected-loss ordering unchanged.
- Floating-point: unchanged in retained code. The isolated default-orientation
  candidate matched the small golden subset, but no full production golden
  proof was established.
- RNG seeds: unchanged; `perf_solve` deterministic seed generation unchanged in
  retained code.
- Golden outputs: retained code remains at
  `5809995418488c93cc66dc6f2dc01a0d5fd8e2d8faab6f9a7c44241e99025bdd`.

## Opportunity Score

| lever | impact | confidence | effort | score | decision |
|-------|--------|------------|--------|-------|----------|
| Replace DirectLU with `nalgebra-lapack 0.27 + lapack-custom` | 3 | 1 | 5 | 0.6 | reject |
| Add `lapack-openblas` provider/backend path | 4 | 1 | 5 | 0.8 | reject |
| Add default `lapack-netlib` backend path | 1 | 1 | 4 | 0.25 | reject |

No LAPACK/backend lever meets the required `Impact x Confidence / Effort >= 2.0`
bar after including link portability, rcond/certificate integration, and full
golden-proof effort.
