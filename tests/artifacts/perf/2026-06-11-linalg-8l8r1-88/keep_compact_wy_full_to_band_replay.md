# `frankenscipy-8l8r1.88` keep: compact-WY full-to-band replay stage

## Scope

- Bead: `frankenscipy-8l8r1.88`
- Lever: private safe-Rust compact-WY replay for the deterministic symmetric full-to-band reduction stage.
- Public dispatch: unchanged. `eigh`, `eigvalsh`, ordering, audit trace, RNG behavior, and nalgebra fallback remain on the existing path.
- Forbidden dependencies: no C BLAS/LAPACK/MKL/XLA and no `unsafe`.

## Baseline

RCH command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 rch exec -- env CARGO_BUILD_JOBS=1 cargo bench -j 1 -p fsci-linalg --bench linalg_bench --locked -- eigh_dense
```

Worker: `vmi1153651`

Criterion baseline:

- `eigh_dense/256x256`: mean `28.781 ms`
- `eigh_dense/512x512`: mean `225.81 ms`

Baseline artifact: `baseline_eigh_dense_rch.txt`

## Behavior Proof

Focused private replay proof:

```bash
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 RCH_WORKER=vmi1153651 rch exec -- env CARGO_BUILD_JOBS=1 cargo test -j 1 -p fsci-linalg --lib --locked -- compact_wy_full_to_band --nocapture
```

Result:

- `compact_wy_full_to_band_replay_matches_scalar_replay`: passed
- `compact_wy_full_to_band_replay_rejects_invalid_shapes`: passed
- Maximum printed `Q` orthogonality residual: `1.33226762955018785e-15`
- Full-to-band replay digests:
  - `n=18 bandwidth=1 panel_width=3`: scalar `0x8d7aa616cc956e53`, compact `0xeab9b611c144296d`
  - `n=37 bandwidth=2 panel_width=4`: scalar `0x6ec4df6bd0b28e98`, compact `0x3b822e48c186910c`
  - `n=64 bandwidth=4 panel_width=6`: scalar `0x926b2f472c4c0fb7`, compact `0x9b613f47c16b5b3f`

Public golden proof:

```bash
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 RCH_WORKER=vmi1153651 rch exec -- env CARGO_BUILD_JOBS=1 cargo test -j 1 -p fsci-linalg --lib --locked -- eigh_index_sort_matches_materialized_pair_sort_bits --nocapture
```

Result:

- `eigh_index_sort_public_golden_digest=0x287a5d3679a8bc6a`
- This matches the previous public `eigh` golden digest.

Isomorphism checklist:

- Ordering/tie-breaking: preserved by unchanged public `eigh` path and matching golden digest.
- Floating point tolerance: private replay matches scalar reference with `max_abs_diff <= 6.13908923696726561e-12` in the release probe.
- RNG: no random source added; deterministic fixtures only.
- Safety: no `unsafe` added.
- External kernels: no C BLAS/LAPACK/MKL/XLA linkage added.

## Same-Worker Rebench

RCH command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 RCH_WORKER=vmi1153651 rch exec -- env CARGO_BUILD_JOBS=1 cargo test --release -j 1 -p fsci-linalg --lib --locked -- compact_wy_full_to_band_replay_perf_probe --ignored --nocapture
```

Worker: `vmi1153651`

Private full-to-band replay stage:

| Shape | Scalar replay | Compact-WY replay | Speedup | Max abs diff |
| --- | ---: | ---: | ---: | ---: |
| `256x256` | `22.181579 ms` | `17.941533 ms` | `1.236326x` | `1.59161572810262442e-12` |
| `512x512` | `239.992412 ms` | `192.779419 ms` | `1.244907x` | `6.13908923696726561e-12` |

Public `eigh` side probe in the same release test:

- `256x256`: `27.565006 ms`
- `512x512`: `236.145619 ms`

The public side probe is recorded only to keep the profiled bottleneck visible; public dispatch was intentionally not changed in this bead.

## Gates

- `rustfmt --edition 2024 --check crates/fsci-linalg/src/lib.rs`: passed
- `git diff --check -- crates/fsci-linalg/src/lib.rs`: passed
- `ubs crates/fsci-linalg/src/lib.rs`: exit 0, zero critical findings; broad pre-existing warning inventory remains.
- `cargo check -j 1 -p fsci-linalg --lib --locked`: passed on `vmi1153651`
- `cargo clippy -j 1 -p fsci-linalg --lib --locked --no-deps -- -D warnings`: passed on `vmi1153651`

## Score

Score formula: `(Impact * Confidence) / Effort`

- Impact: `2.5` because this removes scalar reflector replay from the private full-to-band stage, with a `1.244907x` stage win at `512x512`, but does not yet speed up public `eigh`.
- Confidence: `4.0` because the proof and rebench are same-worker, deterministic, and public golden behavior is unchanged.
- Effort: `2.5` because the lever is private but touches a large numerical module and adds proof/probe scaffolding.

Score: `(2.5 * 4.0) / 2.5 = 4.0`

Verdict: `KEEP`

## Next Route

Wire this private compact-WY full-to-band stage into the next public two-stage `eigh`/`eigvalsh` candidate. Do not repeat scalar reflector replay, lower-triangle packing, direct scalar tridiagonalization, output materialization, or GEMM micro-lever families.
