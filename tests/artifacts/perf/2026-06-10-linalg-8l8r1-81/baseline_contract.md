# frankenscipy-8l8r1.81 GEMM Baseline Contract

Date: 2026-06-10
Owner: PearlValley / BlackThrush coordination
Bead: `frankenscipy-8l8r1.81`
Target crate: `fsci-linalg`

## Profile-Backed Target

The residual hot path is public dense square `matmul` on the flat-workspace GEMM route in
`crates/fsci-linalg/src/lib.rs`. The previous direct B-copy/direct-pack staging lever was
proof-clean but rejected on same-worker Criterion evidence, so this pass must not repeat
B staging or direct-pack variants.

## RCH Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo bench -j 1 -p fsci-linalg --bench linalg_bench --locked -- matmul
```

Worker: `ovh-a`

| Benchmark | Criterion mean |
| --- | ---: |
| `matmul/256x256` | `6.4136 ms` |
| `matmul/512x512` | `28.029 ms` |
| `matmul/768x768` | `68.309 ms` |
| `matmul/1024x1024` | `87.181 ms` |

Baseline artifact SHA256:

```text
59333b3d356d7b50e58a6bbe2edd7d9f2c3d85cca5530c78909f76a82463545e  tests/artifacts/perf/2026-06-10-linalg-8l8r1-81/baseline_matmul_criterion_rch.txt
```

## Alien Primitive Lineage

The parent no-gaps primitive family is communication-avoiding, cache-blocked dense linear
algebra in pure safe Rust. For this bead the measured residual is inside the existing
blocked GEMM compute path, not the copy/setup path. The first allowed lever should alter
only compute-path scheduling/cache geometry or the register-panel microkernel, preserving
the arithmetic contract.

If this lane rejects, route deeper to a materially different safe-Rust blocked GEMM
microkernel, such as a wider row register panel or an explicit multi-panel `NC=32`
compute kernel. Do not file a ceiling conclusion.

## One-Lever Contract

Candidate lever under consideration: row-panel cache geometry / thread scheduling
stability in `matmul_flat_compute_rows` or `matmul_thread_count`.

Constraints:

- Exactly one source lever before rebench.
- No B staging, direct-pack, external BLAS/LAPACK, unsafe code, RNG, or algorithmic
  fallback change.
- Each `c[i][j]` must keep the same monotonic `k = 0..ka` reduction order.
- Output row ownership may change, but no two threads may write the same row.
- Public `matmul` shape checks, rectangularity gate, and fallback behavior must remain
  unchanged.

## Proof Plan

Behavior proof commands:

```bash
sha256sum -c tests/artifacts/perf/2026-06-10-linalg-8l8r1-81/baseline.sha256
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo test -j 1 -p fsci-linalg --lib --locked matmul_flat_workspace_is_bit_identical_to_naive_ijk -- --nocapture --test-threads=1
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo test -j 1 -p fsci-linalg --lib --locked matmul_flat_compute_rows_row_split_is_bit_identical -- --nocapture --test-threads=1
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo test -j 1 -p fsci-linalg --lib --locked matmul_microkernel_golden_digest -- --nocapture --test-threads=1
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo test -j 1 -p fsci-linalg --release --lib --locked matmul_medium_flat_workspace_route_golden_digest -- --ignored --nocapture --test-threads=1
```

Benchmark command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo bench -j 1 -p fsci-linalg --bench linalg_bench --locked -- matmul
```

Keep gate: same-worker or directly paired Criterion evidence, real affected-size win,
behavior proof/golden SHA clean, and Score `Impact x Confidence / Effort >= 2.0`.
