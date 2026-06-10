# A-Panel GEMM Candidate - Provisional Evidence

Date: 2026-06-10
Bead: `frankenscipy-8l8r1.81`
Lever: per-row-chunk K-major A panel for the flat-workspace GEMM compute path.

## Behavior Proof

Baseline artifact hash:

```text
tests/artifacts/perf/2026-06-10-linalg-8l8r1-81/baseline_matmul_criterion_rch.txt: OK
```

RCH proof logs:

- `proof_flat_workspace_bits_rch.txt`: `matmul_flat_workspace_is_bit_identical_to_naive_ijk` passed on `ovh-a`.
- `proof_row_split_bits_rch.txt`: `matmul_flat_compute_rows_row_split_is_bit_identical` passed on `ovh-a`; this also checks the K-major A-panel invariant against the original row-major values.
- `proof_microkernel_golden_rch.txt`: `matmul_microkernel_golden_digest` passed on `ovh-a`.
- `proof_medium_route_golden_rch.txt`: ignored release `matmul_medium_flat_workspace_route_golden_digest` passed on `ovh-a`, digest `0x5fd37bf053d54fb0`.

Isomorphism statement:

- Public shape, rectangularity, and fallback gates are unchanged.
- `packed_b` and scalar-tail B loads are unchanged.
- For each `c[i][j]`, the multiply-add sequence still visits `k = 0..ka` in increasing order.
- The A-panel invariant is `a_panel[k * row_count + row_rel] == a[row_start + row_rel][k]`.
- Thread row ownership remains disjoint; output row order is unchanged.
- No RNG, tie-breaking, or floating-point tolerance surface is introduced.

## Quality Gates

- `cargo fmt -p fsci-linalg --check`: passed.
- `ubs crates/fsci-linalg/src/lib.rs`: exit 0, 0 critical findings; broad existing warning noise remains in the large file.
- RCH `cargo check -j 1 -p fsci-linalg --all-targets --locked`: passed on `ovh-a`.
- RCH `cargo clippy -j 1 -p fsci-linalg --all-targets --locked -- -D warnings`: blocked by existing dependency lint in `fsci-fft` (`manual_is_multiple_of`).
- RCH `cargo clippy -j 1 -p fsci-linalg --all-targets --locked --no-deps -- -D warnings`: blocked by an existing `fsci-linalg` lint outside this GEMM diff at `src/lib.rs:4231` (`manual_memcpy`).

## Candidate Benchmark

Candidate after-run:

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo bench -j 1 -p fsci-linalg --bench linalg_bench --locked -- matmul
```

Worker: `vmi1227854`

| Benchmark | Candidate mean |
| --- | ---: |
| `matmul/256x256` | `5.1824 ms` |
| `matmul/512x512` | `14.883 ms` |
| `matmul/768x768` | `33.870 ms` |
| `matmul/1024x1024` | `185.06 ms` |

The original baseline was on `ovh-a`, so this candidate is not scoreable yet.
RCH refused two forced paired-baseline attempts on `vmi1227854` because the worker was saturated.
Keep/reject is pending a paired baseline on `vmi1227854` or a fresh baseline/candidate pair on another admitted worker.
