# frankenscipy-8l8r1.87 baseline and proof contract

Bead: `frankenscipy-8l8r1.87`

## Coordination

- Claimed `frankenscipy-8l8r1.87` as `in_progress`.
- Existing `TopazGorge` reservations still cover:
  - `crates/fsci-linalg/src/**`
  - `crates/fsci-linalg/benches/**`
  - `.skill-loop-progress.md`
  - `.beads/**`
- Agent Mail macro registered `RubyWaterfall` for this session and granted:
  - `tests/artifacts/perf/2026-06-11-linalg-8l8r1-87/**`
- The identity split is recorded on the bead; both names are the same current
  operator context for this pass.

## Baseline command

Executed from clean detached worktree:

```text
/data/projects/.scratch/frankenscipy-8l8r1-87-baseline-20260611T1455
HEAD ee28d83473072a6833b8e9f7faced500dfd0b207
```

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 \
  rch exec -- env CARGO_BUILD_JOBS=1 \
  cargo bench -j 1 -p fsci-linalg --bench linalg_bench --locked -- eigh_dense
```

The first three attempts were remote-required refusals because no admissible
worker was available. After refreshing/probing worker capabilities, the
successful baseline used worker `vmi1227854`.

## Baseline results

Artifact:

```text
tests/artifacts/perf/2026-06-11-linalg-8l8r1-87/baseline_eigh_dense_rch_retry3.txt
sha256: fa47dda050c2cf41077822c63f5603a3db676fb10d816ad31457811e8d65f320
worker: vmi1227854
```

Criterion intervals:

| shape | lower | mean | upper |
| --- | ---: | ---: | ---: |
| `eigh_dense/256x256` | `13.394 ms` | `14.260 ms` | `14.878 ms` |
| `eigh_dense/512x512` | `105.39 ms` | `108.63 ms` | `114.83 ms` |

## Public golden proof

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 \
  rch exec -- env CARGO_BUILD_JOBS=1 \
  cargo test -j 1 -p fsci-linalg --lib --locked -- \
    eigh_index_sort_matches_materialized_pair_sort_bits --nocapture
```

Artifact:

```text
tests/artifacts/perf/2026-06-11-linalg-8l8r1-87/proof_public_eigh_golden_rch.txt
sha256: 53ac3e0402d658135459492f112d8dbe7d724e66dbe6b7cff623ef5a6ab4e7dd
worker: vmi1227854
digest: eigh_index_sort_public_golden_digest=0x287a5d3679a8bc6a
```

## Isomorphism contract for the next source lever

- Public validation, finite checks, square-matrix errors, dimension guards, and
  trace behavior must stay unchanged.
- Eigenvalue ordering remains ascending with the same `f64::total_cmp` policy.
- No RNG, unsafe code, or external BLAS/LAPACK/MKL/XLA linkage may be introduced.
- The first compact-WY slice stays behind private proof/perf probes; no public
  `eigh` dispatch until stage evidence clears Score >= 2.0.
- Proof must include `B = Q^T A Q` residual, `Q` orthogonality, outside-band
  zeros, public golden digest stability, and a comparison against the rejected
  scalar-panel route on `512x512`.
- Do not repeat scalar reflector replay or panel-chunked scalar full-to-band.
