# frankenscipy-8l8r1.109 sparse symbolic clique update rejection

## Target

- Bead: `frankenscipy-8l8r1.109`
- Crate: `fsci-sparse`
- Worker: `vmi1152480`
- Candidate: replace pairwise sorted-vector MMD clique insertion with a batched sorted clique-union update for eliminations with more than two active neighbors.

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo test -p fsci-sparse --lib --release --locked minimum_degree_ordering_perf_probe -- --ignored --nocapture
```

Rows:

- `lap2d_k20`: `0.631591 ms`, digest `0x0217ba46fa6e6a05`
- `lap2d_k32`: `3.368014 ms`, digest `0xffdd6ca421f7bd89`
- `arrowhead_n1000`: `0.291942 ms`, digest `0xfdd649ab98f97f95`

## Candidate

First after attempt failed RCH remote dependency preflight and refused local fallback, so it is not timing evidence.

Same-worker retry rows:

- `lap2d_k20`: `5.839658 ms`, digest `0x0217ba46fa6e6a05`
- `lap2d_k32`: `2.845249 ms`, digest `0xffdd6ca421f7bd89`
- `arrowhead_n1000`: `0.287189 ms`, digest `0xfdd649ab98f97f95`

## Verdict

Rejected. The candidate preserved the ordering digest and improved the larger `lap2d_k32` probe, but regressed the smaller structured `lap2d_k20` probe by about `9.25x`. Score: `0.0`; not keepable.

The source hunk was restored. `crates/fsci-sparse/src/linalg.rs` SHA-256 after restore:

```text
970c13d153021c9d3eba7ef5885a81e031bd37dd9ce2b798e84cacc5108b96e2
```

## Next route

Do not repeat sorted-vector pairwise insertion, allocation/scratch reuse, threshold tuning, or batched Vec-merge clique construction for this MMD path. The next sparse pass should move to a fundamentally different symbolic primitive: quotient-graph/elimination-tree state, supernodal symbolic analysis, or an AMG/Laplacian-preconditioned route for structured sparse solves.
