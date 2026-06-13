# frankenscipy-8l8r1.103 rejection

Target: `[perf][sparse] reuse MMD neighbor scratch buffer after medium route`.

Profile-backed source:
- Post-`.102` profile artifact:
  `tests/artifacts/perf/2026-06-13-sparse-post-102-reprofile/mmd_ordering_perf_probe_vmi1152480_rch.txt`.
- Current `vmi1152480` rows:
  - `lap2d_k20`: `0.537691 ms`, digest `0x0217ba46fa6e6a05`
  - `lap2d_k32`: `2.327287 ms`, digest `0xffdd6ca421f7bd89`
  - `arrowhead_n1000`: `0.224682 ms`, digest `0xfdd649ab98f97f95`

Rejected lever:
- Reused a single `Vec<usize>` scratch buffer for the sorted small-set MMD
  neighbor list instead of allocating one `Vec` per eliminated node.
- Intended alien primitive: arena/slab scratch reuse for sparse symbolic kernels.
- Source was restored; `crates/fsci-sparse/src/linalg.rs` SHA-256 is back to
  `970c13d153021c9d3eba7ef5885a81e031bd37dd9ce2b798e84cacc5108b96e2`, matching
  `HEAD`.

Same-worker comparison:
- Worker: `vmi1293453`.
- Baseline worktree: `3c024e79`, artifact
  `baseline_mmd_no_scratch_vmi1293453_rch.txt`.
- After artifact: `after_mmd_scratch_vmi1152480_rch.txt` (RCH selected
  `vmi1293453` despite the filename).

Rows:
- `lap2d_k20`: `0.452986 ms -> 0.612677 ms`
- `lap2d_k32`: `2.290531 ms -> 3.136048 ms`
- `arrowhead_n1000`: `0.234837 ms -> 0.309954 ms`

Behavior proof status:
- Order digests stayed unchanged on all rows:
  - `0x0217ba46fa6e6a05`
  - `0xffdd6ca421f7bd89`
  - `0xfdd649ab98f97f95`
- Ordering/tie-breaking/floating-point/RNG would have been preserved, but the
  performance gate failed before keep.

Score:
- Impact: `0.0` because every same-worker row regressed.
- Confidence: `0.90`.
- Effort: `0.25`.
- Score: `0.0`.

Verdict:
- Reject. No source change retained.
- Do not keep iterating on MMD allocation spelling. Next pass should attack a
  deeper sparse symbolic/factorization primitive from the graveyard lane:
  quotient-graph or elimination-tree symbolic analysis, supernodal sparse
  Cholesky/LU panelization, or AMG/Laplacian-preconditioned route for structured
  sparse systems.
