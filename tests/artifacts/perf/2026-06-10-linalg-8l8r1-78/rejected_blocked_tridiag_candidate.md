# frankenscipy-8l8r1.78 rejected lever: test-only compact-WY tridiagonal candidate

Target: dense `eigh` compact-WY symmetric tridiagonalization route.

Baseline evidence:
- Command: `RCH_REQUIRE_REMOTE=1 CARGO_BUILD_JOBS=1 rch exec -- env CARGO_BUILD_JOBS=1 cargo bench -j 1 -p fsci-linalg --bench linalg_bench -- eigh_dense --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2`
- Worker: `vmi1227854`
- `eigh_dense/256x256`: `[15.724 ms 16.156 ms 16.455 ms]`
- `eigh_dense/512x512`: `[120.10 ms 121.65 ms 122.71 ms]`
- Public golden proof: `eigh_index_sort_public_golden_digest=0x287a5d3679a8bc6a`

Rejected lever:
- Added a test-only blocked symmetric tridiagonalization candidate with pending compact-WY-style panel updates.
- Proved small reconstruction cases first on RCH `vmi1227854`: `blocked_tridiagonal_candidate_reconstructs_small_eigh_cases` passed.
- Release probe on RCH `vmi1227854`:
  - `256x256`: public `73.566241 ms`, candidate `207.835509 ms`, speedup `0.353964x`
  - `512x512`: public `242.733268 ms`, candidate `864.807165 ms`, speedup `0.280679x`
  - candidate raw public bits equal: `false` at both sizes

Decision:
- Rejected. Score is below threshold: Impact `0` x Confidence `0.95` / Effort `2` = `0.0`.
- No public route changed. The test-only candidate was removed after measurement.

Isomorphism notes:
- Public route was never modified, so ordering, `f64::total_cmp` tie behavior, sign behavior, RNG behavior, and public golden digest remain governed by the existing implementation.
- Candidate reconstruction and orthogonality were acceptable, but raw public-output digest differed, so it was not behavior-isomorphic enough to route publicly.

Next primitive:
- Do not repeat the same pending-update formulation. Attack a deeper communication-avoiding symmetric band-reduction primitive with contiguous panel storage and direct bulge-chase stage timings, or replace the public decomposition backend around a banded compact storage path whose raw-output fallback preserves the existing public digest.
