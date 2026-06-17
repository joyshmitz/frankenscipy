# frankenscipy-wm1gg - Compact Lane-Local DSBTRD Bulge Chase Baseline

Date: 2026-06-17
Agent: RubyWaterfall
Target: `eig_banded(lower=true, eigvals_only=false)`
Worker: local only, per ts1/RCH-offline override after `rch exec` timed out while syncing for predecessor `frankenscipy-psn7x.19`

## Reason

`frankenscipy-psn7x.19` closed rejected/no-ship after two proofed routes:

- lower-storage/mirror cleanup was behavior-preserving but flat (`205.5 ms +/- 9.6 ms -> 206.0 ms +/- 10.3 ms`)
- dense adjacent-Givens DSBTRD was numerically valid but much slower (`205.5 ms +/- 9.6 ms -> 267.6 ms +/- 32.0 ms`)

This successor attacks the algorithmic gap left by those failures: compact diagonal/band-lane bulge chasing with direct Q metadata replay, avoiding dense full-row/full-column rotations.

## Baseline

Command:

```bash
env RCH_REQUIRE_REMOTE=0 CARGO_TARGET_DIR=/data/projects/.scratch/frankenscipy-rubywaterfall-psn7x19-target \
  hyperfine --warmup 1 --runs 5 --show-output \
  'cargo test -j 1 -p fsci-linalg --lib eig_banded_eigenvectors_perf_probe --release --locked -- --ignored --nocapture --test-threads=1'
```

Transcripts:

- `baseline_probe_local.txt`
- `baseline_hyperfine_local.txt`

Result:

- Wall: `198.7 ms +/- 5.6 ms`
- 128x128 bw32 candidate range: `3.062364-4.340820 ms`
- 256x256 bw32 candidate range: `12.771252-16.866366 ms`
- Residuals: `1.64845914696343243e-12` at 128 and `7.73070496506989002e-12` at 256
- Values digests: `0xd6dbb9200f65bd92` and `0x09ed4d367faab431`
- Vector digests: `0x6cf3573b5b50c275` and `0xc32797c0d224a75a`

## Required Next Lever

Allowed:

- compact diagonal/band-lane DSBTRD-style bulge chase
- direct Q metadata replay into eigenvector rows/columns
- dense oracle residual, orthogonality, ordering, and deterministic no-RNG proof

Do not retry:

- lower-storage native entrypoints or upper-mirror/clone cleanup
- dense full-row/full-column adjacent rotations
- direct-index packet wrappers
- full active lower-envelope storage
- fixed envelope width guesses
- Lanczos vectors
- shifted inverse iteration over band solves
- worker-count retuning
- raw/stale compact-WY panels
- scalar spelling or SIMD rank-2 spelling

## Pass 2 Reject: Fixed-Width Compact Lane Route

Candidate:

- Added a production-facing compact lane envelope route for `eig_banded(lower=true, eigvals_only=false)`.
- The route used the kept diagonal-band adjacent rotation formulas, selected actual Givens rotations from live band lanes, and replayed rotation metadata into tridiagonal eigenvectors.
- The lane envelope was fixed at `bandwidth + 1` and returned `None` to fall back when a bulge escaped the explicit lanes.

Proof:

- `after_compact_lane_probe.txt`: public `eig_banded_eigenvectors_perf_probe` passed.
- Public values/vectors digests and residuals were unchanged, proving fallback preserved behavior:
  - 128x128 values `0xd6dbb9200f65bd92`, vectors `0x6cf3573b5b50c275`, residual `1.64845914696343243e-12`
  - 256x256 values `0x09ed4d367faab431`, vectors `0xc32797c0d224a75a`, residual `7.73070496506989002e-12`

Rebench:

- Baseline: `198.7 ms +/- 5.6 ms`
- Candidate: `202.7 ms +/- 12.1 ms` (`after_compact_lane_hyperfine.txt`)
- The unchanged digests plus the slower wall time show the compact route fell back and only added failed-attempt overhead.

Score:

- `Impact 0.0 * Confidence 5.0 / Effort 2.0 = 0.0`
- Source restored; `git diff -- crates/fsci-linalg/src/lib.rs` is empty after restore.

Route:

- Do not retry a fixed `bandwidth + 1` lane envelope that falls back on escaped bulges.
- Next primitive needs explicit bulge storage/chase queues or a wider adaptive envelope that keeps the rotation path inside compact storage without dense row/column updates.

## Pass 3 Reject: Adaptive Wider Lane Envelope

Candidate:

- Retried the compact lane route with an adaptive envelope width of `2 * bandwidth + 1`.
- The intent was to leave explicit room for a chased bulge while still avoiding dense full-row/full-column rotations.

Proof:

- `after_adaptive_lane_probe.txt`: public `eig_banded_eigenvectors_perf_probe` passed.
- Public values/vectors digests and residuals were unchanged, proving the adaptive route still fell back:
  - 128x128 values `0xd6dbb9200f65bd92`, vectors `0x6cf3573b5b50c275`, residual `1.64845914696343243e-12`
  - 256x256 values `0x09ed4d367faab431`, vectors `0xc32797c0d224a75a`, residual `7.73070496506989002e-12`

Rebench:

- Baseline: `198.7 ms +/- 5.6 ms`
- Candidate: `214.0 ms +/- 7.5 ms` (`after_adaptive_lane_hyperfine.txt`)
- Result: fallback preserved behavior but added overhead.

Score:

- `Impact 0.0 * Confidence 5.0 / Effort 2.0 = 0.0`
- Source restored; `git diff -- crates/fsci-linalg/src/lib.rs` is empty after restore.

Route:

- Do not retry blind envelope widening.
- Next primitive must explicitly represent and chase escaped bulges instead of depending on a fixed/adaptive lane width to contain them.

## Closeout

`frankenscipy-wm1gg` was closed rejected/no-ship after the fixed-width and adaptive-width compact lane attempts both fell back and regressed.

Successor: `frankenscipy-nvsct` for explicit bulge storage/chase queues with direct Q metadata replay.
