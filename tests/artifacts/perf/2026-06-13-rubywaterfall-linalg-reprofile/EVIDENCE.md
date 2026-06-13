# 2026-06-13 linalg matmul 256 flat-workspace threshold

- Bead: `frankenscipy-hgj8i`
- Skill loop: `repeatedly-apply-skill` -> `extreme-software-optimization`
- Target: `fsci-linalg` `matmul/256x256`
- Lever: lower only `MATMUL_FLAT_WORKSPACE_MIN_DIM` from 512 to 256, routing the measured 256x256 GEMM case through the existing flat packed SIMD workspace.

## Baseline

RCH worker `vmi1227854`, current route:

```text
matmul/256x256 time: [4.8540 ms 5.0983 ms 5.2820 ms]
```

Artifact: `baseline_matmul_current_rch.txt`

## After

RCH worker `vmi1227854`, threshold route:

```text
matmul/256x256 time: [4.0516 ms 4.3733 ms 4.6220 ms]
```

Artifact: `after_matmul_256_threshold_vmi1227854_rch.txt`

Midpoint speedup: `5.0983 / 4.3733 = 1.1658x`.

## Behavior proof

- Ordering and floating-point surface: both public route and helper remain k-monotonic; the candidate test compares every output bit against the naive i-j-k reference for the 256x256 dispatch case.
- Tie-breaking: no ties or ordering decisions are introduced by this GEMM dispatch threshold.
- RNG: no random surface is touched.
- Golden output: public route digest is `0x6e401fad043ac8fd`.

Proof artifacts:

- `proof_matmul_flat_workspace_256_route_rch.txt`
- `proof_matmul_256_public_route_golden_rch.txt`
- `proof_matmul_flat_workspace_256_route_after_fix_rch.txt`
- `proof_matmul_256_public_route_golden_after_fix_rch.txt`

The first clippy pass exposed a test-only `needless_range_loop` lint in the SPD golden helper. That helper was rewritten with `split_at_mut`/`enumerate` to preserve the same symmetric matrix values without changing the optimization lever.

## Keep score

Score = `(Impact 2.0 * Confidence 4.0) / Effort 1.0 = 8.0`.

The lever clears the required Score >= 2.0 gate and keeps the optimization in the already-proven flat-workspace implementation rather than introducing a new algorithmic surface.

## Validation

- `rch exec -- cargo test -j 1 -p fsci-linalg matmul_flat_workspace_is_bit_identical_to_naive_ijk -- --nocapture`
- `rch exec -- cargo test -j 1 -p fsci-linalg --release matmul_medium_flat_workspace_route_golden_digest -- --ignored --nocapture`
- `rch exec -- cargo check -j 1 -p fsci-linalg --all-targets --locked`
- `rch exec -- cargo clippy -j 1 -p fsci-linalg --all-targets --no-deps --locked -- -D warnings` (`clippy_fsci_linalg_all_targets_no_deps_after_fix_rch.txt`)
- `rustfmt --edition 2024 --check crates/fsci-linalg/src/lib.rs`
- `git diff --check -- crates/fsci-linalg/src/lib.rs tests/artifacts/perf/2026-06-13-rubywaterfall-linalg-reprofile`
- `ubs crates/fsci-linalg/src/lib.rs .skill-loop-progress.md`

## Post-keep reprofile

Attempted same-worker broad `matmul` Criterion reprofile:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo bench -j 1 -p fsci-linalg --bench linalg_bench -- matmul --sample-size 10
```

Artifact: `post_keep_matmul_reprofile_vmi1227854_rch.txt`.

The command failed before sampling because the remote build reported a missing tracked bin source, `crates/fsci-linalg/src/bin/diff_matfuncs.rs`. The file exists locally and is clean in git; this failure is recorded as an RCH remote-state/sync issue, not as keep/reject evidence for the 256x256 threshold lever.
