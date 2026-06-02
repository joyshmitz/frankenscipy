# Pass 3 - Rcond Cached-State Feasibility

Date: 2026-06-02T02:43:00-0400
Agent: OliveSnow
Target bead: `frankenscipy-perf-linalg-directlu-triangular-solve-v82ao`

## Mission

Pass 2 showed repeated precomputed-LU `lu_solve` calls were dominated by
`fast_rcond_from_lu` in 49/49 timed samples. This pass evaluated exactly one
production lever: compute the existing reciprocal-condition estimate once in
`lu_factor`, store it privately on `LuFactorResult`, and reuse the cached value
from `lu_solve` for the same warning and trace paths.

No estimator algorithm, sign-vector allocation, transpose solve, or final LU
solve logic was changed.

## Change

- Added private `LuFactorResult::rcond_estimate`.
- `lu_factor` computes it with the existing
  `fast_rcond_from_lu(&lu_decomp, a_norm_1, rows)` immediately after
  factorization.
- `lu_solve` uses the cached value instead of recomputing the estimator.
- Replaced derived `Debug` for `LuFactorResult` with a manual implementation
  that preserves the prior visible fields and omits `rcond_estimate`.
- Added `lu_factor_caches_rcond_without_debug_observability` to prove cached
  rcond bits match recomputation, warning propagation is unchanged, and Debug
  output does not expose the new private cache field.

## Baseline

Artifact: `pass3_cached_state_baseline_rch.json`

Command shape:

```bash
RCH_FORCE_REMOTE=1 rch exec -- hyperfine \
  --setup 'env CARGO_TARGET_DIR=/tmp/rch_target_fsci_linalg_tri_pass3_cached_state_baseline_20260602a RUSTFLAGS="-C force-frame-pointers=yes" cargo build -p fsci-linalg --profile release-perf --bin perf_solve --locked' \
  --warmup 3 \
  --runs 10 \
  --export-json tests/artifacts/perf/2026-06-02-linalg-triangular-solve/pass3_cached_state_baseline_rch.json \
  '/tmp/rch_target_fsci_linalg_tri_pass3_cached_state_baseline_20260602a/release-perf/perf_solve lu_factor 1000 1 42' \
  '/tmp/rch_target_fsci_linalg_tri_pass3_cached_state_baseline_20260602a/release-perf/perf_solve lu_solve 1000 1 42' \
  '/tmp/rch_target_fsci_linalg_tri_pass3_cached_state_baseline_20260602a/release-perf/perf_solve lu_solve_cached 1000 200 42'
```

| workload | mean | stddev | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lu_factor 1000 1 42` | 100.486 ms | 5.553 ms | 100.573 ms | 91.910 ms | 110.020 ms |
| `lu_solve 1000 1 42` | 118.635 ms | 7.745 ms | 116.616 ms | 108.211 ms | 132.728 ms |
| `lu_solve_cached 1000 200 42` | 3919.240 ms | 124.981 ms | 3908.202 ms | 3758.875 ms | 4135.931 ms |

Cached baseline per public `lu_solve` call: 19.596 ms/call.

## After

Artifact: `pass3_cached_state_after_rch.json`

| workload | mean | stddev | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lu_factor 1000 1 42` | 117.157 ms | 9.014 ms | 113.846 ms | 107.783 ms | 133.036 ms |
| `lu_solve 1000 1 42` | 118.910 ms | 7.494 ms | 117.933 ms | 109.140 ms | 130.200 ms |
| `lu_solve_cached 1000 200 42` | 159.173 ms | 4.423 ms | 160.297 ms | 148.928 ms | 164.669 ms |

Cached after per public `lu_solve` call: 0.796 ms/call.

## Interpretation

- Repeated precomputed-LU solves improved 24.6x
  (`3919.240 ms / 159.173 ms`).
- `lu_factor` alone regressed by 16.7 ms mean because it now pays one rcond
  estimate at factorization time.
- One-shot `lu_factor + lu_solve` stayed effectively flat:
  `118.635 ms -> 118.910 ms` mean. The same rcond work moved from `lu_solve`
  to `lu_factor`; it was not duplicated.
- This is a good tradeoff for the target workload: one factorization followed
  by repeated public `lu_solve` calls.

## Golden / Isomorphism

Artifacts:

- `pass3_cached_rcond_golden_before.txt`
- `pass3_cached_rcond_golden_before.sha256`
- `pass3_cached_rcond_golden_after.txt`
- `pass3_cached_rcond_golden_after.sha256`
- `pass3_cached_rcond_lu_solve_cached_before.txt`
- `pass3_cached_rcond_lu_solve_cached_before.sha256`
- `pass3_cached_rcond_lu_solve_cached_after.txt`
- `pass3_cached_rcond_lu_solve_cached_after.sha256`

Existing `perf_solve golden` sha256 stayed:

```text
5809995418488c93cc66dc6f2dc01a0d5fd8e2d8faab6f9a7c44241e99025bdd
```

Cached `lu_solve_cached 1000 200 42` canonical checksum sha256 stayed:

```text
999b301ff7191ddd3eacb3e1379c981693484a32889e7566ec812b6c2c9afa35
```

Canonical cached checksum content:

```text
mode=lu_solve_cached n=1000 repeats=200 seed=42 checksum=-1.968966e2
```

Proof:

- Ordering preserved: repeated public `lu_solve` calls still happen in the same
  loop order. The rcond estimator runs once during factorization instead of
  once per solve.
- Tie-breaking unchanged: no solver selection or branch tie-breaking changed.
- Floating-point preserved: `rcond_estimate` is computed by the same function
  on the same immutable LU object and same matrix 1-norm; the focused unit test
  checks cached bits against recomputation. Solution and backward-error golden
  output is byte-identical.
- RNG preserved: deterministic matrix/RHS generators and seed use are
  unchanged.
- Trace/warning observability: `lu_factor` trace content remains
  `rcond: None`; `lu_solve` trace and warning still use the same rcond value,
  now cached. Manual Debug omits the private cache field so a new user-visible
  Debug value is not introduced.
- Golden outputs: `sha256sum -c` passed for all before/after golden and cached
  checksum files; `cmp` showed before/after normalized files are identical.

## Post-Edit Profile

Artifacts:

- `pass3_cached_rcond_profile_after_samples.txt`
- `pass3_cached_rcond_profile_after_counts.txt`
- `pass3_cached_rcond_profile_after_run.txt`

Long sampled run:

```text
{"mode":"lu_solve_cached","n":1000,"repeats":10000,"total_ms":4493.773,"per_call_ms":0.4494,"checksum":-9.844828e3}
```

Reduced counts:

```text
total_sample_blocks=30
attached_sample_blocks=13
unattached_after_process_exit=17
fast_rcond_from_lu=0
lu_solve_or_nalgebra_lu_solve=13
lower_triangular_solve_frames=8
upper_triangular_solve_frames=5
rhs_from_column_slice=0
solution_collect_frames=0
checksum_sum_frames=1
perf_solve_main_frames=13
```

Pass 4 should target the shifted repeated-solve bottleneck: nalgebra final
`LU::solve` triangular solve internals for cached `LuFactorResult` workloads.
RHS construction and solution collection were not sampled.

## Score

| Candidate | Impact | Confidence | Effort | Score | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Cache existing rcond estimate in `LuFactorResult` | 5.0 | 4.5 | 1.5 | 15.0 | Keep |

The score is against the honest repeated-solve target workload. The `lu_factor`
standalone regression is documented above.

## Validation

Passed:

- `RCH_FORCE_REMOTE=1 rch exec -- cargo check -p fsci-linalg --bin perf_solve --locked`
- `RCH_FORCE_REMOTE=1 rch exec -- cargo test -p fsci-linalg lu_factor_caches_rcond_without_debug_observability --lib --locked -- --nocapture`
- `RCH_FORCE_REMOTE=1 rch exec -- cargo clippy -p fsci-linalg --all-targets --locked -- -D warnings`
- `jq empty pass3_cached_state_baseline_rch.json`
- `jq empty pass3_cached_state_after_rch.json`
- `sha256sum -c` for all pass-3 golden/checksum files
- `git diff --check` for scoped paths

Blocked / existing drift:

- `rustfmt --check crates/fsci-linalg/src/lib.rs` exited 1. The captured diff
  in `pass3_cached_rcond_rustfmt_check.txt` includes broad pre-existing
  formatting drift outside this pass; no broad formatting was applied.
