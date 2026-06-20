# Performance Release-Readiness Scorecard

## 2026-06-20 - fsci-cluster linkage row-pack gauntlet

- Agent: cod-a / BlackThrush
- Bead: `frankenscipy-8l8r1.128`
- Decision: KEEP the observation-row packing lever; REJECT AND REVERT lazy
  full-arena zero initialization. The final source closes part of the Ward
  gap but remains a strict `0/2/0` SciPy loss, so release-readiness remains
  partial for this linkage cluster.
- Artifact:
  `tests/artifacts/perf/2026-06-20-cod-a-linkage-lazy-arena-EVIDENCE.md`

| Gate | Result | Notes |
| --- | --- | --- |
| rch Criterion attempt | BLOCKED | `cargo bench -p fsci-cluster --bench cluster_bench -- va60h_gauntlet_linkage --noplot` ran Rust rows on `hz1` but failed when the worker Python could not import `scipy` |
| Local SciPy baseline | PASS | local SciPy 1.17.1, exact crate-scoped Criterion group: Average 7.1834 ms vs SciPy 5.0346 ms, Ward 8.2387 ms vs SciPy 5.3080 ms |
| Rejected candidate A/B | FAIL | lazy zero arena: Average 7.6203 ms, 1.061x slower than current; Ward 8.2002 ms, neutral 1.005x faster than current; source reverted |
| Final row-pack A/B | PASS | Average 7.1304 ms, 1.007x faster than baseline; Ward 6.9591 ms, 1.184x faster than baseline |
| Final SciPy oracle | LOSS | same-run SciPy 1.17.1: Average 4.3843 ms and Ward 4.8687 ms; final Rust remains 1.626x / 1.429x slower |
| Isomorphism harness | PASS | `cargo run --release -p fsci-cluster --bin perf_linkage`: 0 mismatches / 7200 linkage matrices |
| Focused linkage contract test | PASS | `cargo test -p fsci-cluster linkage_flat_core_matches_precomputed_condensed_contract -- --nocapture` via rch |
| Broader linkage filter | PASS | `cargo test -p fsci-cluster linkage_ -- --nocapture` via rch |
| Per-crate release build | PASS | `cargo build --release -p fsci-cluster` via rch; unrelated warning remained in `perf_kmeans.rs` |
| No-deps clippy | PASS | `cargo clippy -p fsci-cluster --lib --no-deps -- -D warnings` via rch |
| Changed-file UBS | PASS | `ubs crates/fsci-cluster/src/lib.rs ...` exited 0 with 0 critical findings; existing warning inventory remains |
| Dependency-inclusive clippy | BLOCKED | `cargo clippy -p fsci-cluster --lib -- -D warnings` stops before this patch on existing `fsci-linalg` lints |
| File formatting | BLOCKED | `rustfmt --edition 2024 --check crates/fsci-cluster/src/lib.rs` reports pre-existing file-wide drift outside this scoped patch |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Baseline Rust `linkage(Average)` | 7.1834 ms | 1.427x slower than SciPy baseline | prior current |
| Final Rust `linkage(Average)` | 7.1304 ms | 1.007x faster than baseline; 1.626x slower than same-run SciPy | neutral/internal keep |
| SciPy `linkage(Average)` | 4.3843 ms | 1.00x final oracle | reference |
| Baseline Rust `linkage(Ward)` | 8.2387 ms | 1.552x slower than SciPy baseline | prior current |
| Final Rust `linkage(Ward)` | 6.9591 ms | 1.184x faster than baseline; 1.429x slower than same-run SciPy | internal keep |
| SciPy `linkage(Ward)` | 4.8687 ms | 1.00x final oracle | reference |
| Lazy arena Average candidate | 7.6203 ms | 1.061x slower than baseline; 1.690x slower than SciPy | reject |
| Lazy arena Ward candidate | 8.2002 ms | neutral 1.005x faster than baseline; 1.560x slower than SciPy | reject/no ship |

Readiness notes:

- Packing nested observations once reduced repeated row indirection in the
  pairwise distance frontier without changing linkage semantics.
- Arena fill elision is not worth retrying on this route. Future work should
  target NN-chain/method-specific nearest-neighbour maintenance or another
  algorithmic primitive that changes the amount of scanned state.

## 2026-06-20 - fsci-ndimage EDT feature-transform line-start gauntlet

- Agent: cod-b / MistyBirch
- Bead: `frankenscipy-8l8r1.127`
- Decision: KEEP. Exact separable line-start enumeration plus flat 2-D index
  materialization improves every `return_indices` row on the same rch worker
  versus the prior feature-transform route. Release-readiness remains partial:
  strict SciPy score is `1/3/0`, with a win at 192x192 and losses at
  64/128/256.
- Artifact:
  `tests/artifacts/perf/frankenscipy-8l8r1.127-edt-line-starts-EVIDENCE.md`

| Gate | Result | Notes |
| --- | --- | --- |
| rch same-worker baseline | PASS | prior feature-transform route on `vmi1152480`: 325.742 us / 1.380 ms / 3.814 ms / 5.854 ms for 64/128/192/256 `return_indices` rows |
| rch same-worker final | PASS | final source on `vmi1152480`: 216.733 us / 1.207 ms / 2.107 ms / 4.855 ms; internal speedups 1.50x / 1.14x / 1.81x / 1.21x |
| SciPy oracle | PASS | local SciPy 1.17.1 via `docs/perf_oracle_edt_indices.py --reps 20`: 173.434 us / 775.685 us / 2.280155 ms / 4.288605 ms |
| Isomorphism harness | PASS | `perf_edt` printed `0 mismatches / 10876 cells` on both baseline and final runs |
| Focused EDT tests | PASS | `cargo test -p fsci-ndimage distance_transform_edt --lib -- --nocapture` via rch: 15 passed / 0 failed |
| Full ndimage lib tests | PASS | `cargo test -p fsci-ndimage --lib -- --nocapture` via rch: 242 passed / 0 failed |
| Per-crate compile | PASS | `cargo check -p fsci-ndimage --all-targets` via rch `hz1`; unrelated warnings remained in `fsci-interpolate` and `diff_geom` |
| Local live SciPy conformance | PASS | `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test diff_ndimage_distance_transform_edt -- --nocapture`: 1 passed / 0 failed |
| Touched-file formatting | PASS | `rustfmt --edition 2024 --check crates/fsci-ndimage/src/lib.rs` |
| Diff hygiene | PASS | `git diff --check -- crates/fsci-ndimage/src/lib.rs` |
| Full crate formatting | BLOCKED | `cargo fmt -p fsci-ndimage --check` is blocked by pre-existing drift in `ndimage_bench.rs` and `diff_fourier.rs` |
| Clippy `-D warnings` | BLOCKED | `cargo clippy -p fsci-ndimage --all-targets -- -D warnings` stops before this patch on existing `fsci-linalg` dependency lints |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust final `return_indices`, 64x64 | 216.733 us | 1.50x faster than prior; 1.25x slower than SciPy | internal keep, SciPy loss |
| SciPy `distance_transform_edt(return_indices=True)`, 64x64 | 173.434 us | 1.00x oracle | reference |
| Rust final `return_indices`, 128x128 | 1.207 ms | 1.14x faster than prior; 1.56x slower than SciPy | internal keep, SciPy loss |
| SciPy `distance_transform_edt(return_indices=True)`, 128x128 | 775.685 us | 1.00x oracle | reference |
| Rust final `return_indices`, 192x192 | 2.107 ms | 1.81x faster than prior; 1.08x faster than SciPy | internal keep, SciPy win |
| SciPy `distance_transform_edt(return_indices=True)`, 192x192 | 2.280155 ms | 1.00x oracle | reference |
| Rust final `return_indices`, 256x256 | 4.855 ms | 1.21x faster than prior; 1.13x slower than SciPy | internal keep, SciPy loss |
| SciPy `distance_transform_edt(return_indices=True)`, 256x256 | 4.288605 ms | 1.00x oracle | reference |

Readiness notes:

- The useful lever was eliminating dead flat-index scans and per-cell coordinate
  allocation, not changing EDT semantics. The nearest-background feature index
  propagation and tie behavior stay on the existing exact transform.
- Future EDT work should go below this layer: feature-index scratch layout,
  fused/tiled axis passes, SIMD-friendly lower-envelope kernels, or a
  size-specialized 2-D feature transform.

## 2026-06-20 - fsci-ndimage gaussian_filter inner1 reflect reject

- Agent: cod-a / MistyBirch
- Bead: `frankenscipy-6l77z`
- Decision: REJECT AND REVERT. The row-contiguous reflect/origin-zero direct
  interior dot specialization was intended to remove boundary-branch overhead in
  the final gaussian pass, but same-worker rch Criterion showed a regression.
  Release-readiness score for this attempted lever is `0/1/0` against the
  restored current Rust route and `0/1/0` against SciPy.
- Artifact:
  `tests/artifacts/perf/2026-06-20-ndimage-gaussian-inner1-reflect-reject/EVIDENCE.md`

| Gate | Result | Notes |
| --- | --- | --- |
| rch same-worker baseline | PASS | current `gaussian_sigma2/256` on `hz2`: 3.4399 ms mean, [3.3426, 3.5375] ms |
| rch same-worker candidate | FAIL | candidate `gaussian_sigma2/256` on `hz2`: 4.0213 ms mean, [3.8424, 4.1989] ms; Criterion reported +16.902% regression |
| SciPy oracle | PASS | local SciPy 1.17.1 `gaussian_filter sigma=2 256x256`: 1.13557 ms p50 |
| Candidate revert | PASS | no production code from the attempted fast path is staged |
| Focused gaussian guard | PASS | `cargo test -p fsci-ndimage gaussian_filter1d_matches_scipy_axis1_reflect --lib -- --nocapture` via rch `hz2`: 1 passed / 0 failed |
| Local live SciPy conformance | PASS | `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test diff_ndimage -- --nocapture`: 5 passed / 0 failed |
| Full clippy/check/fmt | NOT RUN | no production code was kept; existing shared `fsci-ndimage/src/lib.rs` dirt belongs to a separate EDT worker and is not staged here |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Restored current Rust `gaussian_sigma2/256` | 3.4399 ms | 3.03x slower than SciPy | retained current |
| Candidate inner1 reflect direct interior dot | 4.0213 ms | 1.17x slower than current; 3.54x slower than SciPy | reject |
| SciPy `ndimage.gaussian_filter` sigma=2 256x256 | 1.13557 ms | 1.00x oracle | reference |

Readiness notes:

- The final-axis row-contiguous loop is not the isolated bottleneck enough to
  justify scalar border/interior splitting. It increased overhead despite using
  direct in-bounds indexing for the interior.
- Future gaussian work should move one level deeper: transpose/cache-tile the
  separable passes so both axes can run contiguous, or build a vector-friendly
  shared dot kernel and prove it against this restored route.

## 2026-06-20 - fsci-ndimage label mean one-based contiguous gauntlet

- Agent: cod-b / MistyBirch
- Bead: `frankenscipy-8l8r1.126`
- Decision: KEEP. The one-based contiguous index fast path is a measured
  internal win over the prior dense table and turns the smallest label-mean row
  into a SciPy win. The sub-cluster remains mixed at strict `1/3/0` against
  SciPy, with the largest row near parity at 1.03x slower.
- Artifact:
  `tests/artifacts/perf/frankenscipy-8l8r1.126-label-mean-one-based-EVIDENCE.md`

| Gate | Result | Notes |
| --- | --- | --- |
| rch release A/B | PASS | `cargo run --release -p fsci-ndimage --bin perf_label_stats` on `hz2`: one_based is 1.08x / 1.08x / 1.36x / 1.33x faster than reconstructed dense_table, `mism=0/0/0/0/0` |
| Same-host SciPy head-to-head | PASS | transferred rch release binary vs local SciPy 1.17.1; final source is 1.23x faster on N=65536 K=512 and 1.09x / 1.19x / 1.03x slower on the remaining rows |
| Focused one-based test | PASS | `cargo test -p fsci-ndimage mean_one_based_contiguous_lookup_preserves_exact_label_semantics --lib -- --nocapture` via rch `hz2`: 1 passed / 0 failed |
| Full ndimage lib tests | PASS | `cargo test -p fsci-ndimage --lib -- --nocapture` via rch `hz2`: 242 passed / 0 failed |
| Per-crate compile | PASS | `cargo check -p fsci-ndimage --all-targets` via rch `hz1`; unrelated warnings remain in `fsci-interpolate` and `diff_geom` |
| Local live SciPy conformance | PASS | `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test diff_ndimage -- --nocapture`: 5 passed / 0 failed |
| Touched-file formatting | PASS | `rustfmt --edition 2024 --check crates/fsci-ndimage/src/lib.rs crates/fsci-ndimage/src/bin/perf_label_stats.rs` |
| Diff hygiene | PASS | `git diff --check -- crates/fsci-ndimage/src/lib.rs crates/fsci-ndimage/src/bin/perf_label_stats.rs` |
| Changed-file UBS scan | PASS | `ubs` exited 0 with 0 critical issues; broad warnings remain inventory |
| Clippy `-D warnings` | BLOCKED | `cargo clippy -p fsci-ndimage --all-targets -- -D warnings` stopped before this patch on existing `fsci-linalg` dependency lints (`needless_range_loop`, `needless_borrow`) |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust one_based mean, N=65536 K=512 | 153.257 us | 1.23x faster than SciPy; 1.01x faster than dense_table on same host | SciPy win, internal keep |
| SciPy `ndimage.mean`, N=65536 K=512 | 0.189 ms | 1.00x oracle | reference |
| Rust one_based mean, N=262144 K=1024 | 634.996 us | 1.09x slower than SciPy; 1.02x faster than dense_table on same host | SciPy loss, internal keep |
| SciPy `ndimage.mean`, N=262144 K=1024 | 0.585 ms | 1.00x oracle | reference |
| Rust one_based mean, N=262144 K=2048 | 687.054 us | 1.19x slower than SciPy; 1.25x faster than dense_table on same host | SciPy loss, internal keep |
| SciPy `ndimage.mean`, N=262144 K=2048 | 0.576 ms | 1.00x oracle | reference |
| Rust one_based mean, N=589824 K=4096 | 1.423 ms | 1.03x slower than SciPy; 1.25x faster than dense_table on same host | SciPy near-parity loss, internal keep |
| SciPy `ndimage.mean`, N=589824 K=4096 | 1.380 ms | 1.00x oracle | reference |

Readiness notes:

- The profitable lever was not another scalar label classifier tweak; it was
  specializing the common one-based label-index contract and eliminating a
  short-lived dense table from the hot reduction path.
- Future work should target parallel/cache-tiled accumulation or sorted spans.

## 2026-06-20 - fsci-cluster linkage triangular-arena reject

- Agent: cod-a / MistyBirch
- Bead: `frankenscipy-va60h`
- Decision: REJECT and REVERT. The compact upper-triangular inter-cluster arena
  preserved exact linkage outputs but lost wall time versus the retained flat
  full arena. Release-readiness score for this attempted lever is `0/2/0`
  against SciPy and `0/2/0` against the restored current route.
- Artifact:
  `tests/artifacts/perf/2026-06-20-va60h-triangular-arena-reject-EVIDENCE.md`

| Gate | Result | Notes |
| --- | --- | --- |
| Local same-machine SciPy gauntlet | PASS | `va60h_gauntlet_linkage` ran Rust current/candidate plus SciPy oracle in one Criterion group |
| rch exact-output harness | PARTIAL | remote command printed `0 mismatches / 7200 linkage matrices`; rch exited 102 afterward because artifact retrieval timed out |
| Focused linkage unit guard | PASS | `cargo test -p fsci-cluster linkage_from_distances --lib -- --nocapture`: 2 passed / 0 failed |
| Local live SciPy conformance | PASS | `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test diff_cluster_linkage_from_distances -- --nocapture`: 1 passed / 0 failed |
| Candidate revert | PASS | `crates/fsci-cluster/src/lib.rs` restored to the prior flat full arena before staging |
| rch SciPy rows | BLOCKED | worker Python could not import SciPy; local SciPy 1.17.1 supplied head-to-head rows |
| Final code delta | PASS | no production code change remains from the rejected lever |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Restored flat `linkage(Average)`, n=800 d=4 | 7.5772 ms | 1.723x slower than same-run SciPy baseline | retained current |
| Triangular candidate `linkage(Average)`, n=800 d=4 | 8.8260 ms | 2.064x slower than SciPy; 1.165x slower than current | reject |
| SciPy `linkage(method="average")`, n=800 d=4 | 4.2755 ms | 1.00x oracle | reference |
| Restored flat `linkage(Ward)`, n=800 d=4 | 7.4597 ms | 1.480x slower than same-run SciPy baseline | retained current |
| Triangular candidate `linkage(Ward)`, n=800 d=4 | 9.9240 ms | 1.809x slower than SciPy; 1.330x slower than current | reject |
| SciPy `linkage(method="ward")`, n=800 d=4 | 5.4866 ms | 1.00x oracle | reference |

Readiness notes:

- Halving the arena storage did not help this path because the row-major full
  matrix already gives cheap row scans; triangular indexing adds arithmetic and
  less regular accesses during merged-cluster updates.
- Future `fsci-cluster` linkage work should move to algorithmic specializations
  rather than another full-square arena layout tweak.

## 2026-06-20 - fsci-special jnjnp_zeros Cephes j1 gauntlet

- Agent: cod-b / MistyBirch
- Bead: `frankenscipy-wh8ac`
- Decision: KEEP. Replacing the local `j1_core` series/asymptotic split with
  the Cephes fixed rational evaluator turns the latest `jnjnp_zeros` near-parity
  loss into a measured SciPy win. Score for this sub-cluster is now `2/0/0`
  against SciPy.
- Artifact:
  `tests/artifacts/perf/2026-06-20-wh8ac-jnjnp-cephes-j1/EVIDENCE.md`

| Gate | Result | Notes |
| --- | --- | --- |
| Same-worker rch A/B | PASS | `acoco_gauntlet_jnjnp_zeros` on rch `vmi1149989`: `nt=64` 608.21 us -> 381.89 us, `nt=128` 1.1970 ms -> 742.06 us |
| SciPy head-to-head oracle | PASS | local SciPy 1.17.1: `nt=64` 463.913 us, `nt=128` 832.786 us; Rust final is 1.22x / 1.12x faster |
| Focused `jnjnp` tests | PASS | `cargo test -p fsci-special jnjnp -- --nocapture` via rch `hz1`: 3 passed / 0 failed |
| Focused `j1` test | PASS | `cargo test -p fsci-special j1_matches_scipy_reference_values -- --nocapture` via rch `ovh-a`: 1 passed / 0 failed |
| Focused `kve` cleanup guard | PASS | `cargo test -p fsci-special complex_kve_matches_scipy -- --nocapture` via rch `hz1`: 1 passed / 0 failed |
| Local live SciPy conformance | PASS | `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test diff_special_bessel_zeros -- --nocapture`: 1 passed / 0 failed |
| Per-crate compile | PASS | `cargo check -p fsci-special --all-targets` via rch `hz1`; existing warnings only |
| Diff hygiene | PASS | `git diff --check` |
| Changed-file UBS | PASS | `ubs` on changed files: 0 critical issues after removing a pre-existing test-only `panic!` in touched `bessel.rs`; warnings remain inventory |
| rch SciPy rows | PARTIAL | rch Criterion workers could not import `scipy.special`; local SciPy oracle supplied the head-to-head |
| Formatting | BLOCKED | `cargo fmt -p fsci-special --check` remains blocked by pre-existing file-wide rustfmt drift outside this patch |
| Clippy `-D warnings` | BLOCKED | `cargo clippy -p fsci-special --all-targets -- -D warnings` stops in existing `fsci-integrate` and `fsci-linalg` lints before this patch |

| Workload / route | Final Rust | SciPy oracle | Ratio | Verdict |
| --- | ---: | ---: | ---: | --- |
| `jnjnp_zeros(nt=64)` | 381.89 us | 463.913 us | 1.22x faster | SciPy win |
| `jnjnp_zeros(nt=128)` | 742.06 us | 832.786 us | 1.12x faster | SciPy win |
| Prior local J1 series/asymptotic route, `nt=64` | 608.21 us | - | current is 1.59x faster | superseded |
| Prior local J1 series/asymptotic route, `nt=128` | 1.1970 ms | - | current is 1.61x faster | superseded |

Readiness notes:

- The profitable lever was matching the incumbent Cephes fixed rational J1
  evaluator, not another frontier or top-k selection tweak.
- Future `jnjnp_zeros` work should profile the remaining root-generation path
  before trying another Bessel approximation variant.

## 2026-06-20 - fsci-ndimage zoom order=1 no-prefilter gauntlet

- Agent: cod-b / MistyBirch
- Bead: `frankenscipy-oi8hq`
- Decision: KEEP. The direct-original-image order-1 reflect zoom path closes
  the previous `frankenscipy-wm14d` residual SciPy loss. Score for this
  sub-cluster is now `1/0/0` against SciPy.
- Artifact:
  `tests/artifacts/perf/frankenscipy-oi8hq-zoom-order1-no-prefilter-EVIDENCE.md`

| Gate | Result | Notes |
| --- | --- | --- |
| rch release benchmark | PASS | final source `zoom/2x_256/1` on rch `vmi1149989`: 1.2219 ms mean (`[1.0624, 1.2219, 1.5189] ms`) |
| Baseline rerun | PASS | prepatch residual current on rch `hz2`: 8.8419 ms mean (`[7.5432, 8.8419, 9.7257] ms`); cross-worker routing comparison only |
| SciPy head-to-head oracle | PASS | local SciPy 1.17.1 `ndimage.zoom(256x256, 2x, order=1)` median 4.86171 ms; Rust final is 3.98x faster |
| Focused zoom tests | PASS | `cargo test -p fsci-ndimage zoom_ --lib -- --nocapture` via rch `ovh-a`: 6 passed / 0 failed |
| Local live SciPy conformance | PASS | `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test diff_ndimage_zoom -- --nocapture`: 1 passed / 0 failed |
| Per-crate compile | PASS | `cargo check -p fsci-ndimage --all-targets` via rch `hz1`; unrelated dependency/bin warnings remain |
| Touched-file formatting | PASS | `rustfmt --edition 2024 --check crates/fsci-ndimage/src/lib.rs` |
| Diff/UBS hygiene | PASS | `git diff --check` passed; changed-file `ubs` exited 0 with no critical issues |
| Clippy `-D warnings` | BLOCKED | `cargo clippy -p fsci-ndimage --all-targets -- -D warnings` stopped in existing `fsci-linalg` dependency lints before this patch |

| Workload / route | Mean / median | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust final no-prefilter path, `zoom/2x_256/order=1` | 1.2219 ms | 3.98x faster than SciPy; 7.24x faster than prepatch residual rerun across rch workers | SciPy win |
| SciPy `ndimage.zoom(256x256, 2x, order=1)` | 4.86171 ms | 1.00x oracle | reference |
| Rust prepatch residual current | 8.8419 ms | 1.82x slower than current SciPy oracle | superseded |

Readiness notes:

- The order-1 reflect path no longer materializes a padded coefficient image.
  It computes the same linear basis supports over the original image and keeps
  the fixed four-load bilinear pixel loop.
- This row is now done enough for release-readiness accounting. Future ndimage
  work should move to order-3 zoom, row SIMD/tile geometry, or the remaining
  label/distance-transform constant-factor gaps.

## 2026-06-20 - fsci-ndimage label mean cheap dense-probe gauntlet

- Agent: cod-b / MistyBirch
- Bead: `frankenscipy-fa62u`
- Decision: KEEP the bounded cast + exact round-trip dense label probe as an
  internal win, but keep the routine classified as a SciPy LOSS on the measured
  same-host rows. Score for this sub-cluster remains `0/4/0` against SciPy.
- Artifact:
  `tests/artifacts/perf/frankenscipy-fa62u-label-mean-fast-probe-EVIDENCE.md`

| Gate | Result | Notes |
| --- | --- | --- |
| rch release A/B | PASS | `cargo run --release -p fsci-ndimage --bin perf_label_stats` on `hz2`: dense_fast is 2.08-2.26x faster than previous dense_fract, 0 mismatches |
| Same-host SciPy head-to-head | PASS | transferred rch release binary vs local SciPy 1.17.1; final source is 1.33-1.78x slower than SciPy |
| Focused dense-label test | PASS | `cargo test -p fsci-ndimage mean_dense_label_lookup_preserves_exact_label_semantics --lib -- --nocapture` via rch: 1 passed / 0 failed |
| Full ndimage lib tests | PASS | `cargo test -p fsci-ndimage --lib -- --nocapture` via rch: 241 passed / 0 failed |
| Per-crate compile | PASS | `cargo check -p fsci-ndimage --all-targets` via rch; unrelated dependency warnings remain |
| Local live SciPy conformance | PASS | `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test diff_ndimage -- --nocapture`: 5 passed / 0 failed |
| rch conformance filter | PARTIAL | lib-side ndimage filter passed 5/0, then live `diff_ndimage` failed because worker `hz2` lacked Python SciPy |
| Touched-file formatting | PASS | `rustfmt --edition 2024 --check crates/fsci-ndimage/src/lib.rs crates/fsci-ndimage/src/bin/perf_label_stats.rs` |
| Diff hygiene | PASS | `git diff --check -- crates/fsci-ndimage/src/lib.rs crates/fsci-ndimage/src/bin/perf_label_stats.rs` |
| Changed-file UBS scan | PASS | `ubs` exited 0; no critical issues; broad warning inventory left untouched |
| Clippy `-D warnings` | BLOCKED | no-deps clippy stopped on existing unrelated `fsci-ndimage` lib-file lints outside this patch |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust dense_fast mean, N=65536 K=512 | 278.230 us | 1.33x slower than SciPy; 2.13x faster than previous dense_fract on same host | SciPy loss, internal keep |
| SciPy `ndimage.mean`, N=65536 K=512 | 0.210 ms | 1.00x oracle | reference |
| Rust dense_fast mean, N=262144 K=1024 | 1.122 ms | 1.50x slower than SciPy; 2.09x faster than previous dense_fract on same host | SciPy loss, internal keep |
| SciPy `ndimage.mean`, N=262144 K=1024 | 0.749 ms | 1.00x oracle | reference |
| Rust dense_fast mean, N=262144 K=2048 | 1.186 ms | 1.58x slower than SciPy; 2.13x faster than previous dense_fract on same host | SciPy loss, internal keep |
| SciPy `ndimage.mean`, N=262144 K=2048 | 0.751 ms | 1.00x oracle | reference |
| Rust dense_fast mean, N=589824 K=4096 | 3.169 ms | 1.78x slower than SciPy; 1.94x faster than previous dense_fract on same host | SciPy loss, internal keep |
| SciPy `ndimage.mean`, N=589824 K=4096 | 1.781 ms | 1.00x oracle | reference |

Readiness notes:

- The dense-label route is now 1.94-2.13x faster than the previous dense
  `fract` probe on the same host and 2.08-2.26x faster on rch `hz2`, with
  `mism=0/0/0/0`.
- This is still not enough to beat SciPy. Future work should move below scalar
  label probing into parallel/cache-tiled accumulators, vector-friendly label
  generation, or sorted/run-grouped label spans.

## 2026-06-20 - fsci-integrate Radau diagonal-Jacobian stage solve

- Agent: cod-a / MistyBirch
- Bead: `frankenscipy-zpunl`
- Decision: KEEP. The remaining Radau64 stiff-suite loss is now a SciPy win by
  avoiding dense stage-system assembly/LU when the Jacobian is exactly diagonal.
  Final stiff-suite score is `4/0/0` against SciPy.
- Artifact:
  `tests/artifacts/perf/frankenscipy-zpunl-radau64-stage-lu/EVIDENCE.md`

| Gate | Result | Notes |
| --- | --- | --- |
| Same-worker A/B | PASS | Baseline worktree `8e8a5b45` vs candidate on rch `ovh-a`: Radau64 `70047.428100 us -> 1124.530700 us`, 62.29x faster, unchanged counters/checksum |
| Rust release benchmark | PASS | Final-source `radau-stiff64 20` on rch `hz2`: `1687.783900 us/call`; final-source `stiff-suite 10` on rch `hz2`: `4/0/0` vs SciPy |
| SciPy head-to-head oracle | PASS | local SciPy 1.17.1 via `docs/perf_oracle_integrate_stiff.py`; final Radau64 focused row is 21.65x faster than SciPy |
| Focused Radau tests | PASS | `cargo test -p fsci-integrate radau --lib -- --nocapture` via rch: 3 passed / 0 failed |
| Crate compile | PASS | `cargo check -p fsci-integrate --all-targets` via rch |
| Formatting | PASS | `rustfmt --edition 2024 --check crates/fsci-integrate/src/radau.rs` |
| Diff hygiene | PASS | `git diff --check` |
| Clippy `-D warnings` | BLOCKED | existing non-Radau `api.rs`/`rk.rs`/`quad.rs` lint debt; touched Radau helper issue fixed |
| IVP conformance | BLOCKED | `fsci-conformance` currently compiles unrelated `fsci-cluster`, which fails before IVP tests on missing `fsci_linalg::randomized_svd/randomized_eigh` symbols |

| Workload / route | Final Rust | SciPy oracle | Ratio | Verdict |
| --- | ---: | ---: | ---: | --- |
| `bdf-stiff64` | 2306.348100 us | 25448.461401 us | 11.03x faster | SciPy win |
| `bdf-stiff128` | 12041.547000 us | 29874.872195 us | 2.48x faster | SciPy win |
| `radau-stiff32` | 591.275600 us | 33708.498604 us | 57.01x faster | SciPy win |
| `radau-stiff64` | 1306.515700 us | 36488.462600 us | 27.93x faster | SciPy win |
| `radau-stiff64` focused final source | 1687.783900 us | 36545.873049 us | 21.65x faster | SciPy win |

Readiness notes:

- Exactly diagonal Radau Jacobians now use independent `3 x 3` stage solves and
  scalar real-shift solves. Non-diagonal Jacobians keep the dense fallback.
- The prior Radau streamed scaled-RMS candidate remains rejected. Future Radau
  work should target banded/non-diagonal structure detection or LU reuse.

## 2026-06-20 - fsci-integrate stiff BDF/Radau stream-norm gauntlet

- Agent: cod-a / MistyBirch
- Beads: `frankenscipy-8l8r1.119`, `frankenscipy-8l8r1.120`,
  `frankenscipy-8l8r1.121`
- Decision: KEEP the BDF streamed scaled RMS work, REJECT and REVERT the Radau
  streamed scaled RMS work. Final source scores `3/1/0` against SciPy on the
  stiff suite; remaining loss is `radau-stiff64` and is tracked by
  `frankenscipy-zpunl`.
- Artifact:
  `tests/artifacts/perf/frankenscipy-8l8r1-119-121-integrate-stiff-stream-norms/EVIDENCE.md`

| Gate | Result | Notes |
| --- | --- | --- |
| Rust release benchmark | PASS | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo run --release -p fsci-integrate --bin perf_integrate -- stiff-suite 10` |
| Same-worker A/B | PASS | Baseline detached worktree at `d502c748` plus harness patch vs candidate on rch `hz2`; BDF64 1.040x faster, BDF128 neutral, Radau32/Radau64 regressed and were reverted |
| SciPy head-to-head oracle | PASS | local SciPy oracle via `docs/perf_oracle_integrate_stiff.py`; final source wins BDF64/BDF128/Radau32 and loses Radau64 |
| Focused BDF tests | PASS | `cargo test -p fsci-integrate bdf --lib -- --nocapture` via rch: 16 passed / 0 failed |
| Focused Radau tests | PASS | `cargo test -p fsci-integrate radau --lib -- --nocapture` via rch: 2 passed / 0 failed |
| IVP conformance | PASS | `cargo test -p fsci-conformance --test e2e_ivp -- --nocapture` via rch: 11 passed / 0 failed |
| Crate compile | PASS | `cargo check -p fsci-integrate --all-targets` via rch |
| Formatting | PASS | touched-file `rustfmt --edition 2024 --check` |
| Diff hygiene | PASS | `git diff --check` |
| Clippy `-D warnings` | BLOCKED | existing `fsci-integrate` lint debt outside touched files; filed `frankenscipy-3qjah` |

| Workload / route | Final Rust | SciPy oracle | Ratio | Verdict |
| --- | ---: | ---: | ---: | --- |
| `bdf-stiff64` | 1959.286800 us | 26351.239008 us | 13.45x faster | SciPy win, internal keep |
| `bdf-stiff128` | 11052.293000 us | 29334.902694 us | 2.65x faster | SciPy win, internal neutral/keep |
| `radau-stiff32` | 10191.487800 us | 33444.223704 us | 3.28x faster | SciPy win after revert |
| `radau-stiff64` | 70176.946400 us | 35156.708304 us | 2.00x slower | SciPy loss; route deeper |

Readiness notes:

- The Radau streamed-norm candidate was reverted because same-worker `hz2`
  timings regressed Radau32 by 1.19x and Radau64 by 1.04x with unchanged
  `nfev/njev/nlu` and checksum.
- The remaining integrate performance work is not another scaled-RMS norm
  micro-lever. It should target Radau stage linear algebra, LU reuse, allocation
  in matrix/vector assembly, or structured-Jacobian exploitation.

## 2026-06-20 - fsci-sparse public CSR SpMV row-loop gauntlet

- Agent: cod-b / MistyBirch
- Bead: `frankenscipy-2hclc`
- Decision: KEEP the cached-slice + 4-lane unrolled public CSR row sweep.
  Score for this sub-cluster: 4/5. It closes the public `spmv_csr` scale loss
  on the scoped CSR `.dot(x)` benchmark; future work should profile before
  attempting explicit SIMD or sparse-BLAS-style row blocking.
- Artifact: inline in `docs/progress/perf-negative-results.md` and
  `docs/perf_ledger_cc.md` (no new artifact directory).

| Gate | Result | Notes |
| --- | --- | --- |
| SciPy head-to-head oracle | PASS | `python3 docs/perf_oracle_spmv.py`, local SciPy 1.17.1, n=100/1000/10000 |
| Criterion focused bench | PASS | `cargo bench -p fsci-sparse --bench sparse_bench -- sparse_spmv --sample-size 10 --measurement-time 1 --warm-up-time 1` via rch |
| Same-process old/current A/B | PASS | `env FSCI_PUBLIC_SPMV_AB=1 cargo run --profile release-perf -p fsci-sparse --bin perf_csr_matvec` via rch `ovh-a`; all rows `identical=true` |
| Focused sparse tests | PASS | `cargo test -p fsci-sparse spmv -- --nocapture` via rch: 5 unit/property + 4 metamorphic pass |
| Sparse compile | PASS | `cargo check -p fsci-sparse --all-targets` via rch |
| Sparse clippy | PASS | `cargo clippy -p fsci-sparse --all-targets --no-deps -- -D warnings` via rch |
| Touched-file formatting | PASS | `rustfmt --edition 2024 --check crates/fsci-sparse/src/ops.rs crates/fsci-sparse/src/bin/perf_csr_matvec.rs` |
| Crate-wide formatting | BLOCKED | `cargo fmt -p fsci-sparse --check` reports existing drift in untouched `sparse_bench.rs`, `src/lib.rs`, and `src/linalg.rs`; not reformatted in this patch |
| Diff hygiene | PASS | `git diff --check -- crates/fsci-sparse/src/ops.rs crates/fsci-sparse/src/bin/perf_csr_matvec.rs .beads/issues.jsonl` |
| Changed-file UBS scan | PASS | `ubs crates/fsci-sparse/src/ops.rs crates/fsci-sparse/src/bin/perf_csr_matvec.rs`: 0 critical; existing warnings only |
| Sparse conformance | PASS | `cargo test -p fsci-conformance --test diff_sparse spmv -- --nocapture` via rch: 11 passed / 0 failed |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust current `spmv_csr`, n=100 nnz=500 | 387.54 ns | 11.95x faster than SciPy | SciPy win |
| SciPy CSR `.dot(x)`, n=100 nnz=500 | 4.63 us | 1.00x oracle | reference |
| Rust current `spmv_csr`, n=1000 nnz=10000 | 7.077 us | 1.13x faster than SciPy | SciPy win |
| SciPy CSR `.dot(x)`, n=1000 nnz=10000 | 8.00 us | 1.00x oracle | reference |
| Rust current `spmv_csr`, n=10000 nnz=100000 | 68.820 us | 1.41x faster than SciPy | SciPy win |
| SciPy CSR `.dot(x)`, n=10000 nnz=100000 | 96.95 us | 1.00x oracle | reference |
| Legacy public row sweep A/B, n=100/1000/10000 | 550 ns / 12.074 us / 135.043 us | current is 1.54x / 2.10x / 2.14x faster | internal baseline |

Readiness notes:

- Score vs SciPy is 3 wins / 0 losses / 0 neutral. The prior public SpMV
  ledger row was 1 win / 2 losses, so this closes the measured scale gap.
- The private Krylov/eigensolver `csr_matvec_into` route was already present;
  this patch targets only the public `ops.rs` SpMV API and the perf proof mode.

## 2026-06-20 - fsci-ndimage label mean dense-lookup gauntlet

- Agent: cod-a / MistyBirch
- Bead: `frankenscipy-klb7o`
- Decision: KEEP the compact integer-label dense lookup as an internal win,
  but mark the routine as a SciPy LOSS on the measured rows. Score for this
  sub-cluster: 3.5/5.
- Artifact:
  `tests/artifacts/perf/2026-06-20-label-stats-dense-mean/EVIDENCE.md`

| Gate | Result | Notes |
| --- | --- | --- |
| Rust per-crate compile | PASS | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo check -p fsci-ndimage --all-targets`; existing `fsci-interpolate` and `diff_geom` warnings remain |
| Full ndimage lib tests | PASS | `cargo test -p fsci-ndimage --lib -- --nocapture` via rch: 241 passed / 0 failed |
| Same-binary A/B | PASS | `perf_label_stats` compares old O(N*K), bucketed O(N+K), previous flat HashMap O(N+K), and dense O(N+K); mismatches 0/0/0 |
| rch release A/B | PASS | `cargo run --release -p fsci-ndimage --bin perf_label_stats` via rch worker `hz2`: dense is 2.00-2.48x faster than prior flat HashMap |
| SciPy head-to-head oracle | PASS | local SciPy 1.17.1 via `docs/perf_oracle_label_stats.py`; final source remains 1.9-2.6x slower |
| ndimage conformance | PASS | rch `cargo test -p fsci-conformance ndimage -- --nocapture`: lib-side ndimage filter 5 passed / 0 failed; local live SciPy `diff_ndimage` with `FSCI_REQUIRE_SCIPY_ORACLE=1`: 5 passed / 0 failed |
| Touched-file formatting | PASS | `rustfmt --edition 2024 --check crates/fsci-ndimage/src/lib.rs crates/fsci-ndimage/src/bin/perf_label_stats.rs` |
| Diff hygiene | PASS | `git diff --check -- crates/fsci-ndimage/src/lib.rs crates/fsci-ndimage/src/bin/perf_label_stats.rs` |
| Changed-file UBS scan | PASS | `ubs` exited 0; no critical issues; broad warning inventory left untouched |
| Clippy `-D warnings` | BLOCKED | dependency clippy stopped on existing `fsci-linalg` lints; no-deps clippy stopped on existing unrelated `fsci-ndimage` lib-file lints outside this patch |
| Full conformance suite | BLOCKED | unrelated existing failures: missing `P2C-007/contracts/contract_table.json`, Array API tolerance fallback mismatch, and missing `legacy_scipy_code/scipy` on rch worker image |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust dense mean, N=65536 K=512 | 301.407 us | 1.91x slower than SciPy; 2.08x faster than prior flat HashMap Rust | SciPy loss, internal keep |
| Rust prior flat HashMap mean, N=65536 K=512 | 625.594 us | dense is 2.08x faster | internal baseline |
| SciPy `ndimage.mean`, N=65536 K=512 | 158 us | 1.00x oracle | reference |
| Rust dense mean, N=262144 K=1024 | 1.263 ms | 2.35x slower than SciPy; 2.10x faster than prior flat HashMap Rust | SciPy loss, internal keep |
| Rust prior flat HashMap mean, N=262144 K=1024 | 2.652 ms | dense is 2.10x faster | internal baseline |
| SciPy `ndimage.mean`, N=262144 K=1024 | 538 us | 1.00x oracle | reference |
| Rust dense mean, N=262144 K=2048 | 1.383 ms | 2.51x slower than SciPy; 2.02x faster than prior flat HashMap Rust | SciPy loss, internal keep |
| Rust prior flat HashMap mean, N=262144 K=2048 | 2.798 ms | dense is 2.02x faster | internal baseline |
| SciPy `ndimage.mean`, N=262144 K=2048 | 550 us | 1.00x oracle | reference |
| Rust dense mean, N=589824 K=4096 | 3.351 ms | 2.64x slower than SciPy; 2.02x faster than prior flat HashMap Rust | SciPy loss, internal keep |
| Rust prior flat HashMap mean, N=589824 K=4096 | 6.772 ms | dense is 2.02x faster | internal baseline |
| SciPy `ndimage.mean`, N=589824 K=4096 | 1.271 ms | 1.00x oracle | reference |

Readiness notes:

- This is a real measured constant-factor win over the prior flat HashMap
  implementation. The final source scores `0/4/0` against SciPy and `4/0/0`
  against the previous Rust flat HashMap route.
- Future label-stat work should route below per-element lookup overhead:
  sorted-label remapping, fused integer-label generation from `label()`, SIMD
  accumulation over contiguous label spans, or cache-tiled sum/count reductions.

## 2026-06-20 - fsci-spatial Delaunay circumcircle-grid gauntlet

- Agent: cod-b / MistyBirch
- Bead: `frankenscipy-9l5oo`
- Decision: KEEP the n>=4096 circumcircle-grid candidate index. Score for this
  sub-cluster: 4/5. It closes the measured 4000/8000 SciPy losses to parity, but
  still needs a beyond-8000 re-test before claiming asymptotic dominance.
- Artifact: inline in `docs/progress/perf-negative-results.md` and
  `docs/perf_ledger_cc.md` (no new artifact directory).

| Gate | Result | Notes |
| --- | --- | --- |
| SciPy head-to-head oracle | PASS | `python3 docs/perf_oracle_delaunay.py`, local SciPy 1.17.1, n=1000/2000/4000/8000 |
| Criterion focused bench | PASS | `cargo bench -p fsci-spatial --bench spatial_bench -- delaunay --sample-size 10 --measurement-time 1 --warm-up-time 1` via rch |
| Focused Delaunay tests | PASS | `cargo test -p fsci-spatial delaunay -- --nocapture`: 8 unit + 1 metamorphic pass |
| Full spatial lib tests | PASS | `cargo test -p fsci-spatial --lib -- --nocapture`: 208 passed / 0 failed / 2 ignored |
| Spatial conformance | PASS | `cargo test -p fsci-conformance --test e2e_spatial -- --nocapture`: 16/0 |
| Check / clippy / fmt | PASS | `cargo check -p fsci-spatial --all-targets`; `cargo clippy -p fsci-spatial --all-targets --no-deps -- -D warnings`; `cargo fmt --check -p fsci-spatial` |
| Changed-file UBS | PASS | `ubs crates/fsci-spatial/src/lib.rs crates/fsci-spatial/benches/spatial_bench.rs docs/perf_oracle_delaunay.py`; no criticals |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust current `Delaunay`, n=1000 | 754.03 us | 2.56x faster than SciPy | SciPy win |
| SciPy `spatial.Delaunay`, n=1000 | 1.93258 ms | 1.00x oracle | reference |
| Rust current `Delaunay`, n=2000 | 2.6129 ms | 1.74x faster than SciPy | SciPy win |
| SciPy `spatial.Delaunay`, n=2000 | 4.54974 ms | 1.00x oracle | reference |
| Rust current `Delaunay`, n=4000 | 9.4632 ms | parity with SciPy | neutral |
| SciPy `spatial.Delaunay`, n=4000 | 9.50086 ms | 1.00x oracle | reference |
| Rust current `Delaunay`, n=8000 | 20.622 ms | parity with SciPy | neutral |
| SciPy `spatial.Delaunay`, n=8000 | 20.62714 ms | 1.00x oracle | reference |

Readiness notes:

- Score vs SciPy is 2 wins / 0 losses / 2 neutral on the scoped deterministic
  2-D workload. The pre-grid expanded probe showed losses at 4000/8000; the
  grid closes those rows to parity.
- This is a candidate-pruning accelerator for Bowyer-Watson, not a full
  randomized incremental/history-DAG point-location rewrite. Re-test at larger
  n before routing remaining work as purely constant-factor.

## 2026-06-20 - fsci-ndimage label mean flat-accumulator gauntlet

- Agent: cod-a / MistyBirch
- Bead: `frankenscipy-8l8r1.125`
- Decision: KEEP the flat `mean(labels,index)` accumulator as an internal win,
  but mark the routine as a SciPy LOSS on the measured rows. Score for this
  sub-cluster: 3/5.
- Artifact:
  `tests/artifacts/perf/2026-06-20-label-stats-flat-mean/EVIDENCE.md`

| Gate | Result | Notes |
| --- | --- | --- |
| Rust per-crate compile | PASS | `RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo check -p fsci-ndimage --all-targets`; existing `fsci-interpolate` and `diff_geom` warnings remain |
| Focused reduction tests | PASS | `cargo test -p fsci-ndimage measurement_reduction_wrappers -- --nocapture` via rch: 2 passed / 0 failed |
| Full ndimage lib tests | PASS | `cargo test -p fsci-ndimage --lib -- --nocapture` via rch: 240 passed / 0 failed |
| Same-binary A/B | PASS | `perf_label_stats` compares old O(N*K), previous bucketed O(N+K), and flat O(N+K) in one optimized binary; mismatches 0/0 |
| SciPy head-to-head oracle | PASS | local SciPy 1.17.1 via `docs/perf_oracle_label_stats.py`; final source remains 3.7-4.7x slower |
| Touched-file formatting | PASS | `rustfmt --edition 2024 --check crates/fsci-ndimage/src/lib.rs crates/fsci-ndimage/src/bin/perf_label_stats.rs` |
| Diff hygiene | PASS | `git diff --check -- crates/fsci-ndimage/src/lib.rs crates/fsci-ndimage/src/bin/perf_label_stats.rs` |
| Changed-file UBS scan | PASS | `ubs` exited 0; no critical issues; broad warning inventory left untouched |
| Clippy `-D warnings` | BLOCKED | `cargo clippy -p fsci-ndimage --lib -- -D warnings` stopped before this crate on existing `fsci-linalg` dependency lints outside this patch |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust flat mean, N=65536 K=512 | 590.978 us | 3.72x slower than SciPy; 1.77x faster than prior bucketed Rust | SciPy loss, internal keep |
| Rust prior bucketed mean, N=65536 K=512 | 1.047 ms | flat is 1.77x faster | internal baseline |
| SciPy `ndimage.mean`, N=65536 K=512 | 159 us | 1.00x oracle | reference |
| Rust flat mean, N=262144 K=1024 | 2.568 ms | 4.13x slower than SciPy; 1.44x faster than prior bucketed Rust | SciPy loss, internal keep |
| Rust prior bucketed mean, N=262144 K=1024 | 3.695 ms | flat is 1.44x faster | internal baseline |
| SciPy `ndimage.mean`, N=262144 K=1024 | 622 us | 1.00x oracle | reference |
| Rust flat mean, N=262144 K=2048 | 2.713 ms | 4.67x slower than SciPy; 1.53x faster than prior bucketed Rust | SciPy loss, internal keep |
| Rust prior bucketed mean, N=262144 K=2048 | 4.140 ms | flat is 1.53x faster | internal baseline |
| SciPy `ndimage.mean`, N=262144 K=2048 | 581 us | 1.00x oracle | reference |
| Rust flat mean, N=589824 K=4096 | 6.951 ms | 4.12x slower than SciPy; 1.69x faster than prior bucketed Rust | SciPy loss, internal keep |
| Rust prior bucketed mean, N=589824 K=4096 | 11.760 ms | flat is 1.69x faster | internal baseline |
| SciPy `ndimage.mean`, N=589824 K=4096 | 1.688 ms | 1.00x oracle | reference |

Readiness notes:

- This is a real constant-factor win over the prior O(N+K) implementation, not
  a SciPy parity claim. The final source scores `0/4/0` against SciPy and
  `4/0/0` against the previous Rust bucketed route.
- Future label-stat work should target the remaining per-element lookup and
  accumulation constants: dense label lookup for compact integer labels,
  sorted-label remapping, or SIMD/cache-tiled accumulation. Reintroducing
  grouped `Vec<Vec<f64>>` materialization for `mean` is negative evidence.

## 2026-06-19 - fsci-signal coherence fused-Welch gauntlet

- Agent: cod-b / MistyBirch
- Bead: `frankenscipy-8l8r1.118`
- Decision: KEEP the fused coherence route. Score for this sub-cluster: 5/5.
- Artifact: `tests/artifacts/perf/frankenscipy-8l8r1.118/EVIDENCE.md`

| Gate | Result | Notes |
| --- | --- | --- |
| Rust per-crate compile | PASS | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo check -p fsci-signal --all-targets` |
| Focused signal tests | PASS | `cargo test -p fsci-signal coherence_matches --lib -- --nocapture` passed via rch worker `hz2` |
| Criterion focused bench | PASS | `coherence_gauntlet_scipy` group only; local SciPy oracle plus rch Rust-only confirmation |
| SciPy head-to-head oracle | PASS | local Python oracle, `scipy.signal.coherence`; rch worker `hz1` lacked `scipy.signal`, so the remote run skipped the SciPy row |
| Changed benchmark formatting | PASS | `rustfmt --edition 2024 --check crates/fsci-signal/benches/signal_bench.rs` |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust fused `coherence`, 65536 samples, Hann 1024/512 | 2.191980 ms | 8.65x faster than SciPy | SciPy win |
| Rust compositional triple-CSD route | 6.536569 ms | fused is 2.98x faster | internal win |
| SciPy `scipy.signal.coherence` | 18.961613 ms | 1.00x oracle | reference |
| Rust fused `coherence`, rch worker `hz1` | 4.3780 ms | 2.80x faster than compositional | internal win |
| Rust compositional triple-CSD route, rch worker `hz1` | 12.269 ms | 1.00x internal baseline | slower |

Readiness notes:

- The fused coherence API is now release evidence: it wins against original
  SciPy and the internal compositional route on the scoped workload.
- Future signal work should route deeper into FFT/window staging or shared
  Welch segment infrastructure instead of decomposing coherence back into
  independent `csd` calls.

## 2026-06-19 - fsci-cluster linkage flat-arena gauntlet

- Agent: cod-a / MistyBirch
- Bead: `frankenscipy-va60h`
- Decision: KEEP the flat row-major linkage arena as an internal win, but mark
  the full `linkage` routine as a SciPy LOSS on the measured rows. Score for
  this sub-cluster: 3/5.
- Artifact:
  `tests/artifacts/perf/2026-06-19-va60h-linkage-gauntlet/`

| Gate | Result | Notes |
| --- | --- | --- |
| Rust per-crate compile | PASS | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo check -p fsci-cluster --benches` passed; existing `perf_kmeans.rs` warning remains |
| Criterion focused bench | PASS | `va60h_gauntlet_linkage` group only; local SciPy oracle in the same benchmark process |
| SciPy head-to-head oracle | PASS | Python 3.13.7, NumPy 2.4.3, SciPy 1.17.1, `scipy.cluster.hierarchy.linkage` |
| Targeted cluster tests | PASS | `cargo test -p fsci-cluster linkage -- --nocapture` passed via rch: 28 unit tests and 9 metamorphic tests |
| Linkage conformance | PASS | `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test diff_cluster_linkage_from_distances -- --nocapture` passed locally |
| Touched-file formatting | PASS | `rustfmt --edition 2024 --check crates/fsci-cluster/benches/cluster_bench.rs crates/fsci-cluster/src/lib.rs` |
| Crate formatting | BLOCKED | `cargo fmt -p fsci-cluster --check` stopped on existing `crates/fsci-cluster/src/bin/perf_isomap.rs` formatting drift |
| Clippy `-D warnings` | BLOCKED | `cargo clippy -p fsci-cluster --benches -- -D warnings` stopped on existing `fsci-linalg` dependency lints before this benchmark file was linted |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust current flat `linkage(Average)`, n=800 d=4 | 6.1713 ms | 1.385x slower than SciPy | SciPy loss, internal keep |
| Rust legacy nested helper `linkage(Average)`, n=800 d=4 | 6.9616 ms | current flat is 1.128x faster | internal win |
| SciPy `linkage(method="average")`, n=800 d=4 | 4.4550 ms | 1.00x oracle | reference |
| Rust current flat `linkage(Ward)`, n=800 d=4 | 7.5250 ms | 1.497x slower than SciPy | SciPy loss, internal neutral/win |
| Rust legacy nested helper `linkage(Ward)`, n=800 d=4 | 7.6707 ms | current flat is 1.019x faster | internal neutral/win |
| SciPy `linkage(method="ward")`, n=800 d=4 | 5.0256 ms | 1.00x oracle | reference |

Readiness notes:

- A manual production revert probe was run and then undone: the nested-route
  probe measured 9.0669 ms on Average and 9.1589 ms on Ward, while the flat
  route measured 7.0279 ms and 7.3221 ms in the same post-revert run.
- The flat arena should stay because reverting it makes the production route
  slower, but this is not release-speed parity against original SciPy.
- Future linkage work should target the algorithmic gap with SciPy's compiled
  linkage implementation rather than another full-square matrix layout lever.

## 2026-06-19 - fsci-special jnjnp_zeros bracket-reuse gauntlet

- Agent: cod-a / MistyBirch
- Bead: `frankenscipy-acoco`
- Decision: KEEP the bracket-reuse optimization as an internal win, but mark
  the full routine as a SciPy LOSS. Score for this sub-cluster: 3/5.
- Artifact:
  `tests/artifacts/perf/2026-06-19-acoco-jnjnp-zeros-gauntlet/`

| Gate | Result | Notes |
| --- | --- | --- |
| Rust per-crate compile | PASS | `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo check -p fsci-special --benches` |
| Criterion focused bench | PASS | `acoco_gauntlet_jnjnp_zeros` group only; local SciPy oracle in the same benchmark process |
| SciPy head-to-head oracle | PASS | Python 3.13.7, NumPy 2.4.3, SciPy 1.17.1, `scipy.special.jnjnp_zeros(nt)` |
| Targeted special tests | PASS | `jnyn_and_jnjnp_zeros_match_scipy` and `derivative_bessel_zeros_match_scipy_reference_points` passed via rch |
| Benchmark formatting | PASS | `rustfmt --edition 2024 --check crates/fsci-special/benches/special_bench.rs` |
| Clippy `-D warnings` | BLOCKED | `cargo clippy -p fsci-special --benches -- -D warnings` stopped on existing `fsci-integrate` and `fsci-linalg` dependency lints before this benchmark file was linted |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust current `jnjnp_zeros(nt=64)` | 80.728603 ms | 163.53x slower than SciPy | SciPy loss |
| Rust legacy duplicate route, `nt=64` | 101.762454 ms | current is 1.26x faster | internal keep |
| SciPy `scipy.special.jnjnp_zeros(nt=64)` | 0.493655 ms | 1.00x oracle | reference |
| Rust current `jnjnp_zeros(nt=128)` | 410.059973 ms | 443.57x slower than SciPy | SciPy loss |
| Rust legacy duplicate route, `nt=128` | 544.006333 ms | current is 1.33x faster | internal keep |
| SciPy `scipy.special.jnjnp_zeros(nt=128)` | 0.924456 ms | 1.00x oracle | reference |

Readiness notes:

- The bracket-reuse optimization should stay because reverting it would make
  the current Rust path slower on both measured rows.
- This is not release-speed evidence against original SciPy. The next
  performance bead should target the zero-enumeration/root-finding algorithm
  rather than another duplicate-bracketing micro-lever.
- The clippy blocker is outside this benchmark/evidence change and should be
  handled as a separate lint-hygiene bead.

## 2026-06-19 - fsci-special jnjnp_zeros top-k select reject

- Agent: cod-a / MistyBirch
- Bead: `frankenscipy-8l8r1.124`
- Decision: REJECT and revert the top-k candidate partition. Score for this
  sub-cluster: 2/5.
- Artifact:
  `tests/artifacts/perf/2026-06-19-8l8r1-jnjnp-topk-select/EVIDENCE.md`

| Gate | Result | Notes |
| --- | --- | --- |
| Rust per-crate Criterion bench | PASS | baseline current route on RCH worker `vmi1153651`; candidate Criterion run on `hz2` treated as directional only because RCH worker affinity was not stable |
| Same-binary A/B probe | PASS | ignored release probe asserted bit-identical output, then timed full-sort vs top-k in one optimized binary |
| SciPy head-to-head oracle | PASS | local SciPy 1.17.1, `scipy.special.jnjnp_zeros`; RCH workers lacked `scipy.special` |
| Focused special tests | PASS | `cargo test -p fsci-special jnjnp -- --nocapture` passed via RCH during the probe |
| Source revert | PASS | `crates/fsci-special/src/bessel.rs` restored to the cutoff-driven full-sort route |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Restored Rust full-sort `jnjnp_zeros(nt=64)` | 1.5407 ms | 3.65x slower than SciPy | SciPy loss, final source |
| Restored Rust full-sort `jnjnp_zeros(nt=128)` | 3.5199 ms | 4.54x slower than SciPy | SciPy loss, final source |
| SciPy `jnjnp_zeros(nt=64)` | 421.59 us | 1.00x oracle | reference |
| SciPy `jnjnp_zeros(nt=128)` | 774.75 us | 1.00x oracle | reference |
| Top-k candidate, same-binary `nt=64` | 0.911730 ms | 1.019x vs full-sort | reject as near-noise |
| Top-k candidate, same-binary `nt=128` | 1.715855 ms | 1.013x vs full-sort | reject as near-noise |

Readiness notes:

- The final source is unchanged by this rejected probe.
- Future `jnjnp_zeros` work should target root-generation and Bessel
  evaluation constants, not candidate prefix sorting or partial sort variants.

## 2026-06-19 - fsci-fft CSD rfft-route gauntlet

- Agent: cod-b / MistyBirch
- Bead: `frankenscipy-8l8r1.116`
- Decision: REJECT and revert the `cross_spectral_density` rfft real-spectrum
  route. Score for this sub-cluster: 3/5.
- Artifact: `tests/artifacts/perf/2026-06-19-fft-csd-gauntlet/csd_rfft_reject.json`

| Gate | Result | Notes |
| --- | --- | --- |
| Rust per-crate Criterion bench | PASS | same-worker parent/candidate A/B on `hz1`; post-revert confirmation landed on `ovh-a` and is not used for the A/B |
| SciPy head-to-head oracle | PASS | local SciPy 1.17.1 / NumPy 2.4.3, equivalent `scipy.fft.rfft` cross-spectrum formula |
| Source revert | PASS | `crates/fsci-fft/src/helpers.rs` restored to the full-complex route |
| Rust per-crate compile | PASS | `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo check -p fsci-fft --all-targets` |
| CSD conformance guard | PASS | `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo test -p fsci-fft cross_spectral_density -- --nocapture` |
| Clippy `-D warnings` | PASS | `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo clippy -p fsci-fft --all-targets -- -D warnings`; required a test-only `dst` import and one golden-value precision cleanup |
| Golden-value guard | PASS | `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo test -p fsci-fft rfft_matches_exact_numpy_dft_golden_values -- --nocapture` |
| Diff hygiene | PASS | `git diff --check` |
| UBS changed-file scan | PASS | `ubs` on the changed file set exited 0; warnings were broad existing FFT inventory, not commit-blocking criticals |
| Changed-file formatting | PASS/BLOCKED | `rustfmt --edition 2024 --check` passes for the changed bench/import surface; broad crate formatting still reports pre-existing file-wide drift in `src/lib.rs` and `src/helpers.rs`, so this commit does not normalize unrelated formatting |

Same-worker internal A/B:

| Workload | Parent full-complex mean | Candidate rfft mean | Candidate time vs parent | Verdict |
| --- | ---: | ---: | ---: | --- |
| `fft_helpers/cross_spectral_density/4096` | 112.08 us | 125.88 us | 1.123x | loss |
| `fft_helpers/cross_spectral_density/65536` | 4.9543 ms | 2.3509 ms | 0.475x | win |

Candidate vs original SciPy rfft formula:

| Workload | Candidate Rust mean | SciPy p50 | SciPy/Rust | Verdict |
| --- | ---: | ---: | ---: | --- |
| `cross_spectral_density/4096` | 125.88 us | 72.091 us | 0.573x | Rust slower |
| `cross_spectral_density/65536` | 2.3509 ms | 1.653584 ms | 0.703x | Rust slower |

Readiness notes:

- The rfft route is not release evidence: it has one internal win, one
  internal loss, and loses both rows against the fastest equivalent SciPy
  formula.
- The CSD Criterion rows and SciPy oracle script remain as durable gauntlet
  infrastructure for future FFT work.
- Future attempts need a same-worker win on both 4096 and 65536 and must beat
  the SciPy rfft formula before replacing the full-complex route again.

## 2026-06-19 - fsci-opt L-BFGS-B Wolfe-probe gauntlet

- Agent: cod-b / MistyBirch
- Bead: `frankenscipy-8l8r1.122`
- Decision: REJECT and revert the mutable Wolfe finite-difference
  gradient-probe scratch path. Score for this sub-cluster: 4/5.
- Artifact: `tests/artifacts/perf/2026-06-19-opt-lbfgsb-gauntlet/lbfgsb_wolfe_probe_reject.json`

| Gate | Result | Notes |
| --- | --- | --- |
| Rust per-crate compile | PASS | `cargo check -p fsci-opt --all-targets` via rch with `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b` |
| Criterion focused bench | PASS | `lbfgsb` group only; same-worker parent/candidate A/B on `ovh-a`, post-revert current-tree sample on `hz2` |
| SciPy head-to-head oracle | PASS | Local Python oracle, SciPy 1.17.1 / NumPy 2.4.3; rch declined the Python oracle as non-compilation |
| Targeted opt tests | PASS | `cargo test -p fsci-opt lbfgsb -- --nocapture` passed |
| L-BFGS-B conformance | PASS | `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test diff_opt_lbfgsb_minimize -- --nocapture` passed |
| Formatting | PASS | `cargo fmt -p fsci-opt --check` passed |
| Clippy `-D warnings` | PASS | `cargo clippy -p fsci-opt --all-targets -- -D warnings` passed after minimal fsci-opt lint cleanup |

Same-worker internal A/B:

| Workload | Parent p50 (us) | Candidate p50 (us) | Candidate time vs parent | Verdict |
| --- | ---: | ---: | ---: | --- |
| `lbfgsb/rosenbrock_unconstrained_fd/2` | 17.491 | 17.405 | 0.995x | neutral |
| `lbfgsb/rosenbrock_unconstrained_fd/10` | 87.087 | 106.440 | 1.222x | loss |
| `lbfgsb/quadratic_unconstrained_fd/32` | 5.246 | 6.055 | 1.154x | loss |

Post-revert current route vs SciPy:

| Workload | Rust p50 (us) | SciPy p50 (us) | SciPy/Rust p50 | Verdict |
| --- | ---: | ---: | ---: | --- |
| `lbfgsb/rosenbrock_unconstrained_fd/2` | 22.236 | 4585.899 | 206.24x | current route remains fast |
| `lbfgsb/rosenbrock_unconstrained_fd/10` | 105.090 | 18262.642 | 173.78x | current route remains fast |
| `lbfgsb/quadratic_unconstrained_fd/32` | 6.313 | 1447.172 | 229.23x | current route remains fast |

Readiness notes:

- The release candidate is the reverted parent-style Strong-Wolfe path, not the
  mutable finite-difference probe scratch path from `b5dbf124`.
- The original SciPy ratios remain strong on realistic Python-callback
  workloads, but the rejected lever made the in-crate route slower on the
  larger same-worker rows and is not release evidence.
- Future opt work should route to a fresh measured hotspot instead of retrying
  Wolfe finite-difference probe Vec reuse without new allocation-profile proof.

## 2026-06-19 - fsci-linalg wide lstsq row-streaming gauntlet

- Agent: cod-a / MistyBirch
- Bead: `frankenscipy-u0ucw`
- Decision: REVERT row-streamed wide `lstsq`; KEEP current materialized
  normal-equation route. Score for this sub-cluster: 4/5.
- Artifact: `tests/artifacts/perf/2026-06-19-u0ucw-wide-lstsq-gauntlet/`

| Gate | Result | Notes |
| --- | --- | --- |
| Rust per-crate compile | PASS | `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo check -p fsci-linalg --benches` |
| Criterion focused bench | PASS | same-worker row-streaming vs materialized A/B on `vmi1227854`; local SciPy row for original-SciPy ratio |
| SciPy head-to-head oracle | PASS | SciPy 1.17.1 / NumPy 2.4.3, `scipy.linalg.lstsq(check_finite=False)` |
| Targeted linalg tests | PASS | `wide_pinv` filtered tests passed, preserving the surviving row-major wide `pinv` helpers |
| Release route probe | PASS | ignored release probe reported `lstsq_max_abs_diff=3.38840067115597776e-13` |
| Changed bench formatting | PASS | `rustfmt --edition 2024 --check crates/fsci-linalg/benches/linalg_bench.rs` |
| Direct `src/lib.rs` formatting | BLOCKED | file-wide pre-existing rustfmt drift outside this revert; broad-formatting would create unrelated churn in the shared checkout |
| Clippy `-D warnings` | BLOCKED | pre-existing `src/lib.rs` lints plus concurrent `src/cossin.rs` excessive-precision lints; no row-streaming revert-specific issue identified |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust row-streamed wide `lstsq`, 500x1000 | 139.965 ms | 0.966x vs materialized | loss, reverted |
| Rust materialized wide `lstsq`, 500x1000 | 135.206 ms | 1.035x vs row-streamed | keep old route |
| Rust current materialized wide `lstsq`, 500x1000 | 109.370 ms | 11.46x vs SciPy | keep |
| SciPy `scipy.linalg.lstsq`, 500x1000 | 1.253347 s | 1.00x oracle | reference |

Readiness notes:

- The negative result is specific to replacing the wide `lstsq` materialized
  transpose products with row streaming. It does not invalidate the public wide
  normal-equation route, which remains faster than SciPy on the measured
  workload.
- `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo clippy -p fsci-linalg --benches -- -D warnings`
  currently fails on existing lints unrelated to the measured revert, including
  `needless_range_loop` / `needless_borrow` rows in `src/lib.rs` and
  excessive-precision rows in concurrently modified `src/cossin.rs`.
- Future retries need a fresh allocation/cache profile and a same-worker >10%
  win over the materialized route before this formulation should be reconsidered.

## 2026-06-19 - fsci-opt least_squares scratch cluster

- Agent: cod-b / MistyBirch
- Commit under verification: `41bf34a4`
- Beads: `frankenscipy-szky7`, `frankenscipy-y1mzk`
- Decision: KEEP, no revert. Score for this cluster: 4/5.
- Artifact: `tests/artifacts/perf/2026-06-19-opt-least-squares-gauntlet/least_squares_vs_scipy.json`

| Gate | Result | Notes |
| --- | --- | --- |
| Rust per-crate compile | PASS | `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo check -p fsci-opt` |
| Criterion focused bench | PASS | `least_squares` group only, remote worker `vmi1227854` |
| SciPy head-to-head oracle | PASS | SciPy 1.17.1 / NumPy 2.4.3, warmed single-process `method="lm"` |
| Targeted metamorphic tests | PASS | 2 least-squares rows passed via `--test metamorphic_tests mr_least_squares` |
| Release diff probes | PASS | `diff_lsq` and `diff_leastsq` release binaries converged |
| Broad fsci-opt unit gate | BLOCKED | Pre-existing `src/lib.rs` test imports fail before the target tests run; follow-up `frankenscipy-uxs8k` |

| Workload | Rust p50 (us) | SciPy p50 (us) | SciPy p99 (us) | SciPy/Rust p50 | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `least_squares/rosenbrock_residual` | 2.558 | 1404.547 | 2645.985 | 549.08x | win |
| `least_squares/exp_curve_64` | 16.932 | 753.120 | 1452.558 | 44.48x | win |
| `least_squares/exp_linear_curve_128` | 49.724 | 893.946 | 1414.085 | 17.98x | win |

Readiness notes:

- The measured workloads include Python callback overhead on the SciPy side.
  That is intentional for the original-SciPy realistic usage path targeted by
  this gauntlet; it is not evidence about lower-level C-only kernels.
- No neutral or loss rows were observed for this cluster, so the scratch-reuse
  optimization remains in tree.
- The remaining release risk is test-harness hygiene, not this optimization:
  broad `fsci-opt` unit-test compilation needs the missing helper imports fixed
  before this lane can claim a full crate test pass.
