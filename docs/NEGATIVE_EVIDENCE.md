# Negative Evidence Ledger

Canonical detailed ledger: `docs/progress/perf-negative-results.md`.

This file exists as the BOLD-VERIFY entry point requested for measured
win/loss/neutral summaries. Keep detailed attempt records in the canonical
ledger above so the project has one source of truth.

## 2026-06-20 - frankenscipy-i0ghz - pdist Chebyshev d4 SoA SIMD

- Agent: cod-a / BlackThrush
- Decision: KEEP. `pdist(..., Chebyshev)` now uses the existing dim-4
  fixed-row/SoA SIMD-across-pairs route, with an explicit NaN mask to preserve
  the scalar helper's NaN-propagating max fold. The tracked d4 gap closes from
  a 12.60x SciPy loss in routing evidence to parity/slight win.
- Artifact:
  `tests/artifacts/perf/2026-06-20-cod-a-pdist-chebyshev-d4/EVIDENCE.md`
- Additional cod-b corroborating artifact:
  `tests/artifacts/perf/frankenscipy-i0ghz-chebyshev-d4/EVIDENCE.md`; target
  row 0.139 ms versus local SciPy 0.176 ms, spatial E2E and local SciPy
  differential conformance green, changed-file UBS exits 0 after test-only
  panic-macro cleanup.
- Strict final source versus local SciPy 1.17.1 oracle score across the sweep:
  `8/6/0`. The target row is a keep; the remaining d16/d64 Chebyshev rows stay
  negative evidence and should route to a higher-dimensional SIMD/blocking pass.

| Workload | Final Rust | SciPy oracle | Verdict |
| --- | ---: | ---: | --- |
| `pdist/chebyshev/n512/d4` | 0.173 ms | 0.175 ms | keep: Rust 1.01x faster than SciPy |
| Criterion `pdist/chebyshev/512` | 136.38 us median | 175 us oracle | keep: Rust 1.28x faster than SciPy |
| `pdist/chebyshev/n512/d16` | 1.862 ms | 0.555 ms | loss: Rust 3.36x slower |
| `pdist/chebyshev/n512/d64` | 5.767 ms | 2.133 ms | loss: Rust 2.70x slower |
| `pdist/chebyshev/n2048/d64` | 71.833 ms | 39.290 ms | loss: Rust 1.83x slower |

Guards: focused dim-4 bit-identity tests including NaN fold passed; spatial
`pdist/cdist` live SciPy conformance passed; `cargo check -p fsci-spatial
--all-targets`, `cargo clippy -p fsci-spatial --all-targets --no-deps -- -D
warnings`, `cargo fmt --check -p fsci-spatial`, and `git diff --check` passed.
Changed-file `ubs` now exits 0 with 0 critical issues after cod-b converted a
test-only explicit `panic!` mismatch branch to an assertion failure.

Negative evidence: do not spend another pass on dim-4 Chebyshev. The residual
losses are d16/d64 Chebyshev, especially `n512/d16` and `n512/d64`; they need a
generic-width chunked/SIMD max kernel, not another dim-4 specialization.

## 2026-06-20 - frankenscipy-8l8r1.138 - EDT fast-path background and 2-D feature layout

- Agent: cod-b / BlackThrush
- Decision: KEEP. `distance_transform_edt(return_indices=True)` no longer
  materializes every background coordinate before the exact separable fast path,
  and 2-D inputs fuse the final axis pass with row/column output
  materialization. EDT math, axis order, and fallback semantics are unchanged.
- Artifact:
  `tests/artifacts/perf/2026-06-20-cod-b-edt-constant-factor/EVIDENCE.md`
- Same-session lazy-background score versus current Rust on `vmi1293453`:
  `4/0/0` (1.96x / 2.11x / 1.04x / 1.70x faster).
- Comparable fused path versus prior `vmi1152480` Rust scorecard rows: `3/1/0`
  because 192x192 is a small internal loss (`2.107 ms -> 2.166 ms`).
- Strict post-cleanup final source versus local SciPy 1.17.1 oracle: `4/0/0`.

| Workload | Final Rust | SciPy oracle | Verdict |
| --- | ---: | ---: | --- |
| 64x64 `return_indices` | 104.120 us | 186.092 us | keep: Rust 1.79x faster than SciPy |
| 128x128 `return_indices` | 677.777 us | 769.172 us | keep: Rust 1.13x faster than SciPy |
| 192x192 `return_indices` | 1.470 ms | 2.346150 ms | keep: Rust 1.60x faster than SciPy; comparable internal row was 0.97x vs prior Rust |
| 256x256 `return_indices` | 3.486 ms | 4.438267 ms | keep: Rust 1.27x faster than SciPy |

Guards: `perf_edt` isomorphism 0 mismatches / 10876 cells; focused EDT tests
15/0; full ndimage lib tests 246/0 with 5 ignored; `cargo check -p
fsci-ndimage --all-targets`; live SciPy EDT conformance 1/0; touched-file
rustfmt; diff hygiene; changed-file UBS exits 0 with no critical issues. Full
crate rustfmt remains blocked by pre-existing `ndimage_bench.rs` and
`diff_fourier.rs` drift; clippy remains blocked before this patch on existing
`fsci-linalg` lints.

Negative evidence: do not retry full background-coordinate materialization as a
fast-path eligibility test. The 192x192 internal row is a slight loss versus
the prior Rust scorecard and must not be counted as an internal win.

## 2026-06-20 - frankenscipy-8l8r1.137 - linear_sum_assignment first-scan initialization

- Agent: cod-b / BlackThrush
- Decision: KEEP the first-scan shortest augmenting path specialization. It
  removes whole-vector `path` and `shortest_path_costs` fills from each
  augmenting path search by initializing those arrays during the first start-row
  scan. The n=1000 row moves from a strict SciPy loss to parity/slight win; the
  n=500 row remains a small SciPy loss.
- Artifact:
  `tests/artifacts/perf/2026-06-20-cod-b-lsap-136/EVIDENCE.md`
- Same-worker internal score versus restored current: `1/0/1`.
- Strict median score versus local SciPy 1.17.1 oracle: `1/1/0`.
- Rejected subvariants: compact selected row/column lists (`0/2/0`) and
  remaining-template copy (`0/1/1`). Both are reverted.

| Workload | Baseline Rust | Final Rust | SciPy oracle | Verdict |
| --- | ---: | ---: | ---: | --- |
| `linear_sum_assignment/dense/500` | 20.320 ms | 21.009 ms | 18.906268 ms | neutral vs current; Rust 1.11x slower than SciPy |
| `linear_sum_assignment/dense/1000` | 176.03 ms | 124.20 ms | 125.511679 ms | keep: 1.42x faster than current; Rust 1.01x faster than SciPy |

Guards: focused assignment tests, `cargo check -p fsci-opt --all-targets`,
no-deps clippy, release build, local live SciPy conformance, touched-file
rustfmt with `skip_children=true`, and diff hygiene passed. Changed-file UBS is
blocked by the existing broad `fsci-opt/src/lib.rs` inventory and does not point
at the changed SAP first-scan block. Plain rustfmt over `src/lib.rs` follows
pre-existing `linesearch.rs` child-module drift. The
requested `cargo bench --release` form is invalid Cargo syntax and is recorded
as tooling negative evidence; the valid optimized Criterion route is per-crate
`cargo bench`.

Negative evidence: do not retry selected row/column list maintenance or
remaining-template copy initialization in this SAP loop without fresh
same-worker proof. The remaining n=500 gap needs lower-level dense storage or a
more invasive LAPJV-style kernel, not another whole-vector reset micro-variant.

## 2026-06-20 - frankenscipy-9g6ku - kmeans2 k=4/d=4 fused SIMD assignment

- Agent: cod-a / BlackThrush
- Decision: KEEP the guarded `k=4, d=4` `kmeans2` specialization and the
  generic-loop `vq` bypass. The shipped path flattens observations once,
  computes four centroid distances with `std::simd`, fuses assignment with
  centroid accumulation, and keeps the prior generic route for other shapes.
- Same-worker internal score versus the legacy `vq`-inside-Lloyd route:
  `1/0/0`.
- Strict final Rust versus local SciPy 1.17.1 oracle score: `1/0/0`.
- Rejected/superseded pre-fused candidate score versus SciPy: `0/1/0`.

| Workload | Route | Median | Verdict |
| --- | --- | ---: | --- |
| `kmeans2`, n=2000 k=4 d=4 iter=50 | legacy Rust `vq` loop | 1.1880 ms | fresh baseline; 1.37x faster than SciPy on this worker/dataset |
| `kmeans2`, n=2000 k=4 d=4 iter=50 | pre-fused candidate | 2.2659 ms | reject/supersede: 1.39x slower than SciPy |
| `kmeans2`, n=2000 k=4 d=4 iter=50 | final fused SIMD Rust | 378.67 us | keep: 3.14x faster than legacy; 4.29x faster than SciPy |
| `scipy.cluster.vq.kmeans2` | local SciPy 1.17.1 | 1624.576 us | oracle |

Guards: focused `nearest_centroid_k4_d4` and `kmeans2` unit tests pass via
rch, per-crate `fsci-cluster` build/check pass via rch with only the existing
`perf_kmeans.rs` warning, no-deps library clippy passes on final source, diff
hygiene passes, UBS exits 0 on changed files, the local shared-target
`e2e_cluster` `scenario_01_kmeans` conformance smoke passes, and the exact local SciPy
oracle uses the same deterministic `blobs()`/matrix-init workload as the
Criterion bench. Full crate rustfmt remains blocked by pre-existing
`fsci-cluster/src/lib.rs` drift; all-targets clippy remains blocked by
pre-existing test lints after the new specialization lint was fixed.

Negative evidence: do not retry a standalone SIMD helper that still calls
`vq` and then re-walks labels for centroid sums; the intermediate candidate
was still slower than SciPy. The paying lever is fusing assignment and
accumulation for the fixed small-`k` hot shape.

## 2026-06-20 - spatial pdist sweep routing evidence

- Agent: cod-a / BlackThrush
- Decision: ROUTE ONLY. No spatial source changed in this commit. The fresh
  sweep shows dim-4 Euclidean is closed, but `chebyshev` remains the largest
  measured `pdist` loss and should get the next spatial bead.
- Routing score versus local SciPy oracle: `3/5/0`.

| Workload | Rust | SciPy oracle | Verdict |
| --- | ---: | ---: | --- |
| `pdist/euclidean/n512/d4` | 0.318 ms | 0.375 ms | Rust 1.18x faster |
| `pdist/cityblock/n512/d4` | 0.228 ms | 0.191 ms | Rust 1.19x slower |
| `pdist/sqeuclidean/n512/d4` | 0.209 ms | 0.177 ms | Rust 1.18x slower |
| `pdist/chebyshev/n512/d4` | 2.192 ms | 0.174 ms | Rust 12.60x slower |
| `pdist/euclidean/n4096/d4` | 38.131 ms | 54.682 ms | Rust 1.43x faster |
| `pdist/cosine/n4096/d4` | 62.271 ms | 54.693 ms | Rust 1.14x slower |
| `pdist/chebyshev/n2048/d64` | 72.085 ms | 41.911 ms | Rust 1.72x slower |
| `pdist/cityblock/n2048/d64` | 28.007 ms | 48.630 ms | Rust 1.74x faster |

Negative evidence: do not spend more on dim-4 Euclidean first. The biggest
current gap is `pdist` Chebyshev, especially the d=4 row where Rust is still
over an order of magnitude behind SciPy.

## 2026-06-20 - frankenscipy-8l8r1.135 - filter1d contiguous Reflect direct queue

- Agent: cod-b / BlackThrush
- Decision: KEEP the guarded contiguous `Reflect`, `origin=0`,
  `size <= line_len` direct monotonic queue for public
  `maximum_filter1d` / `minimum_filter1d`. It removes full boundary-resolved
  line materialization and replaces the full-line queue with a `size + 1`
  circular deque while preserving the prior generic queue for every other
  shape/mode/origin.
- Artifact:
  `tests/artifacts/perf/2026-06-20-cod-b-filter1d-specialize/EVIDENCE.md`
- Same-process generic-vs-direct score: `4/0/0`.
- Conservative direct-vs-SciPy score: `4/0/0`.
- Absolute Criterion-after-vs-SciPy score: `4/0/0`; worker differs, so the
  same-process direct A/B is the keep gate and Criterion/SciPy is release
  evidence.

| Workload | Direct queue | Criterion after | SciPy oracle | Verdict |
| --- | ---: | ---: | ---: | --- |
| `maximum_filter1d`, n=65536 size=31 | 470.7 us | 344.48 us | 524.98 us | keep: 2.37x faster than generic; Rust 1.12x to 1.52x faster than SciPy |
| `minimum_filter1d`, n=65536 size=31 | 465.8 us | 339.06 us | 575.42 us | keep: 2.35x faster than generic; Rust 1.24x to 1.70x faster than SciPy |
| `maximum_filter1d`, n=65536 size=101 | 464.2 us | 339.74 us | 529.05 us | keep: 2.35x faster than generic; Rust 1.14x to 1.56x faster than SciPy |
| `minimum_filter1d`, n=65536 size=101 | 466.8 us | 321.55 us | 592.31 us | keep: 2.34x faster than generic; Rust 1.27x to 1.84x faster than SciPy |

Guards: direct/generic bit-identity A/B, existing fold/generic byte identity,
live SciPy `diff_ndimage_filter_1d`, rch `cargo check -p fsci-ndimage
--all-targets`, rch `cargo build --release -p fsci-ndimage`, touched-file
rustfmt, diff hygiene, and changed-file UBS pass. UBS exits 0 with no critical
issues while reporting the broad existing `fsci-ndimage` warning inventory.
Strict clippy remains blocked before this patch on existing `fsci-linalg`
dependency lints.

Negative evidence: do not retry full-line `ext` materialization or whole-line
queue storage for this contiguous Reflect route. Future filter1d work should
move to non-contiguous axes, `size > line_len`, or missing SciPy max/min
filter1d conformance coverage.

## 2026-06-20 - frankenscipy-8l8r1.136 - linear_sum_assignment touched-set dual updates

- Agent: cod-a / BlackThrush
- Decision: REJECT/REVERT the touched-row/touched-column dual update variant
  for the modified Jonker-Volgenant LSAP core. The attempt added sparse
  `sr`/`sc` frontier vectors so dual updates would visit only reached rows and
  columns, but dense workloads paid more for push/indirection than they saved
  in branch scans.
- Artifact:
  `tests/artifacts/perf/2026-06-20-cod-a-lsap-touched-sets/EVIDENCE.md`
- Source status: reverted before commit; `crates/fsci-opt/src/lib.rs` has no
  remaining diff.
- Same-worker touched-set versus current Rust score: `0/1/1`.
- Strict touched-set versus SciPy 1.17.1 oracle score: `0/2/0`.
- Current main versus this SciPy oracle snapshot remains `0/2/0`, but the gap
  is now much narrower than the earlier scorecard: 1.11x and 1.06x slower.

| Workload | Current Rust | Touched-set Rust | SciPy oracle | Verdict |
| --- | ---: | ---: | ---: | --- |
| `linear_sum_assignment/dense/500` | 21.121 ms | 26.212 ms | 19.101180 ms | reject: touched-set 1.24x slower than current; 1.37x slower than SciPy |
| `linear_sum_assignment/dense/1000` | 135.72 ms | 167.30 ms | 127.840366 ms | reject: touched-set 1.23x slower than current; 1.31x slower than SciPy |

Guards: exact source revert check, rch focused assignment tests
(`9 passed`), rch `cargo build --release -p fsci-opt`, local live SciPy
`diff_opt_linear_sum_assignment` conformance (`1 passed`), diff hygiene, and
changed-file UBS on the docs/artifact/beads-only closeout all passed.

Negative evidence: do not retry touched-row/touched-column dual updates for
dense LSAP. The remaining credible strict-SciPy work is a deeper dense storage
or LAP-kernel layout change, not sparse frontier bookkeeping inside the current
row-vector route.

## 2026-06-20 - frankenscipy-zl4m5 - linear_sum_assignment SAP route

- Agent: cod-a / BlackThrush
- Decision: KEEP the SciPy-style modified Jonker-Volgenant shortest
  augmenting path core with owned reusable scratch; REJECT/REVERT the row-major
  flat-cost sub-variant.
- Artifact:
  `tests/artifacts/perf/2026-06-20-cod-a-lsap-zl4m5/EVIDENCE.md`
- Same-worker internal score versus current Rust: `2/0/0`.
- Strict median score versus local SciPy 1.17.1 oracle: `0/2/0`; the SciPy
  gap is narrowed but not closed.
- Rejected sub-variant score: `0/1/1`; flat-cost scratch regressed n=500 by
  1.27x versus the first SAP candidate and did not produce a significant n=1000
  win.

| Workload | Baseline Rust | Final Rust | SciPy oracle | Verdict |
| --- | ---: | ---: | ---: | --- |
| `linear_sum_assignment/dense/500` | 43.798 ms | 28.681 ms | 18.578689 ms | keep: 1.53x faster than current; Rust 1.54x slower than SciPy |
| `linear_sum_assignment/dense/1000` | 349.80 ms | 199.52 ms | 122.932709 ms | keep: 1.75x faster than current; Rust 1.62x slower than SciPy |

Guards: focused `fsci-opt` assignment tests, `cargo check -p fsci-opt
--all-targets`, no-deps clippy, release build, and local live SciPy conformance
all passed. The rch conformance attempt failed before comparison because the
worker Python lacked SciPy. Touched-file rustfmt and diff whitespace checks
passed; full workspace rustfmt remains blocked by pre-existing unrelated
formatting drift. UBS exits nonzero on the existing broad `fsci-opt/src/lib.rs`
inventory (test-only panic callbacks and pre-existing unwrap/assert/indexing
findings), so it is recorded as a scoped blocker rather than folded into this
perf lever.

Negative evidence: do not retry naive per-call row-major flat-cost copying
inside this SAP route without a new way to amortize the n=500 penalty. The next
credible strict-SciPy attack needs deeper dense-matrix layout/API work that
removes row indirection without copying, or a lower-level specialized kernel.

## 2026-06-20 - frankenscipy-8l8r1.133 - linkage compact active frontier

- Agent: cod-a / BlackThrush
- Decision: KEEP the compact active-cluster frontier in the NN-array linkage
  core. It replaces boolean-active range scans with a sorted `active_ids`
  frontier, preserving the exact ascending tie order while skipping inactive
  clusters in pair selection, Lance-Williams updates, and NN refresh.
- Artifact:
  `tests/artifacts/perf/2026-06-20-cod-a-linkage-133/EVIDENCE.md`
- Same-machine internal score versus current: `2/0/0`.
- Same-machine median score versus the local SciPy oracle: `2/0/0`, recorded as
  near-parity/slight wins because the independent Criterion intervals overlap.

| Workload | Baseline Rust | Final Rust | SciPy oracle | Verdict |
| --- | ---: | ---: | ---: | --- |
| `linkage(Average)`, n=800 d=4 | 8.5503 ms | 4.5727 ms | 4.8204 ms | keep: 1.87x faster than current; Rust 1.05x faster than SciPy median |
| `linkage(Ward)`, n=800 d=4 | 10.831 ms | 5.4267 ms | 5.6168 ms | keep: 2.00x faster than current; Rust 1.04x faster than SciPy median |

Guards: focused bit-contract and broad linkage tests via rch, plus live SciPy
linkage conformance for raw linkage helpers and precomputed distances. Full
formatting remains blocked by pre-existing `fsci-cluster` rustfmt drift outside
this scoped patch.

Negative evidence: do not retry full-square arena initialization or lazy-fill
tweaks for this route. The profitable layer is compact nearest-neighbour
maintenance that skips inactive state without changing merge order. Future
linkage work should target a true method-specific NN-chain primitive or smaller
distance frontier.

## 2026-06-20 - frankenscipy-8l8r1.132 - gaussian_filter tile-local scratch

- Agent: cod-a / BlackThrush
- Decision: KEEP the tile-local scratch/cache-blocked separable pass for 2-D
  Reflect/order-0 `gaussian_filter`. The vertical pass now writes each worker
  row chunk into a local scratch tile and immediately runs the horizontal pass
  from that hot tile, removing the full-image scratch buffer and the second
  scoped thread barrier.
- Artifact:
  `tests/artifacts/perf/2026-06-20-cod-a-gaussian-tile-scratch/EVIDENCE.md`
- Same-worker internal score versus current: `1/0/0`.
- Strict SciPy score for final source: `1/0/0`; this flips the tracked
  `gaussian_sigma2/256` row from loss to win.

| Workload | Current Rust | Final Rust | SciPy oracle | Verdict |
| --- | ---: | ---: | ---: | --- |
| `gaussian_sigma2/256`, 2-D Reflect | 1.9819 ms | 1.2274 ms | 1.47367 ms | keep: 1.61x faster than current; Rust 1.20x faster than SciPy |

Guards: focused Gaussian tests and live SciPy conformance pass; rch
`cargo check -p fsci-ndimage --all-targets`, `git diff --check`, and changed-file
UBS pass. Full formatting and strict clippy remain blocked by pre-existing
`fsci-ndimage` rustfmt drift and `fsci-linalg` dependency clippy lints,
respectively.

Negative evidence: do not retry the full-image scratch plus two scoped thread
barriers for this fast path. The remaining plausible work is smaller constant
factor cleanup: source-plan caching, fixed-radius specialization, or deeper
fused/tiled source-plan work with same-worker proof.

## 2026-06-20 - frankenscipy-8l8r1.131 - sparse eigsh projected residual certificate

- Agent: cod-a / BlackThrush
- Decision: KEEP the `k<=6` Arnoldi projected-residual certificate for `eigsh`;
  REJECT the unconditional form because the `k=8` row regressed on the same
  worker. Final source guards `k>6` back to the explicit sparse residual
  matvec check.
- Artifact:
  `tests/artifacts/perf/2026-06-20-cod-a-sparse-eigsh-tridiag/EVIDENCE.md`
- Same-worker internal score versus restored current: `2/1/0` for the raw
  candidate rows; final source keeps the two `k=6` wins and reverts the `k=8`
  regression by guard.
- Prior-ledger SciPy score for the final guarded route: `2/1/0`; the remaining
  tracked loss is `eigsh n=8000 k=6`, narrowed from `1.73x` slower to `1.45x`
  slower on the same-worker acceptance row.
- Fresh local SciPy oracle score for final remote Rust rows: `1/2/0`
  cross-host; recorded as routing evidence because Rust ran on `vmi1152480`
  while SciPy ran locally.

| Workload | Baseline Rust | Final/candidate Rust | SciPy oracle | Verdict |
| --- | ---: | ---: | ---: | --- |
| `eigsh n=2000 k=6` | 1.169 ms | 1.024 ms | 3.000 ms prior oracle | keep: internal 1.14x faster; Rust 2.93x faster than SciPy |
| `eigsh n=8000 k=6` | 4.789 ms | 4.003 ms | 2.768 ms prior oracle | keep: internal 1.20x faster; Rust still 1.45x slower than SciPy |
| `eigsh n=20000 k=8` raw projected candidate | 10.672 ms | 12.289 ms | 43.023 ms prior oracle | reject/guard: 1.15x slower than current despite fewer matvecs |

Negative evidence: do not retry unconditional post-hoc residual removal above
`k=6`, row-major Arnoldi basis arenas, or mutable operator scratch without fresh
same-worker proof. The next credible route is a deeper eigensolver primitive
such as implicit/thick restart or a symmetric tridiagonal-only eigensolve path
for the remaining mid-size `n=8000, k=6` loss.

## 2026-06-20 - frankenscipy-8l8r1.128 - linkage row-pack keep + lazy-arena reject

- Agent: cod-a / BlackThrush
- Decision: KEEP the observation-row packing lever for `linkage`, because Ward
  closes a real internal gap while Average stays neutral/slightly better.
  REJECT AND REVERT lazy full-arena zero initialization, which regressed Average
  and did not move Ward enough to justify shipping.
- Artifact:
  `tests/artifacts/perf/2026-06-20-cod-a-linkage-lazy-arena-EVIDENCE.md`
- Baseline SciPy score for current flat arena: `0/2/0`.
- Lazy-arena candidate/internal score: `0/1/1`; reverted.
- Final row-pack internal score versus current baseline: `1/0/1`.
- Final row-pack strict SciPy score: `0/2/0`.

| Workload | Baseline Rust | Final Rust | SciPy oracle | Verdict |
| --- | ---: | ---: | ---: | --- |
| `linkage(Average)`, n=800 d=4 | 7.1834 ms | 7.1304 ms | 4.3843 ms | internal 1.007x faster; Rust 1.626x slower than SciPy |
| `linkage(Ward)`, n=800 d=4 | 8.2387 ms | 6.9591 ms | 4.8687 ms | internal 1.184x faster; Rust 1.429x slower than SciPy |
| Lazy arena Average candidate | 7.1834 ms current | 7.6203 ms candidate | 4.5097 ms | reject: 1.061x slower than current, 1.690x slower than SciPy |
| Lazy arena Ward candidate | 8.2387 ms current | 8.2002 ms candidate | 5.2550 ms | reject/neutral: 1.005x faster than current, 1.560x slower than SciPy |

Negative evidence: do not retry zero/lazy initialization of the full
inter-cluster arena on this NN-array linkage route. The profitable constant
factor was packing the nested observations once before pairwise distance
construction. Further work needs to change the nearest-neighbour maintenance
or method-specific clustering primitive rather than another full-square arena
initialization tweak.
## 2026-06-20 - frankenscipy-8l8r1.129 - gaussian_filter 2D reflect cache-planned separable pass

- Agent: cod-b / MistyBirch
- Decision: KEEP as a measured same-worker Rust speedup and residual SciPy
  loss. The fast path improves the tracked `gaussian_sigma2/256` workload by
  `1.68x` on the same `vmi1152480` worker, but final Rust remains `1.34x`
  slower than the SciPy oracle.
- Artifact:
  `tests/artifacts/perf/frankenscipy-8l8r1.128-gaussian-cache-planned/EVIDENCE.md`
- Same-worker internal score versus clean `ae454655` current: `1/0/0`.
- Strict SciPy score for final source: `0/1/0`.

| Route | Worker | Mean | Ratio |
| --- | --- | ---: | ---: |
| Clean current Rust (`ae454655`) | `vmi1152480` | 3.2989 ms | 2.25x slower than SciPy |
| Candidate Rust | `vmi1152480` | 1.9680 ms | 1.34x slower than SciPy; 1.68x faster than current |
| SciPy `ndimage.gaussian_filter` | local oracle | 1.46523 ms | oracle |

Routing-only rows: pre-edit RCH baseline on `vmi1227854` was `2.8418 ms`
(`1.94x` slower than SciPy); clean baseline on `vmi1149989` was `5.8852 ms`
(`4.02x` slower than SciPy). These were not used for the keep/reject ratio
because they were not same-worker paired with the candidate.

Negative evidence: cache-planned 2-D separable source-index tables remove enough
generic N-D filter overhead to keep, but not enough to beat SciPy. Do not retry
the reverted scalar row-contiguous border/interior split. Route next to
vectorized row/column dot kernels, transposed scratch for the vertical pass, or
cache-blocked separable tiles that preserve the same reflect index plan.

## 2026-06-20 - frankenscipy-8l8r1.130 - gaussian_filter folded AXPY reflect pass

- Agent: cod-b / MistyBirch
- Decision: KEEP as a measured internal win and residual SciPy loss. The folded
  symmetric AXPY path improves the tracked `gaussian_sigma2/256` workload by
  `2.05x` in a paired Criterion row on `vmi1167313`, and by `1.22x` in an
  interleaved same-process A/B toggle. Final Rust remains slower than SciPy.
- Artifact:
  `tests/artifacts/perf/frankenscipy-8l8r1.130-gaussian-axpy/EVIDENCE.md`
- Same-worker internal score versus clean `0cf3cc42` current: `1/0/0`.
- Same-process A/B score versus gather path: `1/0/0`.
- Strict SciPy score for final source: `0/1/0`.

| Route | Worker | Mean | Ratio |
| --- | --- | ---: | ---: |
| Clean current Rust (`0cf3cc42`) | `vmi1167313` | 6.9394 ms | baseline |
| Candidate AXPY Rust | `vmi1167313` | 3.3918 ms | 2.05x faster than current; 2.91x slower than SciPy |
| Same-process gather toggle | `vmi1167313` | 3585.0 us | baseline arm |
| Same-process AXPY toggle | `vmi1167313` | 2943.3 us | 1.22x faster than gather |
| Final-source routing sanity | `vmi1149989` | 3.0285 ms | 2.59x slower than SciPy; routing-only |
| SciPy `ndimage.gaussian_filter` | local oracle | 1.16724 ms | oracle |

Negative evidence: folded row AXPY removes part of the strided-gather penalty,
but still leaves the horizontal pass as a gather over scratch. Do not retry
scalar reflect tap peeling or always-line-walk outer-axis variants. Route next
to transposed scratch/cache-blocked tiles so both separable passes become
stride-1 row work, then remove the runtime test toggle if that pays.

## 2026-06-20 - frankenscipy-8l8r1.127 - EDT feature-transform line starts

- Agent: cod-b / MistyBirch
- Decision: KEEP as a measured same-worker internal win and partial SciPy gap
  close. Strict SciPy score improves to `1/3/0` for the measured rows, but the
  sub-cluster remains a release-readiness loss overall.
- Artifact:
  `tests/artifacts/perf/frankenscipy-8l8r1.127-edt-line-starts-EVIDENCE.md`
- Same-worker rch internal score versus the prior feature-transform route:
  `4/0/0`.
- Strict SciPy score for final source: `1/3/0`.

| Image | Prior Rust | Final Rust | SciPy `return_indices` | Verdict |
| --- | ---: | ---: | ---: | --- |
| 64x64 | 325.742 us | 216.733 us | 173.434 us | internal 1.50x; Rust 1.25x slower |
| 128x128 | 1.380 ms | 1.207 ms | 775.685 us | internal 1.14x; Rust 1.56x slower |
| 192x192 | 3.814 ms | 2.107 ms | 2.280155 ms | internal 1.81x; Rust 1.08x faster |
| 256x256 | 5.854 ms | 4.855 ms | 4.288605 ms | internal 1.21x; Rust 1.13x slower |

Negative evidence: exact line-start enumeration and no per-cell coordinate
allocation help, but they do not fully beat SciPy's compiled C feature
transform. Do not retry flat-index scan filtering or per-cell `unravel`/Vec
allocation in this path; route next to deeper feature-transform constants such
as fused axis passes, scratch layout, SIMD-friendly 1-D lower-envelope work, or
tile-specialized 2-D kernels with the same nearest-background proof.

## 2026-06-20 - frankenscipy-6l77z - gaussian_filter inner1 reflect reject

- Agent: cod-a / MistyBirch
- Decision: REJECT AND REVERT. The row-contiguous reflect/origin-zero direct
  interior dot specialization regressed `gaussian_sigma2/256` on the same rch
  worker.
- Artifact:
  `tests/artifacts/perf/2026-06-20-ndimage-gaussian-inner1-reflect-reject/EVIDENCE.md`
- Same-worker candidate/current score: `0/1/0`.
- Final restored current/SciPy score: `0/1/0`.

| Route | Mean | Ratio |
| --- | ---: | ---: |
| Current Rust on rch `hz2` | 3.4399 ms | 3.03x slower than SciPy |
| Candidate Rust on rch `hz2` | 4.0213 ms | 1.17x slower than current; 3.54x slower than SciPy |
| SciPy `ndimage.gaussian_filter` | 1.13557 ms | oracle |

Negative evidence: do not retry scalar row-contiguous reflect-only interior
tap peeling for this workload without a fresh profile; route next to
transpose/cache-tiled separable layout or a shared vector-friendly dot kernel.

## 2026-06-20 - frankenscipy-8l8r1.126 - label mean one-based contiguous index

- Agent: cod-b / MistyBirch
- Decision: KEEP as a measured internal win and partial SciPy gap close.
- Artifact:
  `tests/artifacts/perf/frankenscipy-8l8r1.126-label-mean-one-based-EVIDENCE.md`
- Strict SciPy score: `1/3/0`.
- Internal same-host score versus the prior dense-table route: `4/0/0`.
- Same-host SciPy rows:

| N | K | Rust one-based | SciPy `ndimage.mean` | Ratio |
| ---: | ---: | ---: | ---: | --- |
| 65536 | 512 | 153.257 us | 0.189 ms | Rust 1.23x faster |
| 262144 | 1024 | 634.996 us | 0.585 ms | Rust 1.09x slower |
| 262144 | 2048 | 687.054 us | 0.576 ms | Rust 1.19x slower |
| 589824 | 4096 | 1.423 ms | 1.380 ms | Rust 1.03x slower |

Negative evidence: do not retry dense-table, `fract()`, `is_finite()`, HashMap,
or `Vec<Vec<f64>>` grouping variants for this workload without a fresh profile.
Next attempts should target deeper reduction primitives such as
parallel/cache-tiled sum/count accumulation or sorted/run-grouped label spans.

## 2026-06-20 - frankenscipy-5smr3 - ndimage min/max filter van Herk/Gil-Werman (WIN, byte-identical)

- Agent: cc / MistyBirch
- Decision: **KEEP**. Replace the per-line monotonic-deque sliding min/max
  (`VecDeque` alloc + pointer-chase + variable `total_cmp` evictions, scanning
  every flat index to find line heads) with van Herk / Gil-Werman block
  prefix/suffix scans over a materialized, boundary-resolved line, plus an
  in-bounds interior fast path (contiguous `copy_from_slice` when stride==1,
  strided direct read otherwise) that skips the per-element `boundary_index_1d`
  match for the `mid`-cell interior. Lines addressed directly (outer × inner).
- Correctness: **byte-for-bit identical** to the deque path — same `total_cmp`
  total order (min/max element bits are uniquely determined, incl. NaN / ±0.0 /
  ±inf), same neighbourhood mapping. Proven by `minmax_hgw_byte_identical_to_deque`
  (lib unit test, 1 passed) across ndim ∈ {1,2,3}, size ∈ {1,2,3,5,8}, all valid
  origins, all 5 boundary modes, min & max, with adversarial NaN/±0/±inf data.
- A/B: in-process atomic toggle (`MINMAX_FILTER_HGW`) interleaved OFF/ON so fleet
  load cancels (the only reliable method under multi-agent contention).
- Conformance: zero new failures. The `diff_ndimage_morph_filters`,
  `diff_ndimage_filters_edges`, `diff_ndimage_grey_morphology`,
  `diff_ndimage_filter_1d`, and `diff_ndimage` live_scipy tests fail identically
  on clean `origin/main` (no scipy on the rch workers under
  `FSCI_REQUIRE_SCIPY_ORACLE=1`) — verified by stash-and-rerun. `maximum_filter1d`
  uses a separate (`filter1d_axis_with_origin`) path that this change does not touch.

| Workload (256×256, Reflect) | deque (same-proc A/B) | HGW (same-proc A/B) | self-speedup |
| --- | ---: | ---: | ---: |
| `maximum_filter` size=7  | 1484.7 us | 630.2 us | **2.36x faster** |
| `maximum_filter` size=15 | 1520.9 us | 659.4 us | **2.31x faster** |
| `maximum_filter` size=31 | 1608.5 us | 692.9 us | **2.32x faster** |

| Workload (standalone criterion, rch worker) | deque baseline | HGW | self | scipy (local, diff CPU) |
| --- | ---: | ---: | ---: | ---: |
| `maximum_256x256/31` | 1.567 ms | 0.946 ms | 1.66x | 0.820 ms |
| `minimum_256x256/31` | ~1.5 ms | 0.904 ms | — | ~0.82 ms |
| `maximum_256x256/7`  | ~1.48 ms | 0.873 ms | — | 0.784 ms |

Score: self-speedup `3/0/0` (load-canceling A/B, byte-identical). vs SciPy: the
documented `minimum/maximum_filter` 1.8-1.9x loss closes to near-parity (cross-box
standalone ~1.1x; load-canceling A/B absolutes beat scipy). Reusable lever:
**any monotonic-deque sliding-window extremum → van Herk block prefix/suffix +
interior-direct (boundary-map only the ~window-1 edge cells).**

## 2026-06-20 - gaussian_filter 2D reflect folded symmetric axpy - REJECT (bandwidth-bound)

- Agent: cc / MistyBirch
- Decision: **REJECT AND REVERT**. Reformulated `gaussian_filter_2d_reflect_order0`
  to exploit the (bit-symmetric, order-0) kernel: fold symmetric pairs
  `w[mid]*x[mid] + Σ w[mid±k]*(x[+k]+x[-k])` (scipy correlate1d order, halves the
  multiplies) AND restructure the row (axis-0) pass as contiguous **axpy** passes
  (stride-1, vectorizable) instead of the stride-`cols` gather, plus a
  reflect-free interior axpy for the col pass.
- Correctness: tolerance-equal to the gather-dot path (exact reordering of the
  same operands; `max|gather-axpy| < 1e-10` across rows/cols/sigma) — proven, but
  NOT byte-identical (reordered FP accumulation).
- Measured (same-process atomic-toggle A/B, load-canceling): row-axpy alone
  **1.16x**; + col-axpy interior **1.18x**. Below the 1.3x keep threshold.
- Root cause: the 256×256×f64 separable pass streams ~512 KB twice — it is
  **memory-bandwidth-bound**, not multiply-bound, so halving the FMAs and
  vectorizing the inner loop cannot close the gap. The residual vs SciPy
  (~1.13 ms) is constant-factor overhead (double buffering, two thread scopes,
  source-plan precompute), not the inner dot.
- Negative evidence: do NOT retry inner-loop fold / axpy / SIMD on the gaussian
  separable pass — the bottleneck is memory traffic + per-call overhead. A real
  flip needs a single fused streaming pass (fewer buffer touches) or a tiled
  cache-blocking that keeps the working set resident, AND must clear ≥1.3x on a
  same-process A/B. Consistent with the prior `6l77z`/`acdq2` direct-interior
  rejects. Reverted to `0cf3cc42`; no source shipped.

## 2026-06-20 - frankenscipy filter1d van Herk routing - KEEP (4-7x self, residual loss)

- Agent: cc / MistyBirch
- Decision: **KEEP**. Route `maximum_filter1d` / `minimum_filter1d` through the
  O(n) van Herk / Gil-Werman block prefix-suffix kernel (`minmax_along_axis_hgw`)
  with a NaN-propagating op, replacing the O(n·size) per-window fold
  (`filter1d_axis_with_origin`) that also allocated a coordinate Vec and a window
  Vec per output pixel.
- Correctness: **byte-for-bit identical** to the fold — the NaN-propagating max/min
  is associative + idempotent, so the HGW reassociation reproduces the per-window
  fold exactly (extremum is one of the inputs, NaN propagates regardless of
  grouping). Proven by `filter1d_hgw_byte_identical_to_fold` across ndim {1,2,3},
  all axes, sizes {1,2,4,5,n+3} (incl. window > axis length), origins, all 5
  boundary modes, min & max, with NaN/±0/±inf data.

| Workload (n=65536, Reflect, same-proc A/B) | old fold (O(n·size)) | new HGW (O(n)) | self-speedup |
| --- | ---: | ---: | ---: |
| `maximum_filter1d` size=31  | 4907.3 us | 1191.5 us | **4.12x faster** |
| `maximum_filter1d` size=101 | 8729.0 us | 1179.5 us | **7.40x faster** |

- vs SciPy `maximum_filter1d` (≈516 us, O(n)): the loss closes from ~9.5x slower
  (size 31) / ~16.8x slower (size 101) to a **constant ~2.3x** — the old path grew
  with `size`, the new path is flat (1191 vs 1180 us). Residual `0/1/0` vs SciPy.
- Negative evidence / next: the residual ~2.3x is HGW's 4 passes (ext materialize
  + prefix g + suffix h + combine) and 3 per-call buffers for a single long line
  vs SciPy's tighter in-place pass. A further flip needs pass fusion or chunked
  parallelism of the single 1-D line; the routing here is the byte-identical
  asymptotic fix and is kept regardless.

## 2026-06-20 - filter1d HGW within-line parallelism - REJECT (bandwidth-bound + spawn overhead)

- Agent: cc / MistyBirch
- Decision: **REJECT AND REVERT**. Parallelize the van Herk passes WITHIN a single
  long line (block prefix/suffix across independent blocks, then combine across
  output chunks) to close the filter1d residual ~2.3x vs SciPy. Gated to
  `stride==1 && mid>=16384 && lines<=2`; byte-identical to serial (proven by
  `filter1d_hgw_parallel_byte_identical_to_serial`, all sizes/min-max/NaN).
- Measured (same-process A/B, n=65536): size=31 **0.58x**, size=101 **0.64x** —
  the parallel path is ~1.5 ms vs serial ~0.87-0.92 ms. SLOWER.
- Root cause: each HGW pass touches only ~512 KB (memory-bandwidth-bound, not
  compute-bound), and the two `thread::scope` barriers spawn ≤16 threads twice
  (~32 spawns) whose overhead + cross-core memory traffic exceed the serial pass.
  Parallelizing a bandwidth-bound 3-pass kernel over one 0.5 MB line does not pay.
- Negative evidence: do NOT parallelize within a single van Herk line. The
  filter1d residual vs SciPy (serial HGW's 4 passes + 3 buffers vs SciPy's single
  in-place pass) needs PASS FUSION (fewer streams over the line), not threads.
  The shipped serial routing (ce1857ab, 4-7x self-win) is kept as-is.

## 2026-06-20 - frankenscipy-8l8r1.134 - filter1d fused monotonic queue - KEEP (internal win, residual SciPy loss)

- Agent: cod-b / BlackThrush
- Decision: **KEEP**. Replace the public `maximum_filter1d` /
  `minimum_filter1d` HGW prefix-suffix route with a single-pass monotonic index
  queue over the same boundary-resolved line. This is the pass-fusion direction
  requested by the prior negative evidence, not another within-line threading
  attempt.
- Correctness: **byte-for-bit identical** to the fold/HGW route. The focused
  `filter1d_hgw_byte_identical_to_fold` test passed via rch `vmi1149989`, and
  the same-process `filter1d_queue_vs_hgw_ab_timing` test compares HGW and queue
  outputs bit-for-bit before timing. Local live SciPy conformance
  `diff_ndimage_filter_1d` also passed 1/0; that target currently exercises
  `uniform_filter1d` only because max/min filter1d SciPy boundary parity is
  already documented as out of scope there.

| Workload (n=65536, Reflect) | HGW baseline (`hz2`) | queue final (`hz2`) | internal ratio | SciPy 1.17.1 median | final vs SciPy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `maximum_filter1d` size=31 | 1.2413 ms | 0.56072 ms | **2.21x faster** | 0.51803 ms | 1.08x slower |
| `minimum_filter1d` size=31 | 1.0365 ms | 0.76956 ms | **1.35x faster** | 0.54051 ms | 1.42x slower |
| `maximum_filter1d` size=101 | 1.0385 ms | 0.82422 ms | **1.26x faster** | 0.51482 ms | 1.60x slower |
| `minimum_filter1d` size=101 | 1.0234 ms | 0.77760 ms | **1.32x faster** | 0.54355 ms | 1.43x slower |

- Same-process release A/B on rch `hz2`: queue/HGW was 1.15x / 1.17x / 1.21x /
  1.21x faster for max31/min31/max101/min101, all bit-identical.
- Gates: rch `cargo check -p fsci-ndimage --all-targets` passed; touched-file
  rustfmt, `git diff --check`, UBS, local live SciPy conformance, and focused
  correctness tests passed. Strict dependency-inclusive clippy remains blocked
  before this patch on existing `fsci-linalg` lints (`needless_range_loop`,
  `needless_borrow`).
- Negative evidence / next: this narrows the residual from the HGW flat ~2.0x
  loss to 1.08-1.60x slower, but does not yet dominate SciPy. Do not call this
  closed. Next routes should target branch-reduced NaN-free contiguous-line
  specialization, boundary-free interior fast paths, or SIMD/block-merge designs
  that preserve NaN propagation and signed-zero newest-tie semantics.

## 2026-06-20 - rfft measured head-to-head vs numpy - MIXED (stale "loss" corrected; mid-size kernel wall)

- Agent: cc / MistyBirch
- Finding (MEASURED, rch vs numpy.fft.rfft same sizes): the believed "rfft ~1.73x
  slower, no real-symmetry path" is STALE — `real_fft_specialized` already packs N
  reals into an N/2-point complex FFT (irfft too). Real state is mixed:

| n | fsci rfft | numpy.rfft | ratio |
| ---: | ---: | ---: | --- |
| 256   | 2.00 us | 4.24 us | **2.12x faster** |
| 1024  | 6.54 us | 6.90 us | 1.05x (parity) |
| 4096  | 27.2 us | 18.6 us | 0.68x (**1.46x slower**) |
| 16384 | 122 us  | 77.8 us | 0.64x (**1.57x slower**) |
| 65536 | 600 us  | 754 us  | **1.26x faster** |

- fsci WINS small (overhead-light) and very-large; LOSES the mid pow2 range
  (4096/16384) by ~1.5x. Root cause: the half-size complex FFT kernel quality
  (fsci radix-2²/radix-4 vs pocketfft split-radix + cache blocking) — and the
  obvious kernel levers (radix-8, four-step transpose) were already MEASURED and
  REJECTED (cache thrash) in `perf_fft_radix4_stage_fusion`. So the mid-size rfft
  loss is the known FFT-kernel wall, not a missing real-FFT path.
- Action: EXPANDED `bench_rfft` to cover 256→65536 (was capped at 1024, which hid
  this entire regime — a benchmark coverage gap). No source change; the residual
  is documented as the FFT-kernel wall. Do NOT re-chase "native real-FFT" (done)
  or radix-8/four-step (rejected); a real flip needs a split-radix kernel rewrite.

## 2026-06-20 - high-dim pdist euclidean - MEASURED WIN (fsci dominates 2.6-5.3x; GEMM lever not needed)

- Agent: cc / MistyBirch
- Finding (MEASURED, rch fsci vs scipy.spatial.distance.pdist euclidean): the
  high-dim (d >> 4) regime was UNBENCHMARKED — the spatial bench only covered
  dim-4. fsci already DOMINATES scipy via 64-thread parallel-over-pairs:

| n, d | fsci pdist | scipy pdist | ratio |
| --- | ---: | ---: | --- |
| n=1000 d=64  | 2.42 ms | 8.54 ms  | **3.5x faster** |
| n=2000 d=64  | 6.80 ms | 34.4 ms  | **5.1x faster** |
| n=1000 d=128 | 3.05 ms | 16.3 ms  | **5.3x faster** |
| n=2000 d=16  | 4.41 ms | 11.5 ms  | **2.6x faster** |

- The RADICAL candidate (euclidean-via-GEMM, `‖x-y‖²=‖x‖²+‖y‖²-2x·y` as BLAS-3)
  was considered: it would further self-speed the kernel, but vs SciPy this is
  NOT a gap — fsci already wins 2.6-5.3x because scipy's pdist is single-threaded
  C while fsci parallelizes the O(n²) pair work across 64 cores. Per "target gaps
  where we LOSE", no source change ships. (GEMM would also risk close-point
  cancellation accuracy vs scipy's direct sqrt(Σ(a-b)²); deferred unless a
  same-thread-count loss is ever found.)
- Action: added `bench_pdist_highdim` (n/d ∈ {1000/64, 2000/64, 1000/128,
  2000/16}) so this winning regime has permanent regression coverage.

## 2026-06-20 - CubicSpline eval_many 100k - MEASURED WIN (fsci 7.1x faster), coverage added

- Agent: cc / MistyBirch
- MEASURED (rch vs scipy.interpolate.CubicSpline.__call__): scipy evals 100k
  query points (1024-knot spline) in **5985 us**; fsci `CubicSplineStandalone::
  eval_many` does it in **843 us = 7.1x faster** (4096 pts: 81 us). Sequential Rust
  (binary search + cubic per point) already beats scipy's per-call Python/numpy
  overhead by 7x — not a gap. Parallelizing the query loop is NOT pursued (already
  winning; and a prior NdPPoly/BPoly evaluate_many parallel attempt was REVERTED at
  0.88x — query-parallel doesn't pay for this cheap per-point kernel).
- Action: extended `bench_splines` with a 100k-point case so the large-batch
  eval regime (was only 4096) has regression coverage.

## 2026-06-20 - KDTree query_many parallel batch - WIN (2.2-2.5x self, 4.8x vs scipy, byte-identical)

- Agent: cc / MistyBirch
- Decision: **KEEP**. Added `KDTree::query_many` — a parallel batch nearest-
  neighbour query (matches `scipy.spatial.cKDTree.query(X, k=1)`). fsci had only a
  single-point `query`; the bench mapped it sequentially. Each query is an
  independent read-only `nn_search` traversal, so the batch parallelizes across
  query points; tree traversal is LATENCY-bound (pointer-chasing), so — unlike the
  bandwidth-bound 1-D scans that regressed — the fan-out scales.
- Correctness: **byte-for-bit identical** to per-point `query` (same traversal +
  sqrt, input order), proven by `kdtree_query_many_matches_per_query` across
  d∈{2,3,8}, batch sizes spanning the serial/parallel gate, + error propagation.
  Full fsci-spatial lib suite 209/0.

| Workload (n=10000) | seq (per-query) | query_many (parallel) | self | vs scipy cKDTree |
| --- | ---: | ---: | ---: | ---: |
| query k=1, d=3 | 2.71 ms | 1.23 ms | **2.2x** | 5.95 ms → **4.8x faster** |
| query k=1, d=8 | 3.25 ms | 1.32 ms | **2.5x** | (scipy randn d=8 108ms — data differs) |

- Scaling is modest (~2.2x not 16x) because each query is cheap (~0.27us) and the
  shared-tree traversal is memory-latency-bound; still strictly positive and
  byte-identical. LEVER: parallelize independent COMPUTE/LATENCY-bound batches
  (tree queries, root-finds) — distinct from bandwidth-bound 1-D scans which wall.

## 2026-06-20 - KDTree query_k_many parallel batch k-NN - WIN (4.8-5.1x self, 8.9x vs scipy, byte-identical)

- Agent: cc / MistyBirch
- Decision: **KEEP**. Added `KDTree::query_k_many` — parallel batch k-NN matching
  `scipy.spatial.cKDTree.query(X, k)`. Generalizes the `query_many` lever to k>1:
  each query runs the same independent read-only `knn_search` + total_cmp sort +
  sqrt as `query_k`, parallelized across the query batch.
- Correctness: **byte-for-bit identical** to per-point `query_k` (same neighbours,
  order, distance bits) — proven by `kdtree_query_k_many_matches_per_query` across
  d∈{2,3,8}, k∈{1,5,12}, batch sizes across the gate, k=0 and error propagation.
  Full fsci-spatial lib suite 210/0.

| Workload (n=10000, k=10) | seq (per-query) | query_k_many | self | vs scipy cKDTree |
| --- | ---: | ---: | ---: | ---: |
| query k=10, d=3 | 10.1 ms | 2.09 ms | **4.8x** | 18.6 ms → **8.9x faster** |
| query k=10, d=8 | 11.9 ms | 2.35 ms | **5.1x** | (scipy randn 263ms; seq alone 22x) |

- k-NN's heavier per-query work (bounded heap + sort over k) amortizes spawn far
  better than k=1 (query_many got 2.2x) → 4.8-5.1x. Confirms the lever: the
  heavier the independent per-element compute, the better the batch fan-out scales.

## 2026-06-20 - KDTree query_ball_point_many parallel batch radius query - WIN (7.9x self, byte-identical)

- Agent: cc / MistyBirch
- Decision: **KEEP**. Added `KDTree::query_ball_point_many` — parallel batch radius
  query matching `scipy.spatial.cKDTree.query_ball_point(X, r)`. Third application
  of the batch-parallel lever: each query runs the same independent read-only
  `ball_search` + `sort_unstable` as `query_ball_point`, parallelized across points.
- Correctness: **byte-for-bit identical** to per-point `query_ball_point` (same
  sorted index lists) — proven by `kdtree_query_ball_point_many_matches_per_query`
  across d∈{2,3}, r∈{0.1,0.5,1.5}, batch sizes across the gate, empty/error paths.
  Full fsci-spatial lib suite 211/0.

| Workload (n=10000, d=3, r=0.3) | seq (per-query) | ball_many | self-speedup |
| --- | ---: | ---: | ---: |
| query_ball_point | 205.5 ms | 25.9 ms | **7.9x** |

- BEST scaling of the three batch APIs (query_many 2.2x, query_k_many 4.8x, ball
  7.9x): the radius query is the HEAVIEST per-query (touches many nodes + sorts a
  large hit list), so it amortizes spawn best — monotone confirmation that batch
  fan-out scales with per-element work. vs scipy cKDTree.query_ball_point
  (uniform data, similar n/d, ~156-236ms) the win is ~6-9x but data distributions
  differ, so the rigorous claim is the same-data 7.9x self-speedup.

## 2026-06-20 - Delaunay find_simplex_many - MARQUEE WIN (flips ~30-48x LOSS to 2.7-3.0x faster than scipy)

- Agent: cc / MistyBirch
- Decision: **KEEP**. fsci's single-point `Delaunay::find_simplex` is an O(num_
  simplices) LINEAR SCAN with a barycentric test per triangle — a SEVERE loss:
  sequential over 50000 queries it takes 1.13s (npts=2000) / 3.13s (npts=5000),
  i.e. ~30-48x SLOWER than scipy.Delaunay.find_simplex (37.7/65.6ms, which walks).
  Added `find_simplex_many` (batch, matches `scipy.spatial.Delaunay.find_simplex(X)`)
  with two amortized accelerations: (1) precompute each triangle's PADDED AABB once
  per batch and cheap-reject before the barycentric test (pad 1e-8·extent safely
  dominates the 1e-10 barycentric tolerance → never skips a containing triangle);
  (2) parallelize the independent per-point scans.
- Correctness: **byte-for-bit identical** to per-point `find_simplex` (same lowest-
  index simplex, identical barycentric bits) — proven by
  `delaunay_find_simplex_many_matches_per_point` incl. interior/exterior/on-vertex
  queries across the serial/parallel gate. Full fsci-spatial lib suite 212/0.

| Workload (50000 queries) | seq linear-scan | find_simplex_many | self | vs scipy |
| --- | ---: | ---: | ---: | ---: |
| npts=2000 | 1127 ms | 13.95 ms | **80.8x** | 37.7 ms → **2.7x faster** |
| npts=5000 | 3128 ms | 21.9 ms  | **142.8x** | 65.6 ms → **3.0x faster** |

- The bbox prefilter (kills barycentric for non-candidate triangles) is most of the
  win; parallelism stacks on top. Feeds griddata / LinearNDInterpolator point
  location. NEXT (perf_precompute_per_element_predicate): a uniform grid over
  triangle bboxes would make EACH query O(1) (the scan is still O(num_simplices)
  cheap bbox checks) — a further flip, but find_simplex_many already dominates.

## 2026-06-20 - Delaunay find_simplex_many GRID index - WIN (14.7-16.4x faster than scipy, byte-identical)

- Agent: cc / MistyBirch
- Decision: **KEEP**. Stacked a uniform spatial grid on `find_simplex_many`
  (the precompute-per-element-predicate lever from [[perf_precompute_per_element_predicate]]):
  bin each triangle, in ASCENDING index order, into every grid cell its padded
  bbox overlaps; a query scans only its own cell's (sorted) candidate list and
  returns the first containing triangle. The cell list is a superset of every
  triangle whose padded bbox contains the query point, so the lowest-index hit is
  **bit-for-bit identical** to the O(num_simplices) bbox linear scan. Degenerate /
  small (ns<64) inputs use g=1 (single cell = the full scan). g=ceil(sqrt(ns)),
  capped 1024. Proven by the existing `delaunay_find_simplex_many_matches_per_point`
  (interior/exterior/on-vertex); full fsci-spatial lib suite 212/0.

| Workload (50000 queries) | bbox scan (prev) | + GRID | grid gain | vs scipy |
| --- | ---: | ---: | ---: | ---: |
| npts=2000 | 13.95 ms | 2.57 ms | **5.4x** | 37.7 ms → **14.7x faster** |
| npts=5000 | 21.9 ms  | 4.00 ms | **5.5x** | 65.6 ms → **16.4x faster** |

- Cumulative: the original single-point linear scan was ~30-48x SLOWER than scipy
  (1.1-3.1s); find_simplex_many + grid is now **14.7-16.4x FASTER** — a ~450-770x
  swing, byte-identical. Each query is now O(1) (cell candidates) instead of
  O(num_simplices). Feeds griddata / LinearND point location.

## 2026-06-20 - KDTree sparse_distance_matrix parallel collection - WIN (1.31-1.34x vs scipy, byte-identical)

- Agent: cc / MistyBirch
- Decision: **KEEP**. `sparse_distance_matrix_triplets` was the lone SEQUENTIAL
  outlier among its siblings (`query_ball_tree`, `count_neighbors` are already
  parallel): it looped over `self.nodes` calling `other.query_ball_point` per
  point. Parallelized the outer loop (chunk `self.nodes`, collect per-thread,
  concat) — the entries are sorted by (row,col) with UNIQUE keys at the end, so
  the result is independent of thread/collection order = byte-identical.
- Correctness: proven by `sparse_distance_matrix_triplets_matches_brute_force` —
  parallel output equals an all-pairs brute-force reference bit-for-bit (incl.
  distance bits) at n=1500 (above the parallel gate). Full lib suite 213/0.

| Workload (cross-tree, d=2) | fsci (parallel) | scipy cKDTree | ratio |
| --- | ---: | ---: | --- |
| n=5000, r=0.05 (nnz~188k)  | 60.3 ms | 79.1 ms  | **1.31x faster** |
| n=10000, r=0.04 (nnz~486k) | 157 ms  | 211 ms   | **1.34x faster** |

- Modest because the final (row,col) sort and the DOK (HashMap) build are
  sequential and now dominate; the parallelized collection is a fraction of the
  total. Still a strict, byte-identical improvement that removes the
  sequential-outlier inconsistency and beats scipy end-to-end. Further gain would
  need a parallel sort / parallel DOK assembly.

## 2026-06-20 - theilslopes/siegelslopes MEASURED WIN (8-54x vs scipy); build-parallel = no-op reject

- Agent: cc / MistyBirch
- Finding (MEASURED, rch vs scipy.stats.theilslopes/siegelslopes — both O(n²) in
  scipy's C): fsci already DOMINATES. Added a `robust_slopes` bench (was uncovered).

| Op (n) | fsci | scipy | ratio |
| --- | ---: | ---: | --- |
| theilslopes n=2000  | 9.81 ms | 81.9 ms | **8.3x faster** |
| theilslopes n=4000  | 36.9 ms | 365 ms  | **9.9x faster** |
| siegelslopes n=2000 | 3.10 ms | 71.0 ms | **22.9x faster** |
| siegelslopes n=4000 | 5.53 ms | 297 ms  | **53.7x faster** |

- siegelslopes wins via its already-parallel per-anchor repeated-median;
  theilslopes wins via the count-based (`count_le`) rank selection / O(n log n)
  inversion fast path (falling back to materialized O(n²) only for x-ties).
- REJECT (reverted): parallelizing the `theilslopes_materialized` O(n²) slope BUILD
  (byte-identical — slopes feed only multiset median/rank statistics). Measured
  no-op: n=4000 36.9→34.1 ms (<8%, noise). The materialized fallback is not the hot
  path for distinct-x data (the count-based fast path is), and even when it is, the
  `select_nth`/median dominate the build. Don't parallelize the build alone. Only
  the bench (regression coverage) is kept.

## 2026-06-20 - linear_sum_assignment - MEASURED LOSS 5.6-7.4x (algorithmic; LAPJV port is the fix)

- Agent: cc / MistyBirch
- Finding (MEASURED, rch vs scipy.optimize.linear_sum_assignment, continuous-cost
  dense matrices to match scipy's uniform input — NOT tie-heavy, which masks it):

| n×n | fsci | scipy | ratio |
| --- | ---: | ---: | --- |
| 500  | 40.4 ms | 7.2 ms  | **5.6x SLOWER** |
| 1000 | 279 ms  | 37.6 ms | **7.4x SLOWER** |

- Root cause is ALGORITHMIC, not memory: fsci's `hungarian_rectangular` is the
  basic e-maxx O(n³) Hungarian — every augmenting step does an O(n) `used`-skipping
  column rescan AND an O(n) delta sweep (`minv[col] -= delta` over all cols), and
  `minv`/`used` are re-allocated per row. SciPy uses LAPJV (Crouse 2016):
  column-reduction + augmenting-row-reduction WARMSTART (assigns most rows cheaply →
  few expensive augmenting paths) with LAZY dual updates and a shrinking
  remaining-column list.
- REJECT (reverted, no-op): flattening the `&[Vec<f64>]` cost matrix to a
  contiguous buffer + hoisting `u[row0]`/row-slice — measured 14.2→16.6 ms / 80→84
  ms (slightly WORSE; the per-row Vecs weren't the bottleneck). The matrix layout is
  not the gap.
- FIX RECIPE (deferred — substantial, conformance-critical port): port scipy's exact
  `rectangular_lsap.cpp` LAPJV. Byte-identity is SAFE for continuous costs (the
  optimum is unique) — verify via cost-equality (sum) vs the current Hungarian, not
  assignment-equality (ties give non-unique assignments). A cheaper byte-identical
  partial: shrinking remaining-col list + LAZY delta (track a running offset instead
  of the O(n) per-step `minv -= delta` sweep) + buffer reuse ≈ 2x. Filed as a bead.
- Bench `linear_sum_assignment/dense` added (regression coverage; quantifies the gap).

## 2026-06-20 - linear_sum_assignment LAPJV port - WIN ~2x self (5.6-7.4x loss → 3.0-3.7x), byte-matches scipy

- Agent: cc / MistyBirch (closes the loss filed in frankenscipy-zl4m5)
- Decision: **KEEP**. Replaced the basic e-maxx O(n³) Hungarian with a faithful
  port of scipy's `rectangular_lsap.cpp` LAPJV (Crouse 2016): shortest-augmenting-
  path with LAZY dual updates, a shrinking remaining-column list, and SciPy's exact
  tie-break (strictly-cheaper column, else equal-cost UNASSIGNED column).
- Correctness: matches `scipy.optimize.linear_sum_assignment` bit-for-bit on the
  existing scipy-reference tests, AND a new `linear_sum_assignment_cost_matches_
  brute_force` test proves the returned assignment is the true optimum (cost ==
  brute-force min over all permutations) for square + wide continuous matrices.
  Full fsci-opt lib suite 311/0.

| n×n | old (e-maxx) | LAPJV | self | scipy | vs scipy |
| --- | ---: | ---: | ---: | ---: | --- |
| 500  | 40.4 ms | 21.9 ms | **1.84x** | 7.2 ms  | 3.0x slower (was 5.6x) |
| 1000 | 279 ms  | 140 ms  | **1.99x** | 37.6 ms | 3.7x slower (was 7.4x) |

- Closes the documented loss from 5.6-7.4x to 3.0-3.7x. The residual ~3x is the
  safe-Rust (`#![forbid(unsafe_code)]`) bounds-check tax on the hot scatter loop
  (`row_i[j]`, `v[j]`, `shortest[j]` indexed by the scattered remaining-column j)
  + scipy's contiguous-array C micro-optimizations — not algorithmic. Further close
  would need bounds-check elision (iterator restructuring of the scatter loop),
  which is uncertain in safe Rust; documented for follow-up.

## 2026-06-20 - find_peaks_cwt - MEASURED WIN (fsci 3.6x faster); CWT-build parallel = no-op reject

- Agent: cc / MistyBirch
- Finding (MEASURED, rch vs scipy.signal.find_peaks_cwt, n=5000, 29 ricker widths):
  fsci 22.1 ms vs scipy 80.5 ms = **3.6x faster** (even sequential). Added a
  `find_peaks_cwt/n5000_w29` bench (was uncovered).
- REJECT (reverted): parallelizing the CWT-matrix build (the per-width ricker+
  convolve loop) across widths — byte-identical, but a measured no-op (22.1→21.3 ms,
  <5% noise). The CWT build is NOT the bottleneck; the sequential ridge-tracking
  (`identify_ridge_lines`, gap-aware ridge following) dominates the 22 ms and is
  inherently sequential. Don't parallelize the width loop; a real further win would
  need a parallel/restructured ridge tracker. fsci already dominates, so no source
  change ships.

## 2026-06-20 - binned_statistic_2d accumulator fast path - WIN 1.2-1.32x self (4.1x vs scipy), byte-identical

- Agent: cc / MistyBirch
- Decision: **KEEP**. `binned_statistic_2d` materialized every point into a
  `Vec<Vec<Vec<f64>>>` (one Vec per bin, 2500 for 50×50) even for count/sum/mean/
  min/max — which need only running aggregates. Added an accumulator fast path
  (flat count/sum/min/max arrays + a per-bin NaN flag) that skips the
  materialization; median/std keep the materialize path (need all values / two-pass).
- Correctness: **byte-for-bit identical** to the materialize-then-fold path —
  per-bin sum accumulates in point (== push) order, nan-min/max is order-independent
  with the NaN flag — proven by `binned_statistic_2d_fast_path_matches_materialize`
  (count/sum/mean/min/max vs a brute-force reference incl. NaN values + empty bins).
  Existing scipy-reference test still passes; no new failures (the 5 failing stats
  tests — zscore_mad/sklearn helpers — fail identically on origin, unrelated).

| stat (n=200k, 50×50) | materialize | accumulate | self | scipy | vs scipy |
| --- | ---: | ---: | ---: | ---: | --- |
| mean  | 5.42 ms | 4.10 ms | **1.32x** | 16.67 ms | 4.1x faster (was 3.1x) |
| sum   | 4.88 ms | 4.00 ms | 1.22x | — | — |
| count | 4.62 ms | 3.85 ms | 1.20x | — | — |

- Modest at 50 bins (the binning floor/min loop dominates), but the win GROWS with
  bin count (materialize allocates a Vec per bin → O(bins²) allocs + cache misses;
  accumulate is flat). Strictly byte-identical structural cleanup, not a tweak.

## 2026-06-20 - binned_statistic_dd accumulator fast path - WIN 1.25-1.49x self (3.0-3.8x vs scipy), byte-identical

- Agent: cc / MistyBirch
- Decision: **KEEP**. Extend the binned_statistic_2d accumulator fast path to the
  N-D `binned_statistic_dd`: count/sum/mean/min/max use flat aggregate arrays of
  size `bins^ndim` instead of materializing `Vec<Vec<f64>>` (a Vec per bin — the
  dominant cost in high dimensions). median/std keep the materialize path.
- Correctness: **byte-for-bit identical** to the materialize-then-fold path —
  proven by `binned_statistic_dd_fast_path_matches_materialize` (3-D, count/sum/
  mean/min/max vs a brute-force reference incl. NaN values + empty bins). Existing
  dd scipy-reference test passes; isolated change (the 5 failing zscore/mad/sklearn
  tests fail identically on origin, unrelated — verified prior cycle).

| stat (n=200k, 3-D) | materialize | accumulate | self | scipy | vs scipy |
| --- | ---: | ---: | ---: | ---: | --- |
| mean bins=20 (8000 cells)  | 9.35 ms  | 7.49 ms | 1.25x | 28.4 ms | **3.8x faster** |
| mean bins=30 (27000 cells) | 12.46 ms | 8.34 ms | **1.49x** | 27.6 ms | **3.3x faster** |

- Self-speedup GROWS with bin count (1.25x at 8000 cells → 1.49x at 27000) exactly
  as predicted — materialize is O(bins^ndim) Vec allocations, accumulate is flat.
  For higher ndim the gap widens further. Byte-identical structural cleanup. (1-D
  `binned_statistic` left as-is: tiny bin counts + an ambiguous empty-bin NaN policy
  vs its own doc — not worth the risk for a negligible win.)

## 2026-06-20 - binned_statistic (1-D) accumulator fast path - WIN 1.17-1.61x self (6.7-7.5x vs scipy), byte-identical

- Agent: cc / MistyBirch. Completes the binned-statistic accumulate family (1-D + 2-D + N-D dd).
- Decision: **KEEP**. 1-D `binned_statistic` count/sum/mean/min/max now use flat
  aggregate arrays instead of materializing `Vec<Vec<f64>>`; median/std keep the
  materialize path. Preserves this helper's distinct EMPTY-bin policy (NaN for
  EVERY statistic, count/sum included — its `is_empty` check comes first, unlike
  2-D/dd which give 0 for count/sum).
- Correctness: **byte-for-bit identical** to materialize-then-fold — proven by
  `binned_statistic_fast_path_matches_materialize` (137 bins → guaranteed empties,
  NaN values, vs brute-force ref). Existing scipy-reference test passes; isolated.

| stat (n=200k) | materialize | accumulate | self | scipy | vs scipy |
| --- | ---: | ---: | ---: | ---: | --- |
| mean bins=1000 | 2.46 ms | 2.10 ms | 1.17x | 14.1 ms | 6.7x faster |
| mean bins=5000 | 3.53 ms | 2.19 ms | **1.61x** | 16.4 ms | **7.5x faster** |

- Self-speedup grows with bin count (1.17x→1.61x) as the materialize alloc cost
  (O(bins) Vecs) scales. Family now fully on the accumulate path.

### Rejected this cycle (measured, not a practical loss)
- `somersd` continuous/distinct-rank data: fsci is O((R·C)²)=O(n⁴), but **scipy is
  also catastrophic** (2276 ms at n=200, O(n²) crosstab) — both unusable on large
  continuous input; the practical (categorical/small-table) case is fast in both.
  Not a target. `sosfiltfilt` (2.5ms, sequential-IIR wall), RegularGridInterpolator
  (5.5ms) and RectBivariateSpline (7ms) are fast scipy C — low headroom.
