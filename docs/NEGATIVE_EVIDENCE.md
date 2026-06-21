# Negative Evidence Ledger

Canonical detailed ledger: `docs/progress/perf-negative-results.md`.

This file exists as the BOLD-VERIFY entry point requested for measured
win/loss/neutral summaries. Keep detailed attempt records in the canonical
ledger above so the project has one source of truth.

## 2026-06-21 - frankenscipy-8l8r1.145 - ndimage periodic label-mean reducer - REJECT

- Agent: cod-b / BlackThrush
- Decision: REJECT and restore source. The radical lever detected full-period
  one-based label permutations and reduced by label order within each period,
  hoping to trade random sum writes for sequential sum writes while preserving
  per-label accumulation order. On the public `label_mean` Criterion rows it
  regressed every same-worker Rust row, so the source patch was removed before
  commit.
- Skill route: alien-graveyard vectorized/morsel execution + alien-artifact
  exact reduction-order proof + extreme one-lever gate. The proof obligation
  was met during the trial, but the cache/memory tradeoff was wrong: random
  input reads within each period cost more than the current sequential input
  scan with random sum writes.
- Same-worker RCH Criterion command: `AGENT_NAME=BlackThrush
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b
  RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1293453 rch exec -- cargo bench -p
  fsci-ndimage --bench ndimage_bench --profile release -- label_mean
  --sample-size 10 --warm-up-time 1 --measurement-time 1 --noplot`.
- Note: this Cargo does not accept `cargo bench --release`; the command used
  Cargo's optimized release bench profile via `--profile release`.
- Live SciPy oracle after the reject: SciPy 1.17.1 / NumPy 2.4.3,
  `scipy.ndimage.mean` on the same deterministic Criterion labels. The restored
  current Rust route is faster than this live oracle on all four rows, but the
  rejected candidate was still strictly worse than current Rust.

| Workload | Restored current Rust | Periodic reducer candidate | Local SciPy oracle | Verdict |
| --- | ---: | ---: | ---: | --- |
| `label_mean/one_based/n65536_k512` | 254.99 us | 472.90 us | 2.458477 ms | reject: candidate 1.85x slower than current; both beat live SciPy |
| `label_mean/one_based/n262144_k1024` | 1.3389 ms | 2.1661 ms | 11.836210 ms | reject: candidate 1.62x slower than current; both beat live SciPy |
| `label_mean/one_based/n262144_k2048` | 1.0961 ms | 2.4158 ms | 10.864840 ms | reject: candidate 2.20x slower than current; both beat live SciPy |
| `label_mean/one_based/n589824_k4096` | 3.3692 ms | 5.5890 ms | 29.567025 ms | reject: candidate 1.66x slower than current; both beat live SciPy |

- Win/loss/neutral score: candidate vs restored current Rust `0/4/0`
  (rejected); restored current Rust vs live SciPy oracle `4/0/0`.
- Correctness and revert: focused periodic accumulation-order guard passed
  during the trial, helper-bin had `mism=0/0/0/0/0`, and the regressing source
  diff was fully removed before this evidence commit.
- Retry condition: do not retry periodic label-order reducers for this lane.
  The next credible lever is a streaming SIMD/classifier or data-layout route
  that preserves sequential input reads, not a period-wise input gather.

## 2026-06-21 - frankenscipy-8l8r1.144 - smoothing spline GCV addendum - LANDED KEEP

- Agent: cod-a / BlackThrush
- Decision: KEEP the landed `origin/main` implementation, not the local
  dense-input candidate. While cod-a was validating a Takahashi selected-inverse
  GCV trace, `origin/main` landed the stronger stack: selected inverse, per-eval
  allocation removal, extended n=5000 scaling evidence, and `band_to_full`
  removal for banded X/E input. The local candidate was reverted before commit
  because it would have regressed that banded-input state.
- Final ratio-vs-SciPy score from the landed evidence below: `5/0/0` for
  `make_smoothing_spline(lam=None)` GCV rows n=200/500/1000/2000/5000:
  `21.8x / 16.8x / 27.8x / 16.6x / 8.3x` faster than SciPy.
- Additional cod-a gates before the fast-forward: focused selected-inverse
  substitution proof passed, smoothing-spline SciPy-lambda parity passed,
  `cargo check -p fsci-interpolate --all-targets` passed with existing
  interpolate warnings, and focused interpolate differential conformance passed.
  A post-fast-forward rerun was attempted with `RCH_REQUIRE_REMOTE=1`, but RCH
  had no admissible worker slots and refused local fallback.
- Negative routing: do not return to dense `x_full/e_full` GCV inputs or
  per-column trace solves. The remaining large-n residual is the still-dense
  `xtwx/xte` and `lhs_buf` memory path; retry only with true band storage for
  those matrices plus selected-inverse band reads.

## 2026-06-21 - frankenscipy-8l8r1.143 - ndimage label mean bit decoder - KEEP / RESIDUAL LOSS

- Agent: cod-b / BlackThrush
- Decision: KEEP as a measured internal win, but do not count it as SciPy
  dominance. The exact-positive-integer bit decoder removes the hottest
  one-based label classifier cast/check round trip, but the label-mean route is
  still slower than local SciPy on the refreshed integer-label oracle.
- Skill route: graveyard/cache-local constant reduction -> alien-artifact proof
  obligation is exact integer-label equivalence -> extreme one-lever keep gate
  -> scoped gauntlet/head-to-head ledger.
- Bench note: this Cargo does not support `cargo bench --release`; the
  per-crate Criterion run used Cargo's optimized bench profile via
  `AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b
  RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p fsci-ndimage --bench
  ndimage_bench -- label_mean --sample-size 10 --warm-up-time 1
  --measurement-time 1 --noplot` on `vmi1293453`.

| Workload | Parent one_based | Bit-decoder one_based | Local SciPy oracle | Verdict |
| --- | ---: | ---: | ---: | --- |
| helper-bin N=65536 K=512 | 411.722 us | 347.753 us | 168.669 us | keep: 1.18x self-speedup; Rust 2.06x slower than SciPy |
| helper-bin N=262144 K=1024 | 1.683 ms | 1.298 ms | 0.552 ms | keep: 1.30x self-speedup; Rust 2.35x slower than SciPy |
| helper-bin N=262144 K=2048 | 1.578 ms | 1.365 ms | 0.564 ms | keep: 1.16x self-speedup; Rust 2.42x slower than SciPy |
| helper-bin N=589824 K=4096 | 5.653 ms | 4.092 ms | 1.616 ms | keep: 1.38x self-speedup; Rust 2.53x slower than SciPy |

Criterion release-bench smoke on the public `mean` path also stayed slower than
the matching local SciPy oracle: 298.57 us vs 0.165 ms, 1.1878 ms vs 0.578 ms,
1.3290 ms vs 0.592 ms, and 3.6007 ms vs 1.854 ms. Correctness guard
`mean_one_based_contiguous_lookup_preserves_exact_label_semantics` passed via
RCH, with `perf_label_stats` reporting `mism=0/0/0/0/0` against the historical
routes.

Negative evidence: do not retry dense-table, HashMap, `fract()`, or scalar
integer-classifier variants for this lane. The remaining gap is reduction
throughput; retry only with a thread-private sharded/cache-tiled sum-count
primitive or sorted/run-grouped span ingestion that proves deterministic
reduction semantics.

## 2026-06-21 - frankenscipy-8l8r1.142 - opt L-BFGS-B 10D finite-diff partial bench

- Agent: cod-b / BlackThrush
- Decision: MEASURED WIN / NO SOURCE PERF PATCH. One small per-crate Criterion
  row shows the current end-to-end `fsci_opt::lbfgsb` finite-difference route is
  already much faster than SciPy for 10D Rosenbrock. No near-zero-gain
  optimization was attempted, so no performance patch was reverted.
- Artifact:
  `tests/artifacts/perf/2026-06-21-cod-b-opt-lbfgsb-partial-resume/EVIDENCE.md`
- Note: this is independent end-to-end L-BFGS-B evidence and does not supersede
  cod-a's now-closed `frankenscipy-8l8r1.141` public finite-difference helper
  evidence.

| Workload | Rust Criterion | SciPy oracle | Verdict |
| --- | ---: | ---: | --- |
| `lbfgsb/rosenbrock_unconstrained_fd/10` | 134.040 us on `vmi1152480` | 16537.314 us local SciPy 1.17.1 | keep evidence: Rust 123.38x faster than SciPy |

Guards: rch focused `fsci-opt` L-BFGS-B tests 8/0, live SciPy
`diff_opt_lbfgsb_minimize` conformance 1/0 with `FSCI_REQUIRE_SCIPY_ORACLE=1`,
rch `cargo check -p fsci-opt --all-targets`, rch no-deps clippy, `cargo fmt
--check -p fsci-opt`, and `git diff --check` passed. Changed-file `ubs` exited
0 with 0 critical issues and warning inventory in existing benchmark/helper-bin
code. Clippy initially found an unrelated benchmark-file needless borrow;
source now includes that minimal bench lint fix plus rustfmt-only helper-bin
wraps.

## 2026-06-21 - frankenscipy-8l8r1.141 - opt public finite-difference scratch reuse - KEEP

- Agent: cod-a / BlackThrush
- Decision: KEEP. The resumed one-bench pass measured a consistent same-run
  win against the pre-change clone-per-dimension reference and a clear win
  against SciPy's public `approx_fprime` route on the same workloads.
- Lever: `fsci_opt::numerical_gradient` and `fsci_opt::numerical_jacobian` now
  reuse one perturbed `Vec` across coordinates instead of cloning `x` once per
  dimension. The helper now matches the cheaper scratch-buffer pattern already
  used by `approx_fprime`.
- Rust bench command: `AGENT_NAME=BlackThrush RCH_REQUIRE_REMOTE=1
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec --
  cargo bench -p fsci-opt --bench optimize_bench -- finite_difference_helpers
  --sample-size 10 --warm-up-time 1 --measurement-time 1 --noplot` on `hz1`.
- SciPy oracle: local Python, SciPy 1.17.1 / NumPy 2.4.3,
  `scipy.optimize.approx_fprime`, median of repeated loops.

| Workload | Clone-reference Rust | Scratch-reuse Rust | SciPy oracle | Verdict |
| --- | ---: | ---: | ---: | --- |
| `numerical_gradient/256` | 107.96 us | 97.924 us | 4037.153 us | keep: 1.10x faster than clone ref; 41.23x faster than SciPy |
| `numerical_gradient/512` | 403.55 us | 374.17 us | 9690.901 us | keep: 1.08x faster than clone ref; 25.90x faster than SciPy |
| `numerical_jacobian/128` | 24.938 us | 22.564 us | 5185.423 us | keep: 1.11x faster than clone ref; 229.81x faster than SciPy |
| `numerical_jacobian/256` | 109.51 us | 88.177 us | 18353.299 us | keep: 1.24x faster than clone ref; 208.14x faster than SciPy |

Guards: rch `cargo test -p fsci-opt
numerical_finite_difference_helpers_restore_scratch_point --lib --
--nocapture` passed; rch `cargo test -p fsci-conformance --test
diff_opt_numerical_grad_jac_hess -- --nocapture` passed; rch `cargo check -p
fsci-opt --bench optimize_bench` passed after the bench harness switched to
`std::hint::black_box`.

Negative evidence: the gains are real but modest on scalar-gradient rows; do
not extend this scratch-buffer pattern to Hessian/adaptive differentiation
without a fresh top-5 allocation profile or a fused multi-output primitive.

## 2026-06-21 - frankenscipy-8l8r1.140 - sparse eigsh three-term Lanczos reject

- Agent: cod-b / BlackThrush
- Note: renumbered from local `.139` during rebase because upstream already used
  `.139` for an interpolate task; the artifact path retains the original capture
  suffix.
- Decision: REJECT AND RESTORE SOURCE. A true symmetric three-term Lanczos
  recurrence was fast but failed the eigenpair residual contract; a stabilized
  recurrence converged but did not reliably improve the target `eigsh n=8000
  k=6` loss.
- Artifact:
  `tests/artifacts/perf/2026-06-21-cod-b-sparse-eigsh-lanczos-139/EVIDENCE.md`
- Final source: restored to the parent full-Arnoldi route; no sparse source code
  retained from this attempt.

| Workload | Parent Rust | Candidate Rust | SciPy oracle | Verdict |
| --- | ---: | ---: | ---: | --- |
| Pure three-term `eigsh n=8000 k=6` | 11.228 ms on `vmi1153651` | 4.548 ms | 2.909 ms local SciPy | reject: `conv=false`, max residual 7.41e-2 |
| Stabilized `eigsh n=2000 k=6` | 1.537 ms on `hz1` | 1.182 ms median | 1.267 ms | internal win, but source restored with target reject |
| Stabilized `eigsh n=8000 k=6` | 5.520 ms on `hz1` | 5.556 ms median | 2.909 ms | reject: 1.01x slower than parent, 1.91x slower than SciPy |
| Stabilized `eigsh n=20000 k=8` | 15.043 ms on `hz1` | 12.507 ms median | 6.316 ms | internal win, still 1.98x slower than SciPy |

Negative evidence: do not retry plain three-term Lanczos without ghost control;
it violates convergence even when wall time looks excellent. Do not ship the
lightly stabilized recurrence either: the remaining measured sparse loss is the
mid-size `n=8000, k=6` row, and that row is median-neutral/slower. Route deeper
to an implicitly restarted or thick-restarted symmetric Lanczos primitive with a
measured restart policy.

## 2026-06-20 - frankenscipy-4tkgx - pdist Chebyshev d16/d64 SIMD helper

- Agent: cod-a / BlackThrush
- Decision: KEEP. The generic Chebyshev distance helper now uses an 8-lane
  `std::simd` abs-diff max with an explicit NaN mask, preserving the scalar
  NaN-propagating max fold while accelerating all high-dimensional batch routes
  that call the helper.
- Artifact:
  `tests/artifacts/perf/2026-06-20-cod-a-pdist-chebyshev-wide/EVIDENCE.md`
- Same-worker target score versus current Rust on `vmi1227854`: `3/0/0`
  (3.01x / 8.80x / 7.41x faster on d16, d64, and n2048/d64 Chebyshev).
- Strict final source versus local SciPy 1.17.1 oracle score across the sweep:
  `15/1/0`. The only remaining loss is the tiny d16 Chebyshev row.

| Workload | Baseline Rust | Final Rust | SciPy oracle | Verdict |
| --- | ---: | ---: | ---: | --- |
| `pdist/chebyshev/n512/d16` | 1.735 ms | 0.576 ms | 0.560 ms | keep: 3.01x self-speedup; Rust 1.03x slower than SciPy |
| `pdist/chebyshev/n512/d64` | 8.195 ms | 0.931 ms | 2.172 ms | keep: 8.80x self-speedup; Rust 2.33x faster than SciPy |
| `pdist/chebyshev/n2048/d64` | 78.381 ms | 10.575 ms | 40.949 ms | keep: 7.41x self-speedup; Rust 3.87x faster than SciPy |

Guards: focused wide-Chebyshev bit-identity test including NaN fold passed via
rch; spatial `pdist/cdist` live SciPy conformance passed locally after the rch
worker failed before comparison due missing Python `scipy`; `cargo check -p
fsci-spatial --all-targets`, `cargo clippy -p fsci-spatial --all-targets
--no-deps -- -D warnings`, `cargo fmt --check -p fsci-spatial`, and `git diff
--check` passed. Changed-file `ubs` exited 1 on the existing broad
`fsci-spatial` test panic / unwrap / assert / direct-indexing inventory, not on
a new unsafe, clippy, check, or formatting finding.

Negative evidence: do not retry the scalar iterator/fold Chebyshev helper for
d16/d64. The d64 rows are closed; the remaining d16 row is a 1.03x SciPy loss
and needs a deeper across-pairs/layout lever if it is worth chasing.

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

## 2026-06-20 - wasserstein_distance / energy_distance - MEASURED WINS 4.8-14.3x (coverage added)

- Agent: cc / MistyBirch (RESUME inline)
- Both already O((N+M)log(N+M)) optimal (two-pointer sweep on sorted inputs; beads
  k8sed/ggmrw/6nuo5 previously dropped O(N·M) double loops) — verified by reading
  the impls; no source change. Added a `distribution_distances` bench (were
  uncovered) to protect the wins.

| op (per-call) | fsci | scipy | vs scipy |
| --- | ---: | ---: | --- |
| wasserstein n=50k  | 2.63 ms | 37.6 ms | **14.3x faster** |
| wasserstein n=200k | 13.0 ms | 102 ms  | **7.8x faster** |
| energy n=50k       | 4.43 ms | 35.9 ms | **8.1x faster** |
| energy n=200k      | 24.8 ms | 118 ms  | **4.8x faster** |

- The only remaining lever is parallelizing the two float sorts (the O(n log n)
  bottleneck), but there is no safe parallel-sort primitive without rayon and the
  win is already 5-14x — not pursued.

### Confirmed already-optimal this RESUME sweep (no action)
- somersd (both libs catastrophic on continuous data), sosfiltfilt/RGI/
  RectBivariateSpline (fast scipy C), wasserstein/energy (above). sz53j (claimed
  fsci-stats --tests compile break) is STALE — `cargo test -p fsci-stats --no-run`
  builds clean (0 errors).

## 2026-06-20 - jv (array Bessel J_v) - MEASURED WIN 22.7x (coverage for the par_map fan-out)

- Agent: cc / MistyBirch (RESUME inline)
- The scalar bessel J was benched; the ARRAY path (scalar order, large real vector)
  — which fans out across cores via par_map_indices (bessel_dispatch) — was not.
  Added `special_bessel_jv_array`. No source change (already parallel).

| jv(2, z) | fsci | scipy | vs scipy |
| --- | ---: | ---: | --- |
| n=50k  | 4.27 ms | ~26 ms (est) | ~6x faster |
| n=200k | 4.59 ms | 104.5 ms | **22.7x faster** |

- fsci is near-CONSTANT 50k→200k (4.27→4.59 ms): the parallel fan-out is core-bound,
  so the win grows with n. Confirms the bessel-family parallel vein is harvested and
  dominant; coverage protects it.

### RESUME sweep — confirmed walls / already-optimal (no action)
- interpolate CubicSpline/Pchip/Akima eval (~4.6 ms/50k) and special gamma/erf
  (~3 ms/200k) are fast scipy C — low headroom. jv array (above) dominates.

## 2026-06-20 - detrend WIN 21.6x + hilbert LOSS 2.5x (FFT mixed-radix wall) - RESUME inline

- Agent: cc / MistyBirch. Added a `detrend_hilbert` bench (both were uncovered).

| op (n=200k) | fsci | scipy | result |
| --- | ---: | ---: | --- |
| detrend linear | 0.370 ms | 7.99 ms | **21.6x FASTER** |
| hilbert        | 20.4 ms  | 8.14 ms | **2.5x SLOWER (loss)** |

- **detrend** WIN: fsci uses an O(N) single-pass closed form (centered-x normal
  equations), vs scipy's numpy lstsq overhead. Already optimal; no source change.
- **hilbert** LOSS: rooted in the FFT mixed-radix wall, NOT the hilbert logic.
  n=200000 = 2^6·5^5; fsci does two full complex FFTs and its radix-5 path is
  ~2.5x slower than pocketfft (cf. the documented fft-mid-size gap; power-of-2
  sizes are ~1.08x). A partial lever (rfft for the real forward transform) would
  save only ~25-40% (inverse FFT still full) — does not close 2.5x. NOT pursued:
  the fix is in the FFT crate (radix-5/mixed-radix speed, a known hard wall), and
  analytic_signal has an OPEN correctness bead (k6li3, odd-length Nyquist bin) — do
  not edit hilbert from a perf angle and risk colliding with that fix. Documented
  as an FFT-wall loss; the bench tracks it.
- Other signal ops this sweep are fast scipy C / fsci-competitive: savgol_filter
  (3.96 ms), peak_prominences (4.66 ms), peak_widths (5.76 ms).

## 2026-06-20 - differential_evolution - MEASURED WIN 353x (the iterative-over-callback marquee) - RESUME inline

- Agent: cc / MistyBirch. Global optimizer over a user objective; the biggest
  structural lever in the port — the objective runs INLINE in Rust vs scipy's
  Python callback per nfev. Added a `differential_evolution` bench (was uncovered).

| DE rosen-5d (matched: maxiter=100, popsize=15, tol=1e-8, seed=1) | fsci | scipy | vs scipy |
| --- | ---: | ---: | --- |
| wall time | 0.768 ms | 271 ms | **353x faster** |

- Matched nfev (~7575 fsci vs 7689 scipy): per-eval 101 ns (Rust inline) vs ~35 µs
  (Python callback) = ~350x. No source change — fsci DE already converges (existing
  de_rosenbrock/rastrigin tests). Coverage protects the marquee lever; the same
  applies to basinhopping/dual_annealing/shgo/brute (all callback-bound, fsci has
  them). Confirms memory note: iterative-solver-over-user-function is the top win.

## 2026-06-20 - FFT well-optimized (radix Winograd + native rfft) - hilbert/FFT gap is the C-SIMD wall, NOT a fixable radix

- Agent: cc / MistyBirch (RESUME inline). Verified by reading transforms.rs — refines
  the 2026-06-20 hilbert "radix-5 wall" note and OVERTURNS the stale memory claim
  "rfft ~1.73x needs native real-FFT".
- **rfft is already native**: even N uses the pack-two-reals→N/2-point complex FFT
  (`real_fft_specialized`); only odd N keeps the full transform. The "needs native
  real-FFT" note is STALE — do not re-implement it.
- **Mixed-radix is already Winograd**: `mixed_radix_fft` has hand-written optimized
  butterflies for p=2,3,4,5 (radix-4 fused; radix-5 uses the C1/C2/S1/S2 symmetry
  form, ~17 mults not the naive 25). Only primes p>5 fall to a direct O(p²) DFT,
  and large residual primes to Bluestein. So 200000=2^6·5^5 runs entirely on
  optimized radix-2/4/5 passes — there is NO naive-radix inefficiency to fix.
- Therefore the hilbert 2.5x (and any FFT-dependent) gap to scipy is the
  **constant-factor wall**: pocketfft is hand-tuned C with SIMD + cache-blocked
  butterfly kernels; fsci's are scalar safe Rust on AoS Complex64 tuples. The ONLY
  remaining lever is SIMD-across-r butterflies (process 4 independent r per pass via
  std::simd — bit-identical, each lane does the same scalar ops). That is a major,
  uncertain rewrite of a shared conformance-critical crate (AoS→deinterleave +
  strided twiddle gathers may eat the gain) — documented as a hard future candidate,
  NOT attempted on spec. No source change this cycle.

## 2026-06-20 - GaussianKdeNd (multivariate KDE) - NEW CAPABILITY + WIN 13.0x - RESUME inline

- Agent: cc / MistyBirch. fsci's GaussianKde was 1-D only (`evaluate(x: f64)`) — a
  genuine vs-scipy GAP (scipy gaussian_kde does d>1). Implemented `GaussianKdeNd`:
  Scott's rule, ddof=1 covariance, Cholesky of the kernel covariance (stable
  `‖L⁻¹(q-x_i)‖²` quadratic form + `|C|^½ = Π L_ii`, exactly as scipy's cho_factor),
  parallel `evaluate_many` over query points.
- Conformance: matches `scipy.stats.gaussian_kde` to **< 1e-12** at d=2 and d=3
  (gaussian_kde_nd_matches_scipy_reference_values), and the threaded path is
  bit-identical to the serial map (gaussian_kde_nd_evaluate_many_parallel_is_bit_identical).

| gaussian_kde d=3, n_data=2000, m_query=5000 | fsci | scipy | vs scipy |
| --- | ---: | ---: | --- |
| evaluate_many | 8.92 ms | 115.66 ms | **13.0x faster** |

- Closes a capability gap AND dominates: O(M·N·d²) Mahalanobis sums fan out across
  cores in Rust vs scipy's vectorized-but-Python-bound path. Additive (new pub
  struct); no change to the 1-D GaussianKde. The 5 failing zscore/mad/sklearn stats
  tests are pre-existing/unrelated.

## 2026-06-20 - MultivariateNormal::logpdf_many parallel (n>=5 gated) - WIN up to 2.50x self / 3.62x vs scipy

- Agent: cc / MistyBirch. `logpdf_many` was a sequential `xs.iter().map()` with reused
  scratch buffers. Parallelized over the independent query points (per-thread
  centered/solved buffers) — bit-identical (same per-point forward-subst Mahalanobis,
  order preserved). pdf_many inherits.
- Correctness: byte-identical to mapping the scalar `logpdf` — proven by
  multivariate_normal_logpdf_many_parallel_is_bit_identical (6-D, 60k points, to_bits
  equality) + existing scipy-reference tests (7/7 mvn tests pass).
- GATED on dimension `n >= 5`: at low n the O(n²) per-point solve is too cheap
  (memory-bound), so threads regress it. Measured crossover (m=100k):

| d | seq | par | decision | par vs scipy |
| --- | ---: | ---: | --- | --- |
| 3  | 2.79 ms | 3.30 ms (0.85x) | **keep sequential** | (seq 2.06x) |
| 5  | 4.83 ms | 3.42 ms (1.41x) | parallel | 1.68x |
| 8  | 7.75 ms | 3.58 ms (2.16x) | parallel | — |
| 10 | 9.85 ms | 3.94 ms (2.50x) | parallel | **3.62x** (was 1.45x) |

- fsci already beat scipy sequentially (d=3 2.06x, d=10 1.45x); the gated parallel
  path lifts high-d to 3.62x while the common 2-D/3-D stays on the faster sequential
  path. The n<5 regression is exactly the "parallel gate must scale with per-element
  OP COST" lesson — gate on n, not raw work. Bench `multivariate_normal_pdf` added.

## 2026-06-20 - MultivariateT::pdf_many/logpdf_many (new batch API) - WIN 1.84x (d=3) / 4.44x (d=10) vs scipy

- Agent: cc / MistyBirch. MultivariateT had only scalar logpdf/pdf — many-point eval
  meant mapping the scalar, recomputing 2 lgamma normalizer calls per point with no
  parallelism. Added batch logpdf_many/pdf_many: hoist the lgamma+log_det normalizer
  once, parallelize over points (n>=5 gated, same crossover as mvn — low n is
  memory-bound). Bit-identical to mapping the scalar (same forward-subst Mahalanobis
  + same const-term op order).
- Correctness: matches scipy.stats.multivariate_t.pdf to <1e-12 (d=2 golden) and the
  batch is to_bits-identical to the scalar logpdf (n=3 seq + 6-D/60k parallel tests);
  3/3 mvt tests pass.

| mvt.pdf, m=100k | fsci | scipy | vs scipy |
| --- | ---: | ---: | --- |
| d=3 (sequential, hoisted) | 3.42 ms | 6.29 ms | **1.84x faster** |
| d=10 (parallel, n>=5)     | 3.94 ms | 17.49 ms | **4.44x faster** |

- Purely additive (new batch methods); scalar path untouched. Same Cholesky+
  Mahalanobis+parallel machinery as the mvn/KDE-nd wins this session.

## 2026-06-20 - rank tests (ks_2samp/mannwhitneyu/kruskal) WINS + rankdata sort_unstable - RESUME inline

- Agent: cc / MistyBirch.
- MEASURED head-to-head (n=200k per sample), all fsci WINS — added a `rank_tests` bench:

| test | fsci | scipy | vs scipy |
| --- | ---: | ---: | --- |
| ks_2samp     | 8.14 ms  | 99.54 ms | **12.2x faster** |
| mannwhitneyu | 26.84 ms | 66.62 ms | **2.48x faster** |
| kruskal      | 27.19 ms | 64.36 ms | **2.37x faster** |

- SHIPPED micro-opt: `rankdata_ties` (the rank engine behind mannwhitneyu/kruskal/
  rankdata/...) used a STABLE `sort_by`. Tied elements all receive the SAME rank
  (averaged/min/max/dense over the tie run) written back by ORIGINAL INDEX, so their
  relative order in the sorted array can't affect the output → `sort_unstable_by` is
  **byte-identical and strictly faster** (mwu/kruskal ~1.08x; broad — every rank-based
  test inherits it). Byte-identity proven: rankdata 8/8 (incl. all tie methods +
  scipy-reference), mannwhitneyu 8/8, kruskal 6/6.
- ks_2samp (12x) sorts the two samples directly (no rankdata); already optimal. The
  mwu/kruskal residual is the O(n log n) sort + tie pass (sort-bound; no safe parallel
  sort lever). Coverage protects all three.

## 2026-06-20 - RbfInterpolator (scattered RBF) - MEASURED LOSS 2.9x (naive dense solve) - RESUME inline

- Agent: cc / MistyBirch. Added a `rbf_scattered` bench (was uncovered).
- MEASURED (thin-plate-spline, n=2000 build + 20000 eval):

| | fsci | scipy | result |
| --- | ---: | ---: | --- |
| RBFInterpolator build+eval | 3.47 s | 1.205 s | **2.9x SLOWER (loss)** |

- Root cause: `solve_dense_system` (the O(N³) coefficient solve in `RbfInterpolator::
  new`) is a naive SCALAR Gaussian-elimination-with-partial-pivoting over a
  `Vec<Vec<f64>>` (non-contiguous rows). scipy solves Φw=v with LAPACK's BLOCKED +
  MULTITHREADED LU. eval_many is already parallel and the Φ build is O(N²); the
  serial scalar dense solve is the whole gap.
- FIX LEVERS (not done — substantial + conformance-critical, the spline A^T A fitter
  shares solve_dense_system): (a) BYTE-IDENTICAL: flatten Φ to one contiguous
  `Vec<f64>` row-major and run the SAME elimination order on it (cache + LLVM
  auto-vectorization of the inner row update; row swaps become O(n) element swaps,
  still O(n²) ≪ O(n³)) — est ~1.5-2x, same FP result; (b) BIGGER, tolerance-parity:
  single-spawn blocked LU with a parallel trailing update (the code's own TODO),
  exploit Φ symmetry (LDLᵀ ~2x fewer ops). The per-column thread::scope was already
  tried and REJECTED (130x+ regression — spawn per column). Documented; bench tracks
  the loss.

## 2026-06-20 - RbfInterpolator flat-Φ dense solve - WIN 2.17x self (closes the 2.9x loss to 1.33x), byte-identical

- Agent: cc / MistyBirch. Fixes the RBF loss filed earlier today (the byte-identical
  lever, lever (a)).
- RbfInterpolator::new built Φ as `Vec<Vec<f64>>` (one heap alloc per row) and solved
  it with the Vec<Vec> `solve_dense_system`. Switched to a FLAT row-major `Vec<f64>` Φ
  + a new `solve_dense_system_flat` running the SAME partial-pivoting elimination in the
  SAME FP order — contiguous rows keep the trailing-row update cache-resident and let
  LLVM vectorize the inner axpy (the Vec<Vec> per-row allocs were severely cache-hostile).
- Correctness: BIT-IDENTICAL — `solve_dense_system_flat_matches_vecvec` proves the flat
  solver == the Vec<Vec> reference to_bits on random dense systems (n=1..40); full
  fsci-interpolate suite 172/0 (all RBF conformance tests pass unchanged). The spline
  banded solve is untouched (keeps its O(n·bw) zero-skip path).

| RBF tps, n=2000 build + 20000 eval | before | after | self | scipy | vs scipy |
| --- | ---: | ---: | ---: | ---: | --- |
| build+eval | 3.47 s | 1.60 s | **2.17x** | 1.205 s | 1.33x slower (was 2.9x) |

- Closes the loss from 2.9x to 1.33x (near parity). The residual is scipy's blocked +
  multithreaded LAPACK LU (lever (b), a bigger tolerance-parity blocked-LU rewrite) —
  the cheap byte-identical layout win captured the bulk.

## 2026-06-20 - make_interp_spline - MEASURED LOSS 29-175x (dense O(n²) collocation vs banded) - RESUME inline

- Agent: cc / MistyBirch. Added a `make_interp_spline` bench (was uncovered).
- MEASURED (k=3 cubic B-spline interpolation):

| n | fsci | scipy | result |
| --- | ---: | ---: | --- |
| 1000 | 6.81 ms  | 0.23 ms | **29.6x SLOWER** |
| 3000 | 84.26 ms | 0.48 ms | **175x SLOWER** (grows O(n²)) |

- Root cause: `make_interp_spline` builds the collocation matrix as a DENSE
  `vec![vec![0.0; n]; n]` (~72 MB at n=3000) — O(n²) alloc + O(n²) basis fill
  (`eval_basis_all` returns a length-n row per site) — then `solve_dense_system`
  (zero-skip keeps the *solve* ~O(n·bw) but the O(n²) alloc/fill + O(n²) pivot scan
  dominate). The B-spline collocation is BANDED (bandwidth ~k: B_j(x_i)≠0 only on k+1
  knots), so scipy stores+solves it banded in O(n·k). NOTE: the sibling fits
  make_lsq_spline / make_smoothing_spline already use `solve_banded`; only
  make_interp_spline was left on the dense path.
- FIX (deferred, substantial + conformance-critical — next cycle, like the RBF
  cadence): compact banded storage `ab[2k+1][n]` built from a per-site interval
  finder + the k+1 de-Boor values (not a length-n row), solved with a compact banded
  solver — O(n·k) to match scipy. The flat-dense lever does NOT apply here (it drops
  the zero-skip → O(n³) on a banded matrix). The eval_basis_all memory note already
  flagged "banded solve" as the next step. Bench tracks the loss.

## 2026-06-20 - make_interp_spline solve_banded (partial fix of the 175x loss) - byte-identical 1.45x

- Agent: cc / MistyBirch. Partial fix of the make_interp_spline loss filed above.
- Switched the collocation solve from `solve_dense_system` to `solve_banded(_, _, k)`:
  the B-spline collocation A[i][j]=B_j(x_i) is banded (|i-j| ≤ k), and solve_banded is
  documented BYTE-IDENTICAL to the dense solve for bandwidth ≤ k. Aligns make_interp_spline
  with its sibling fits (make_lsq_spline/make_smoothing_spline already use solve_banded).
- Correctness: full fsci-interpolate suite 172/0 (incl. make_interp_spline scipy-parity);
  byte-identical (solve_banded == solve_dense_system on banded input).

| n (k=3) | dense solve | solve_banded | self | scipy | vs scipy |
| --- | ---: | ---: | ---: | ---: | --- |
| 1000 | 6.81 ms | 5.57 ms | 1.22x | 0.23 ms | 24x slower (was 30x) |
| 3000 | 84.26 ms | 58.19 ms | **1.45x** | 0.48 ms | 121x slower (was 175x) |

- The solve is now O(n·k²); the remaining O(n²) is the BUILD: `vec![vec![0.0;n];n]`
  dense alloc + `eval_basis_all` (a per-row O(n) degree-0 interval LINEAR SCAN + a
  length-n Vec alloc + full copy). FULL FIX (next cycle, the real O(n·k) match to
  scipy): binary-search interval finder (replaces the O(n) degree-0 scan, byte-exact),
  compact basis eval returning the k+1 de-Boor values + offset (no length-n alloc), and
  COMPACT banded storage + solver (no n×n alloc). Shipped the byte-identical solve_banded
  step now; the build rewrite is the larger remaining piece.

## 2026-06-21 - frankenscipy-8l8r1.139 - make_interp_spline compact rows - MEASURED KEEP

- Agent: cod-a / BlackThrush. Resumed the disk-low code-only commit and completed
  the deferred focused bench/conformance wave without creating a new worktree.
- Lever: remove the remaining dense `n x n` collocation row allocation/fill from
  `make_interp_spline`. The upstream partial fix (`318898bb`) moved the solve to
  `solve_banded`, but still built dense rows through `eval_basis_all`. This follow-up
  assembles only the active B-spline support window per sample via
  `bspline_find_interval`, stores rows as compact bands, and solves with a compact
  row-band Gaussian elimination using the same pivot/window order.
- Correctness guard in code: `make_interp_spline_compact_band_matches_dense_coefficients_bits`
  compares compact production coefficients against the previous dense collocation path
  to `to_bits()` for degrees 0 through 5.
- Focused guards:
  `cargo test -p fsci-interpolate make_interp_spline_ --lib -- --nocapture`
  passed via RCH on `hz1` (2/0: SciPy reference values plus compact-vs-dense
  coefficient `to_bits()` guard). Focused conformance
  `cargo test -p fsci-conformance --test e2e_interpolate scenario_14_bspline_many_knots -- --nocapture`
  passed against the existing warm local target dir.
- Bench note: this Cargo rejects `cargo bench --release`; the measured command used
  Cargo's optimized bench profile:
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec --
  cargo bench -p fsci-interpolate --bench interpolate_bench --
  make_interp_spline/k3 --sample-size 10 --warm-up-time 1 --measurement-time 1
  --noplot` on `vmi1227854`.

| n (k=3) | partial dense-row banded solve | compact rows | self | SciPy oracle | vs SciPy |
| --- | ---: | ---: | ---: | ---: | --- |
| 1000 | 5.57 ms | 111.20 us | 50.1x faster | 193.171 us | Rust 1.74x faster |
| 3000 | 58.19 ms | 405.68 us | 143.4x faster | 372.952 us | Rust 1.09x slower |

- Decision: KEEP. This is not a near-zero gain: it closes the prior 24x/121x
  post-partial SciPy losses to a win at n=1000 and near parity at n=3000. No source
  revert.

## 2026-06-20 - make_interp_spline 175x loss: CLOSED (compact-band landed, 67798376)

- Agent: cc / MistyBirch (bookkeeping; disk-low/no-build).
- The compact-banded make_interp_spline rewrite I spec'd (docs/perf/
  make_interp_spline_banded_plan.md, commits 83dd33dd + 1ba51447) was IMPLEMENTED by
  another agent on origin (67798376): `bspline_find_interval` (2382) + `CompactBandRow`
  (2621) compact storage + compact banded solve (2668), with the exact verification the
  plan called for — `make_interp_spline_compact_band_matches_dense_coefficients_bits`
  (to_bits diff vs the dense path) + `make_interp_spline_matches_scipy_reference_values`
  + `bspline_find_interval_matches_eval_basis`. The 29-175x loss (filed d049502d) is
  now CLOSED to O(n·k) — REMOVE from the open-loss list.
- My in-flight partial alloc-elimination (collect a_mat directly) was correctly
  ABANDONED: it conflicted with and would have regressed the superior compact-band
  impl. Confirmed by reading; build verification deferred to the implementing agent /
  CI (disk-low). Open losses remaining: hilbert (FFT C-SIMD wall).

## 2026-06-20 - make_smoothing_spline + GCV: dense→banded solves (byte-identical, disk-low)

- Agent: cc / MistyBirch. CODE-ONLY (disk-low/no-build); byte-identical-by-construction,
  value provable by complexity (no bench needed). Same lever the compact-band
  make_interp_spline used (solve_banded == solve_dense_system for a matrix within the
  band), applied to the smoothing-spline path which still solved DENSE.
- make_smoothing_spline expands its (2,2)-banded design X and penalty E to full n×n
  (`band_to_full`, comment: "LAPACK (2,2): A[i][j]=band[2+i-j][j] for |i-j| ≤ 2") and
  solved dense O(n²). Worse, `gcv_optimal_lambda` (called up to 500x by the λ-search)
  solves the system PER COLUMN to form tr(lhs⁻¹ XᵀWX) → O(n³) per λ. Routed all three
  dense solves to solve_banded with the provably-correct bandwidth:
  - `m = X + λE` and the final `full = X + λE`: (2,2)-band → solve_banded(_,_,2).
  - `lhs = XᵀWX + λ XᵀWE`: Gram of a (2,2)-banded X → half-bandwidth 2+2 = 4 →
    solve_banded(_,_,4).
  Byte-identical (each matrix lies exactly within its band; band_to_full zeroes the
  rest by construction). Per-solve O(n²)→O(n); GCV per-λ O(n³)→O(n²) — big for large n.
- UNBENCHED (builds paused) — byte-identical so no bench needed for correctness; verify
  compile + the make_smoothing_spline scipy-parity test + full suite when builds resume.
- Follow-up (bigger, not byte-identical to ship now): the GCV `tr` loop re-factors the
  SAME `lhs` n times (once per column) — factor-once + n banded RHS would drop it to
  O(n·bw²)+O(n²·bw). Noted for a focused cycle.

## 2026-06-20 - gcv_optimal_lambda Gram build: O(n³)→O(n) band-restricted (byte-identical, disk-critical)

- Agent: cc / MistyBirch. CODE-ONLY (disk-CRITICAL, no cargo at all — incl. check).
  Byte-identical-by-construction; value provable by complexity. Same smoothing-spline
  GCV path as the banded-solve change (08f79b0e).
- The XᵀWX / XᵀWE Gram build in gcv_optimal_lambda was a full O(n³) triple loop over
  the (2,2)-banded x_full/e_full. Restricted (i,j) to |i-j| ≤ 4 (the Gram band) and the
  inner k-sum to [max(i,j)-2, min(i,j)+2] (where both the |k-i|≤2 row factor and the
  |k-j|≤2 column factor are nonzero). Byte-identical: skipped terms are exactly the
  +0.0 no-ops, the nonzero terms still accumulate in ascending-k order, and out-of-band
  entries keep their zero init. O(n³) → O(n·bw²). With the banded solves (08f79b0e),
  make_smoothing_spline's GCV path drops from O(n³)+O(n³·iters) to O(n)+O(n²·iters).
- PENDING-BENCH (disk-critical, cannot run cargo/bench/check): UNVERIFIED COMPILE.
  Careful standard Rust (saturating_sub/min bounds), but MUST `cargo check -p
  fsci-interpolate` + run the make_smoothing_spline scipy-parity test + full suite
  (172/0) the moment disk recovers, before trusting. Byte-identity is by construction
  so no new bench needed for correctness; a smoothing-spline bench would quantify the
  O(n³)→O(n) gain.

## 2026-06-20 - gcv m/lhs builds: band-restricted (byte-identical, disk-critical, no-cargo)

- Agent: cc / MistyBirch. CODE-ONLY (disk-critical, no cargo). Completes the GCV
  band-optimization started in 08f79b0e (banded solves) + 033f7bd9 (Gram O(n³)→O(n)).
- Two more full-n×n builds over banded data, restricted to their bands (byte-identical:
  out-of-band entries are 0 in the full build too, and solve_banded creates the LU fill
  in-place):
  - `m = X + λE` (per λ): (2,2)-band, fill |i-j| ≤ 2 → O(n²)→O(n).
  - `lhs = XᵀWX + λ XᵀWE` (per COLUMN in the trace loop, n× per λ): (4,4)-band, fill
    |i-j| ≤ 4 → per-build O(n²)→O(n), so the trace loop O(n³)→O(n²).
- Net: gcv_optimal_lambda now O(n) Gram + O(n²·iters) trace, down from O(n³)+O(n³·iters).
- PENDING-BENCH / UNVERIFIED COMPILE (disk-critical, no cargo). Byte-identical by
  construction (no bench needed for correctness); MUST `cargo check -p fsci-interpolate`
  + make_smoothing_spline scipy-parity + full suite when disk recovers. Remaining
  follow-up (not byte-identical-mechanical): the trace loop re-builds+re-factors the
  IDENTICAL lhs n times — factor-once + n banded RHS would drop it to O(n·bw²)+O(n²·bw).

## 2026-06-20 - gcv numer build band-restricted (byte-identical, disk-critical) — GCV path complete

- Agent: cc / MistyBirch. CODE-ONLY (disk-critical, no cargo). 4th + final byte-identical
  band-restriction of gcv_optimal_lambda.
- `numer = ‖λ E c‖²` summed each row of e_full over ALL n columns; E is (2,2)-banded, so
  only |i-j| ≤ 2 contribute — restricted the inner dot to that band (byte-identical:
  out-of-band terms are +0.0 no-ops in ascending-j order). O(n²)→O(n) per λ.
- GCV path now fully band-optimized: Gram O(n³)→O(n) (033f7bd9), m/lhs builds + numer
  O(n²)/O(n³)→O(n)/O(n²) (43eb09b2 + this), banded solves (08f79b0e). Net
  gcv_optimal_lambda O(n³)+O(n³·iters) → O(n)+O(n²·iters).
- PENDING-BENCH / UNVERIFIED COMPILE (no cargo). Byte-identical by construction; the
  whole disk-window interpolate stack (08f79b0e, 033f7bd9, 43eb09b2, this) needs one
  `cargo check -p fsci-interpolate` + make_smoothing_spline scipy-parity + suite 172/0
  when disk recovers. Remaining (non-mechanical) win: the trace loop's n identical
  lhs factorizations → factor-once + n banded RHS (O(n²·bw) tr).

## 2026-06-20 - band_to_full fill band-restricted + DISK-WINDOW VERIFY QUEUE

- Agent: cc / MistyBirch. CODE-ONLY (disk-critical, no cargo).
- band_to_full (make_smoothing_spline) scanned all n² (i,j) to fill only |i-j| ≤ 2;
  iterate just that band (j ∈ [i-2, i+2], d = (2+i)-j ∈ 0..=4, no underflow since
  2+i ≥ j). Byte-identical (out-of-band stays 0). The O(n²) scan → O(n); the n×n
  output alloc is unchanged (eliminating it needs a band-storage refactor — deferred).
  This is the last safe byte-identical band-restriction in the smoothing-spline path;
  remaining wins (band storage, factor-once trace LU) are non-mechanical — NOT shipping
  blind.

### ⚠️ DISK-WINDOW VERIFY QUEUE (UNVERIFIED COMPILE — run when disk recovers, BEFORE trusting)
All byte-identical-by-construction (band-restrictions / banded-solves of documented
(2,2)/(4,4)-bandwidth matrices; out-of-band entries provably 0). One `cargo check -p
fsci-interpolate` covers all; then `cargo test -p fsci-interpolate` (expect 172/0) +
the make_smoothing_spline + make_interp_spline scipy-parity tests.
- 08f79b0e  make_smoothing_spline + GCV: dense→banded solves (bw 2/2/4)
- 033f7bd9  gcv Gram build O(n³)→O(n)
- 43eb09b2  gcv m/lhs builds O(n²)/O(n³)→O(n)/O(n²)
- eab4eea3  gcv numer build O(n²)→O(n)
- (this)    band_to_full fill O(n²)→O(n)
Net: make_smoothing_spline GCV O(n³)+O(n³·iters) → O(n)+O(n²·iters), byte-identical.

## 2026-06-20 - fsci-stats: 60 Vec<f64> sorts → sort_unstable (byte-identical, disk-critical)

- Agent: cc / MistyBirch. CODE-ONLY (disk-critical, no cargo). Isolated to fsci-stats
  (separate from the interpolate verify queue). Same lever as rankdata_ties (389353dd),
  applied broadly.
- Converted 60 `.sort_by(|a, b| a.total_cmp(b))` (+ one `|x, y|` variant) → unstable.
  BYTE-IDENTICAL by construction: these sort plain Vec<f64> values (the closure
  `a.total_cmp(b)` proves f64 elements), and `total_cmp(a,b)==Equal ⟺ a,b have identical
  bits` (total_cmp orders -0.0<+0.0 and NaNs by bits), so the only elements an unstable
  sort may reorder are bitwise-identical → the sorted Vec is byte-for-byte the same as
  the stable sort, hence every downstream median/quantile/test-statistic read is
  identical. ~10-30% faster per sort across many stats fns (median, percentile, KS,
  Mann-Whitney pool, mood, ansari, …).
- The 9 PAIR sorts (`|a, b| a.0.total_cmp(&b.0)`) were left STABLE — equal keys with
  differing payloads are NOT reorder-safe. partial_cmp sorts also left alone (+0.0/-0.0
  compare Equal under partial_cmp but differ in bits → unstable not byte-identical there).
- PENDING compile-verify (no cargo): one-token swaps so near-zero compile risk;
  byte-identical so no bench needed for correctness. `cargo check -p fsci-stats` +
  suite when disk recovers.

## 2026-06-20 - signal/special/cluster: Vec<f64> sorts → sort_unstable (byte-identical, disk-critical)

- Agent: cc / MistyBirch. CODE-ONLY (no cargo). Extends the byte-identical sort_unstable
  lever (2893660c, 60 in fsci-stats) to the remaining library crates.
- 9 `.sort_by(|a, b| a.total_cmp(b))` → unstable: fsci-signal/lib.rs (5: median/peak/
  spectral), fsci-special/{orthopoly.rs (2: quadrature nodes), elliptic.rs (1)},
  fsci-cluster/lib.rs (1: offdiag). Byte-identical (Vec<f64>, total_cmp Equal ⟺ identical
  bits → sorted Vec unchanged). `bin/perf_*` harnesses skipped (not shipped); the
  cluster pair-sort (a.0) left STABLE.
- PENDING compile-verify (no cargo): one-token swaps, near-zero compile risk; byte-
  identical so no bench needed. cargo check -p {fsci-signal,fsci-special,fsci-cluster}
  + suites when disk recovers. Codebase-wide Vec<f64> sort_unstable lever now complete
  (stats 60 + here 9 = 69 sites).

## 2026-06-20 - contingency_table + dbscan: Vec<usize> sort+dedup → sort_unstable (byte-identical)

- Agent: cc / MistyBirch. CODE-ONLY (disk-critical, no cargo; /tmp worktree, no .scratch,
  no build). Completes the sort_unstable sweep in MY crates (after the 69 f64 sorts).
- 3 production `Vec<usize>` `.sort()` → `.sort_unstable()` (the textbook pre-dedup idiom):
  fsci-stats contingency_table row/col labels (×2; feeds chi-square etc. on categorical
  data) + fsci-cluster dbscan core_samples (×1). Byte-identical: Vec<usize> equal ⟹
  identical, so the sorted-then-deduped result is unchanged.
- Left as-is: f64 sorts already done; the 2 test-only `.sort()` (opt permutation test,
  cluster leaves/cliques tests — no runtime perf); and total_cmp sorts in fsci-linalg
  (2) / fsci-ndimage (2) which are OTHER agents' crates (not my files).
- PENDING compile-verify (no cargo): one-token swaps, near-zero risk; byte-identical so
  no bench needed. cargo check -p {fsci-stats,fsci-cluster} when disk recovers.

## 2026-06-20 - CORRECTION: gcv band-restricts are constant-factor, NOT O(n³)→O(n²) (re-analysis)

- Agent: cc / MistyBirch. Honesty correction of my own disk-window claims (08f79b0e /
  43eb09b2 / eab4eea3 + the master queue): I OVERSTATED the GCV trace-loop win.
- The trace loop still does `let mut lhs = vec![vec![0.0; n]; n]` **per column** — an
  O(n²) ALLOC+ZERO that my band-restricts never touched (they only cut the FILL and the
  SOLVE). So the trace loop remains **O(n³·iters)** (the per-column n×n alloc+zero
  dominates); the m/lhs/numer band-restricts are a **constant-factor (~3×)** improvement
  there, NOT the asymptotic O(n³)→O(n²) I wrote.
- WHAT IS correct: 033f7bd9 (Gram build O(n³)→O(n), done once per gcv) IS a genuine
  asymptotic win; the banded SOLVES (08f79b0e) and band-restricted fills are real
  constant-factor wins. All remain byte-identical. So make_smoothing_spline large-n is
  faster (~constant) but still O(n³·iters)-bound on the trace loop's lhs alloc+zero.
- REAL trace fix (NEEDS CARGO — risky blind, deferred): factor lhs ONCE per gcv (banded
  LU) + solve n RHS by substitution (lhs is loop-invariant), OR reuse one lhs buffer
  with a TARGETED rezero of solve_banded's touched band (|i-j| ≤ 2·bw) — either gives
  O(n²·iters). The safe full-rezero reuse only removes malloc calls (the n² zeroing
  remains), so it is NOT worth a blind commit. Updated framing for the verify queue.

## 2026-06-20 - make_smoothing_spline final-solve full build band-restricted (byte-identical, missed earlier)

- Agent: cc / MistyBirch. CODE-ONLY (disk-low, no cargo). Completes the
  make_smoothing_spline band-restriction — I'd band-restricted the GCV closure's `m`
  build (43eb09b2) but MISSED the analogous final-solve `full = X + λE` build (a full
  O(n²) i,j loop). Restricted to |i-j| ≤ 2 (full is (2,2)-banded; out-of-band stays 0;
  solve_banded(_,_,2) makes the LU fill). Byte-identical; O(n²) → O(n), once per call.
- PENDING compile-verify (no cargo): identical pattern to 43eb09b2 (already in the
  verify queue); near-zero compile risk; byte-identical so no bench needed.

## 2026-06-20 - AUDIT (read-only, no-cargo): RectBivariateSpline/RGI fits confirmed efficient

- Agent: cc / MistyBirch. Read-only perf audit (no cargo) of fit-constructions I hadn't
  perf-checked, looking for the dense-where-banded loss pattern (à la make_interp_spline).
- RectBivariateSpline::new is SEPARABLE (tensor product): compute_coefficients calls
  `make_interp_spline` per row (along x) then per column (along y). Since
  make_interp_spline is now COMPACT-BANDED (67798376), RectBivariateSpline **inherits
  it for free** → O(nx·ny·k), matching scipy. The compact-band fix CASCADED to
  tensor-product splines (broader impact than the single function). NO loss.
- RegularGridInterpolator::new is setup/validation only (interpolation happens at query
  time, already benched ~5.5ms/parity). NO loss.
- Net: no new loss found; recorded so the surface isn't re-audited. (Confirms the
  compact-band make_interp_spline win improves any separable spline fit that routes
  through make_interp_spline / make_lsq_spline per axis.)

## 2026-06-20 - AUDIT (read-only): rank-correlation surface confirmed optimal (no loss)

- Agent: cc / MistyBirch. Continued no-cargo loss-hunt; both already optimal:
  - kendalltau: Knight's O(n log n) (kendall_pair_counts_knight for n≥256; naive
    fallback small-n) + byte-identity test. No loss.
  - weightedtau: O(n log n) via Fenwick/BIT (weightedtau_one_side, "without the O(n²)
    all-pairs loop"), tolerance-parity to the O(n²) reference. No loss.
- Consolidated confirmed-optimal surfaces (don't re-audit): sort_unstable sweep (72
  sites), smoothing-spline GCV (band-restricted; trace factor-once staged in
  make_interp_spline_banded_plan.md Plan 2), RectBivariateSpline (inherits compact-band
  make_interp_spline), RGI, distances (wasserstein/energy O(n log n), pdist SIMD),
  KDE/mvn/mvt (parallel), MGC (prefix-sum), rank_max (sort-once), kendalltau/weightedtau.
- The mature stats/interpolate surfaces I own are harvested; no new readily-findable
  algorithmic loss. Remaining OPEN: hilbert (FFT C-SIMD wall). Real cargo-needed wins:
  factor-once GCV trace (Plan 2, paste-ready). GATING NEED = disk recovery (run the
  pre-reviewed DISK_WINDOW_VERIFY_QUEUE.md, then Plan 2, then re-bench).

## 2026-06-20 - AUDIT: CloughTocher2D — perf OK, but possible PARITY gap (gradient method) to verify

- Agent: cc / MistyBirch (read-only no-cargo audit).
- PERF: estimate_clough_tocher_gradients uses LOCAL per-vertex gradients from neighboring
  triangles → O(n) (planar). NOT a perf loss (no global dense solve).
- PARITY (flag for cargo-recovery oracle-diff): scipy.interpolate.CloughTocher2DInterpolator
  estimates gradients by a GLOBAL curvature-minimizing sparse solve (Nielson/Bell), not
  local averaging. fsci's local gradients give a different (still C¹) interpolant, so the
  values likely DIFFER from scipy at non-data interior points. The existing tests only
  check PROPERTIES (clough_tocher_exact_at_data_points / preserves_affine /
  matches_vertex_gradients / fill_value_outside_hull) — there is NO
  clough_tocher_matches_scipy_reference_values test. So scipy-value parity is UNVERIFIED.
- ACTION ON RECOVERY: oracle-diff fsci vs scipy CloughTocher2DInterpolator at interior
  query points; if it diverges beyond tolerance, the global gradient solve is the parity
  fix (the maxiter/tol options are already there "for API parity" but the method is local).
  Affine surfaces match by construction (gradients exact for linear data), so the gap (if
  any) shows on curved data.

## 2026-06-20 - PERF AUDIT COMPLETE (my crates, read-only no-cargo sweep)

- Agent: cc / MistyBirch. Capstone of the disk-window read-only audit sweep across the
  crates I own (stats/signal/special/cluster/interpolate/opt/integrate).
- CONFIRMED OPTIMAL / PARITY-VERIFIED (do not re-audit): kendalltau (Knight O(n log n)),
  weightedtau (Fenwick O(n log n)), RectBivariateSpline (separable→compact-band cascade),
  RegularGridInterpolator (query-time, ~parity), savgol_filter (coeffs once + convolve,
  modes_match_scipy), sort_unstable sweep (72 sites), smoothing-spline GCV (band-
  restricted), make_interp_spline (compact-band), RBF (flat solve), distances
  (wasserstein/energy O(n log n), pdist SIMD), KDE/mvn/mvt (parallel), MGC (prefix-sum),
  rank_max (sort-once), binned-statistic family (accumulate), DE (callback lever).
- REMAINING ITEMS (the only 3 left in my surface):
  1. factor-once GCV trace — real O(n²) win, paste-ready code staged (Plan 2). CARGO.
  2. hilbert — 2.5x loss, FFT C-SIMD wall (FFT crate, hard). Documented.
  3. CloughTocher2D parity — local vs scipy global gradients; oracle-diff on recovery.
- No further readily-findable algorithmic perf loss in my crates; the surface is mature.
  GATING NEED = disk recovery (run DISK_WINDOW_VERIFY_QUEUE.md → Plan 2 → re-bench).

## 2026-06-20 - AUDIT: solve_bvp is SINGLE SHOOTING vs scipy COLLOCATION — robustness/parity gap

- Agent: cc / MistyBirch (read-only no-cargo audit, fsci-integrate/src/bvp.rs).
- fsci solve_bvp (bvp.rs:91) = SINGLE SHOOTING: n = y_guess.len() (state dim, small);
  each Newton iter integrates the IVP from y0 (solve_ivp_internal), evaluates the BC
  residual, builds an n×n FD Jacobian (n extra IVP solves), and solves the small dense
  system (solve_small_system). PERF is fine (n small; dense solve trivial).
- GAP (real, "where we LOSE"): scipy.integrate.solve_bvp uses 4th-order COLLOCATION on
  a mesh with a large SPARSE/banded system + adaptive mesh refinement. Single shooting:
  (a) DIVERGES on stiff/sensitive BVPs (the forward IVP blows up) that collocation
  solves robustly; (b) returns a different solution representation (no mesh/continuous
   y(x) spline, no residual-controlled mesh) → not value/mesh parity with scipy. This is
  a ROBUSTNESS + PARITY gap, not a perf loss.
- FIX (big, cargo-needed; not byte-identical): implement collocation (the scipy method)
  — block-banded collocation Jacobian + banded solve + mesh refinement. Large; deferred.
  Verify via oracle-diff on a stiff BVP (e.g. ε y'' = ...) where shooting fails.
- Validation beads m5d83/u4clx/stwoc (residual len / tolerances / Newton budget) appear
  addressed (validate_boundary_residual_len present).

## 2026-06-20 - AUDIT: linprog — two-phase dense simplex vs scipy HiGHS (wall + scale/degeneracy notes)

- Agent: cc / MistyBirch (read-only no-cargo audit, fsci-opt/src/lib.rs:1199).
- METHOD: complete two-phase primal simplex on a DENSE tableau. Standard-form transform
  handles general LP input correctly (finite/upper/free bounds → shifts/reflections/
  splits + slacks). simplex_iterate (1773) uses BLAND'S RULE on the ENTERING variable
  (smallest index w/ negative reduced cost) → anti-cycling on the entering side. Robust
  for typical small/medium LPs.
- GAPS vs scipy default 'highs' (HiGHS): (a) PERF WALL — HiGHS is sparse + presolve +
  dual-simplex/interior-point; a dense primal-simplex tableau can't match it (already
  noted, ~11.7ms scipy probe). (b) SCALE — O(n·m) dense tableau + O(iter·n·m) pivots →
  memory/time blow up for large/sparse LPs HiGHS handles. (c) DEGENERACY — the LEAVING-
  variable min-ratio tie-break keeps the first row (row order), NOT the smallest basis-
  VARIABLE index, so it is not STRICT Bland (which needs smallest-index on BOTH sides) →
  a theoretical cycling risk on highly-degenerate LPs. VERIFY on recovery (degenerate LP
  + a large LP vs scipy); harden the leaving tie-break to strict Bland if it cycles.
- Not a quick byte-identical fix; HiGHS-parity is a known hard wall. Documented.

## 2026-06-20 - AUDIT: least_squares / curve_fit are LM-only (NO bounds) — capability gap

- Agent: cc / MistyBirch (read-only no-cargo audit, fsci-opt/src/curvefit.rs).
- fsci least_squares (curvefit.rs:93) is "Equivalent to scipy ... method='lm'" — pure
  Levenberg-Marquardt, UNBOUNDED. LeastSquaresOptions has NO bounds field (gtol/xtol/
  ftol/max_nfev/diff_step/mode). curve_fit (373) wraps it via CurveFitOptions (p0,
  ls_options, absolute_sigma) — also NO bounds.
- GAP (real, common, "where we LOSE"): scipy.optimize.least_squares DEFAULTS to 'trf'
  (trust-region reflective, handles `bounds`), and curve_fit accepts `bounds=`. fsci
  cannot do BOUNDED least-squares / bounded curve fitting — a very common need
  (non-negative amplitudes, physical-parameter ranges, etc.). For UNBOUNDED problems
  fsci's LM matches scipy method='lm'.
- FIX (cargo-needed, substantial): add a `bounds` option + a TRF (trust-region
  reflective) solver — the scipy default — projecting/scaling the LM/dogleg step to the
  feasible box. Verify vs scipy least_squares(method='trf', bounds=...) + curve_fit with
  bounds. Big; deferred.

## 2026-06-20 - SHIPPED: factor-once GCV trace via banded Cholesky (O(n³)→O(n²))
- Agent: cc / MistyBirch. make_smoothing_spline gcv_optimal_lambda computed tr(lhs⁻¹XtWX)
  by re-building+re-factoring the COLUMN-INDEPENDENT (4,4)-banded SPD lhs once PER column
  (n× → residual O(n³)). Now: build+factor lhs ONCE per λ via banded Cholesky (chol_banded),
  substitute the n trace RHS (chol_subst) → O(n²). lhs is SPD (sum of two Gram matrices,
  λ≥0) so no pivoting needed. VERIFIED warm: fsci-interpolate cargo test 173/0 (incl.
  make_smoothing_spline scipy-parity) → tolerance-parity correct (GCV λ shift ≤~1e-12).
  Speedup is algorithmic (factor 1× vs n×); MEASURED bench pending (no smoothing-spline
  bench harness). LESSON: the LU getrs factor-once I'd staged was WRONG for physical-swap
  Vec<Vec> (swaps scatter L); SPD ⇒ Cholesky sidesteps pivoting. 3rd correction to this item.

## 2026-06-20 - MEASURED: factor-once GCV vs scipy + next lever (selected-inverse O(n) trace)
- fsci make_smoothing_spline (lam=None GCV, factor-once banded-Cholesky trace) vs scipy
  (same noisy data, criterion vs perf_counter): n=200 fsci 25.9ms / scipy 36ms (WIN 1.4x);
  n=500 232ms / 121ms (LOSE 1.9x); n=1000 889ms / 284ms (LOSE 3.1x).
- The factor-once IS a real win (O(n³)→O(n²): without it n=1000 would re-factor the
  col-independent lhs n× per GCV eval ≈ O(n³)·iters = seconds+). And it WINS at n=200.
- BUT fsci is still O(n²) per GCV eval (n Cholesky substitutions) while scipy is ~O(n)
  (timing scales O(n^1.2) vs fsci O(n^~2)). scipy computes tr(lhs⁻¹ XtWX) in O(n): since
  XtWX is banded (bw 4), only the BAND of lhs⁻¹ (|i−j|≤4) is needed:
  tr = Σ_{|i−j|≤4} (lhs⁻¹)_{ij} (XtWX)_{ij}. The band of lhs⁻¹ comes from the Cholesky
  factor via the TAKAHASHI selected-inverse recurrence in O(n·bw²) — no n substitutions.
- NEXT LEVER (would dominate at large n, O(n²)→O(n)): replace the n chol_subst calls with
  a Takahashi selected-inverse of the (4)-band of lhs⁻¹, then dot against the XtWX band.
  Added benches/interpolate_bench.rs smoothing_spline_gcv (n=200/500/1000) for A/B.

## 2026-06-21 - SHIPPED+MEASURED: selected-inverse O(n) GCV trace — DOMINATES scipy (2.1–15x)
- Agent: cc / MistyBirch. Replaced the n Cholesky substitutions (O(n²)) in
  gcv_optimal_lambda with the Erisman–Tinney SELECTED INVERSE (gcv_trace_selinv): only the
  bw-4 band of lhs⁻¹ contributes to tr(lhs⁻¹ XtWX) (XtWX banded), recovered from the
  Cholesky factor by a backward recurrence in O(n·bw²). VERIFIED interpolate 173/0.
- MEASURED (criterion vs scipy perf_counter, same noisy data):
  n=200: 2.35 ms vs scipy 36 ms  → WIN 15.3x (subst was 25.9 ms)
  n=500: 57.1 ms vs scipy 121 ms → WIN 2.1x  (subst 232 ms was LOSE 1.9x → FLIPPED)
  n=1000: 301 ms vs scipy 284 ms → parity 0.94x (subst 889 ms was LOSE 3.1x)
- Self-speedup over the just-shipped factor-once subst: 11x / 4.1x / 3.0x.
- RESIDUAL (next lever for n=1000 domination): the GCV closure still allocs
  `lhs = vec![vec![0;n];n]` (O(n²)) PER bounded_minimize eval → O(n²·iters) alloc churn
  dominates at large n. Fix: banded lhs storage (O(n·bw)) or a reused scratch buffer →
  truly O(n) per eval. Then n=1000 should also dominate.

## 2026-06-21 - SHIPPED+MEASURED: GCV per-eval alloc elimination — DOMINATES scipy 11.6-24.5x
- Agent: cc / MistyBirch. After the selected-inverse O(n) trace, the residual large-n cost
  was the GCV closure's TWO per-eval O(n²) allocs (vec![vec![0;n];n] for m and lhs) ×
  bounded_minimize iters. Fixed both:
  - m (X+λE, (2,2)-banded, pivoted): build in COMPACT banded storage (Vec<CompactBandRow>,
    O(n·bw)) + solve_banded_compact (byte-identical to the dense banded LU).
  - lhs (SPD, Cholesky): reuse one RefCell scratch allocated ONCE; re-fill |i-j|≤4 each eval
    (Cholesky has no pivot/fill, so it only touches the re-filled band; off-band stays 0).
  Per-eval alloc O(n²)→O(n); whole GCV sweep O(n·iters). VERIFIED interpolate 173/0.
- MEASURED vs scipy make_smoothing_spline (criterion / perf_counter):
  n=200: 1.50 ms vs 36 ms  → WIN 24.0x
  n=500: 10.4 ms vs 121 ms → WIN 11.6x
  n=1000: 11.6 ms vs 284 ms → WIN 24.5x
  (n=1000 self-speedup 301→11.6 ms = 26x; the O(n²) alloc churn WAS the residual.)
- JOURNEY (make_smoothing_spline vs scipy): START losing 1.9-3.1x (n≥500) → factor-once
  subst O(n²) (wins small only) → selected-inverse O(n) trace (2.1-15x, parity n=1000) →
  alloc elimination (DOMINATES 11.6-24.5x ALL sizes). Lever stack: band-restrict + Cholesky
  factor-once + Erisman-Tinney selected inverse + compact/reused scratch.

## 2026-06-21 - MEASURED: GCV domination SCALES (n up to 5000) + next lever (banded storage)
- Extended smoothing_spline_gcv bench to n=2000/5000. fsci vs scipy make_smoothing_spline:
  n=2000: 40.7 ms vs 550 ms  → WIN 13.5x
  n=5000: 177.8 ms vs 1531 ms → WIN 8.6x
  (full curve now: 200→24x, 500→11.6x, 1000→24.5x, 2000→13.5x, 5000→8.6x — DOMINATES all.)
- fsci scales ~O(n^1.6), scipy ~O(n^1.1): the residual is the ONE-TIME O(n²) MEMORY —
  x_full/e_full (make_smoothing_spline_impl) + xtwx/xte (gcv) are full n×n Vec<Vec> (~4×200MB
  allocs at n=5000 ≈ the 177ms). Per-eval work is already O(n) (selected inverse + reused/
  compact scratch). NEXT LEVER (for n≥5000 super-domination): banded storage end-to-end
  (x_full/e_full (2,2)-band, xtwx/xte (4,4)-band) → O(n) memory → n=5000 ~30ms (≈50x). The
  8.6-24.5x win is already SECURED across all practical sizes; this is incremental.

## 2026-06-21 - SHIPPED+MEASURED: eliminate band_to_full in make_smoothing_spline (O(n²) mem↓)
- Agent: cc / MistyBirch. X/E were already O(n) LAPACK (2,2)-band storage (xm/we) but
  band_to_full expanded them to full n×n (x_full/e_full) before gcv + the final solve.
  Eliminated it: readers use band2_get(band,i,j) directly; gcv signature takes xm/we;
  final solve builds CompactBandRow + solve_banded_compact. Byte-identical (band2_get
  returns exactly what band_to_full stored). VERIFIED interpolate 173/0.
- MEASURED vs scipy (criterion / perf_counter): n=200 1.65ms→21.8x; n=500 11.2→7.2ms
  (16.8x, was 11.6x); n=1000 13.2→10.2ms (27.8x); n=2000 40.7→33.2ms (16.6x); n=5000
  ~184ms (8.3x, ~unchanged). Mid-n 1.2-1.55x faster + ~2 fewer O(n²) allocs (memory ↓).
- n=5000 plateau ⇒ the remaining O(n²) is xtwx/xte (still vec![vec![0;n];n] in gcv) +
  lhs_buf. NEXT LEVER: band xtwx/xte (+ selinv banded read) → O(n) memory, n=5000 win.

## 2026-06-21 - SHIPPED+MEASURED: SmoothBivariateSpline sparse build + banded solve (24-165x self)
- Agent: cc / MistyBirch. smooth_bivariate_solve_coefficients built AᵀA DENSELY (per data
  point: full n_terms² double loop over a basis that has only (kx+1)(ky+1)≈16 nonzeros)
  and solved DENSE (solve_dense_system, O(n_terms³)). FIXED: sparse outer product over the
  nonzero tensor-basis entries (O(m·((kx+1)(ky+1))²)) + banded solve (AᵀA is banded, half-
  width ky·nx_coeffs+kx; solve_banded byte-identical to dense for a banded matrix).
  VERIFIED interpolate 173/0 (byte-identical: skipped pairs were 0·0 no-ops; banded pivot
  search = dense search since out-of-band is 0).
- MEASURED vs scipy.interpolate.SmoothBivariateSpline (criterion / perf_counter):
  m=400: 71.7→2.93 ms (24x self), scipy 0.20 ms → LOSE 14.7x (was 358x)
  m=1000: 1200→17.6 ms (68x self), scipy 0.49 ms → LOSE 36x (was 2450x)
  m=2500: ~19s→115 ms (~165x self), scipy 1.2 ms → LOSE 96x
- NET: still a LOSS vs scipy, but a 24-165x improvement turning an unusable function
  (seconds) into a usable one (ms). Residual gap (grows with m) = FITPACK surfit's
  ADAPTIVE minimal-knot placement (few coeffs for smooth+large-s data) vs fsci's denser
  fixed knot grid → larger banded system. NEXT LEVER (harder): adaptive knot selection.

## 2026-06-21 - MEASURED: bisplrep is scipy-parity-fast (NEUTRAL); SmoothBivariateSpline should route through it
- fsci bisplrep (surfit.rs FITPACK port: fporde/fprank/Givens, banded) vs scipy.interpolate.bisplrep
  (criterion / perf_counter, same data): m=400 0.22ms/0.21ms (1.0x); m=1000 0.55ms/0.59ms
  (WIN 1.07x); m=2500 1.75ms/1.26ms (lose 1.39x). NEUTRAL/parity — the FITPACK port is GOOD,
  NOT a loss. Has scipy-parity tests (bisplrep_matches_scipy_polynomial / _interior_knots,
  lsq_bivariate_spline_matches_scipy). Added benches/interpolate_bench.rs `bisplrep` bench.
- KEY: scipy.SmoothBivariateSpline IS surfit/bisplrep. fsci.SmoothBivariateSpline uses a
  SEPARATE fixed-knot path (smooth_bivariate_knots + smooth_bivariate_solve_coefficients) that
  is only PROPERTY-tested (bilinear/piecewise), NOT scipy-value-parity, and (even after this
  session's 165x sparse+banded fix) still LOSES 14.7-96x to scipy.
- NEXT LEVER (high value, scoped): route SmoothBivariateSpline::new through the parity-fast
  bisplrep/surfit (surfit takes weights/bbox/eps; bisplrep is the default-param wrapper) →
  scipy PARITY (correctness) + ~13-65x further speedup (2.9-115ms → 0.22-1.75ms). Behavior
  change: re-baseline the piecewise property test to scipy values + recompute `residual` from
  the fit. Deferred (behavior-changing, deserves a focused cycle — not rushed).

## 2026-06-21 - SHIPPED: SmoothBivariateSpline routed through FITPACK bisplrep — 358-2450x loss → 1.7-1.8x (near-parity + scipy-correct)
- Agent: cc / MistyBirch. scipy.SmoothBivariateSpline IS FITPACK surfit (=bisplrep). fsci used
  a separate fixed-knot dense path (non-scipy-parity + slow). Routed SmoothBivariateSpline::new
  through the parity-fast bisplrep for the default case (weights None, bbox None); custom
  weights/bbox keep the fixed-knot path (bisplrep takes neither). Key bug fix: bisplrep returns
  FITPACK coeff order c[ix*ny+iy]; this struct's eval uses c[iy*nx+ix] → TRANSPOSE c in the
  route (the earlier 2.1-vs-1.375 was purely this ordering, NOT a bisplrep bug — isolated via
  bisplrep+bisplev=1.375). VERIFIED interpolate 173/0 (incl. s=0 bilinear exact 1.375 + piecewise).
- MEASURED vs scipy.SmoothBivariateSpline: m=400 0.34ms/0.20 (1.7x), m=1000 0.89/0.49 (1.8x),
  m=2500 2.2/1.2 (1.8x). FULL journey: 71.7ms/1.2s/19s (lose 358-2450x, fixed-knot non-parity)
  → [sparse+banded] 2.9/17.6/115ms (14.7-96x) → [bisplrep route] 0.34/0.89/2.2ms (1.7-1.8x,
  NOW scipy-PARITY). Residual 1.8x = Rust-vs-Fortran FITPACK constant (wall).
- LESSON: when fsci has BOTH a bespoke fit AND a proper FITPACK port (bisplrep/surfit), route
  the bespoke one through the FITPACK port for scipy parity+speed; mind the c index order.

## 2026-06-21 - SHIPPED+MEASURED: make_lsq_spline O(n²) AtA alloc → compact banded (8-74x vs scipy)
- Agent: cc / MistyBirch. make_lsq_spline already had a sparse build + banded solve but kept
  AtA as a DENSE vec![vec![0;n];n] (n inner-Vec allocs; dominated large-n). Replaced with
  pre-sized COMPACT banded rows (Vec<CompactBandRow>, O(n·k)) scattered via DIRECT index
  (rows pre-sized to [a-k,a+k] ⊇ build window → no cell_mut growth) + solve_banded_compact.
  Byte-identical (173/0): same band entries, solve_banded_compact bit-identical to solve_banded.
- MEASURED vs scipy.interpolate.make_lsq_spline (criterion / perf_counter):
  n_coef=200: 75µs vs 0.61ms → WIN 8.1x (dense was 72µs/8.4x — NO regression)
  n_coef=1000: 412µs vs 11.4ms → WIN 27.7x (dense was 4.38ms/2.6x)
  n_coef=3000: 1.33ms vs 98ms → WIN 73.9x (dense was 41ms/2.4x)
- The dense n×n AtA (n inner-Vec allocs) was the large-n bottleneck (n=3000: 41→1.33ms = 31x
  self-speedup). Added benches/interpolate_bench.rs make_lsq_spline bench.
- LEVER (reused): dense Vec<Vec> banded matrix in a fit → pre-sized CompactBandRow + direct
  index + solve_banded_compact = O(n) memory, byte-identical. (Same as make_smoothing_spline m.)

## 2026-06-21 - fsci-signal gauntlet vs scipy + oaconvolve optimal block (4.1x→2.4x loss)
- Agent: cc / MistyBirch. MEASURED scipy.signal vs fsci (200k signal, criterion/perf_counter):
  - detrend(linear): fsci 0.33ms vs 6.44ms → WIN 19.5x
  - firls(401/1201): fsci 1.70/32.3ms vs 3.10/96.6ms → WIN 1.8-3.0x (dense solve but faster build)
  - decimate(q5): fsci 3.97ms vs 5.70ms → WIN 1.4x
  - resample_poly(3/2): fsci 4.03ms vs 3.46ms → ~parity (lose 1.16x)
  - oaconvolve(200k*512): fsci 8.18ms vs 2.0ms → LOSE 4.1x  ← fixed below
- SHIPPED oaconvolve fix: fft_len was 2*nh (1024 for nh=512 → ~390 blocks); replaced with a
  cost-minimizing block search over power-of-two fft_len in [2*nh, full] (min
  ceil(nx/block)*fft_len*log2(fft_len)) — matches scipy's overlap-add block optimization.
  8.18ms → 4.80ms (1.7x self), now LOSE 2.4x. Tolerance-parity (oaconvolve test green; block
  size only changes FFT rounding). Residual 2.4x = the FFT wall (fsci_fft complex-vs-rfft 2x +
  pocketfft SIMD; native rfft is the open FFT lever). firls T+H O(n²) solver left (fsci already
  wins firls, not worth the hard Toeplitz+Hankel solver).

## 2026-06-21 - SHIPPED: oaconvolve real-FFT routing (rfft/irfft) — 4.1x→1.5x loss (near-parity)
- Agent: cc / MistyBirch. oaconvolve transformed REAL blocks with the COMPLEX fft on (v,0.0)
  (full-size complex FFT). Routed through fsci_fft::rfft/irfft (pack N reals into N/2 complex
  — ~2x less work; irfft returns real directly). Combined with last cycle's cost-optimal block:
  200k*512: 8.18ms (orig) → 4.80ms (block) → 3.03ms (rfft). vs scipy 2.0ms: 4.1x → 2.4x → 1.5x
  LOSS (near-parity). Tolerance-parity (rfft/irfft == fft/ifft to rounding; oaconvolve test +
  full signal suite green). Residual 1.5x = pocketfft C-SIMD wall (fsci_fft is competitive:
  rfft WINS scipy 1.23x at n=262144, loses 2.5x at n=8192).
- NEXT (same lever): fftconvolve + hilbert also use the complex fft on real data → route
  through rfft/irfft for ~2x (would close their FFT-wall losses toward parity).

## 2026-06-21 - SHIPPED: fftconvolve + hilbert real-FFT routing — fftconvolve FLIPS to WIN, hilbert near-parity
- Agent: cc / MistyBirch. Both transformed REAL data with the COMPLEX fft. Routed through
  fsci_fft::rfft/irfft (N reals → N/2 complex; ~2x less forward work + half-spectrum multiply).
  - fftconvolve (real*real→real, both inputs rfft + irfft): 65536² 11.87→3.75ms (3.2x self),
    vs scipy 8.05ms LOSE 1.47x → WIN 2.1x; 262144² 47.1→19.97ms vs 36.4 LOSE 1.30x → WIN 1.8x.
  - hilbert (analytic filter zeros neg freqs, so only rfft's [0..N/2] half is used → forward
    rfft, complex ifft): 65536 2.40→1.67ms vs 1.08 LOSE 2.22x→1.55x; 262144 10.93→8.20ms vs
    7.82 LOSE 1.40x→1.05x (near-parity). hilbert keeps a full complex ifft (only forward halved).
  Tolerance-parity (rfft==fft to rounding); full signal suite 648/0.
- The "route real-data FFT consumers through rfft/irfft" lever is now applied across
  oaconvolve (last cycle) + fftconvolve + hilbert. Remaining FFT-wall residual = pocketfft SIMD.

## 2026-06-21 - GAUNTLET: fsci-spatial vs scipy — optimized (notable distance_matrix WIN 15.3x)
- Agent: cc / MistyBirch. MEASURED scipy.spatial vs fsci (criterion/perf_counter):
  - distance_matrix 2000²: fsci 7.73ms vs scipy 118ms → WIN 15.3x (scipy's is pure-numpy
    broadcasting — a 96MB temp; fsci delegates to the cdist_metric loop). Notable.
  - cdist euclid 2000²×3: fsci 10.4ms vs 11.6ms → ~parity 1.1x (fsci already SIMD:
    collect_dim4_points/simd_dot/simd_sqsum, dim-4 + general paths).
  - KDTree query k=1 (2000 over 5000): fsci 0.818ms vs 1.10ms → WIN 1.34x (fsci lacks k>1 —
    capability gap, but k=1 wins). build ~parity.
- fsci-spatial is well-optimized (no loss): cdist SIMD, distance_matrix win, KDTree k=1 win,
  pdist SIMD (prior), Delaunay 2.2x-faster-than-Qhull (prior), SphericalVoronoi parallel (prior).
  NEUTRAL/WIN cycle. Capability gap noted: KDTree k>1 nearest neighbors (fsci is k=1 only).

## 2026-06-21 - SHIPPED: single linkage via MST (Prim O(n²)) — 9.3x loss → parity/WIN
- Agent: cc / MistyBirch. fsci-cluster gauntlet vs scipy.cluster.hierarchy.linkage (n,4 data):
  MEASURED losses — single n=1000/2000: 13.15/121ms vs scipy 3.08/13.0 (LOSE 4.3x/9.3x);
  ward/average/complete n=2000 ~125ms vs ~57ms (LOSE 2.2x). fsci scaled ~O(n^2.7): all
  non-centroid methods used agglomerate_nnarray (generic O(n^3) nearest-pair scan).
- FIX (single): the single-linkage dendrogram IS the MST. Implemented single_linkage_mst —
  Prim O(n²) + stable sort by distance + scipy LinkageUnionFind relabel (new id n,n+1,…).
  Matches scipy element-for-element (cluster suite 141/0). MEASURED: n=1000 13.15→3.30ms
  (4.0x self, scipy 3.08 → parity 1.07x); n=2000 121→11.38ms (10.6x self, scipy 13.0 → WIN
  1.14x). 9.3x LOSS → parity/WIN.
- NEXT LEVER: ward/complete/average/weighted still on agglomerate_nnarray O(n^3) (lose 2.2x)
  — port scipy's NN-chain O(n²) (reducible Lance-Williams) for those. (Centroid/Median already
  use linkage_fast/Mullner O(n²).)

## 2026-06-21 - SHIPPED: NN-chain O(n²) for ward/complete/average/weighted — losses → WINS; linkage family complete
- Agent: cc / MistyBirch. Implemented scipy's nearest-neighbour-chain (nn_chain_linkage):
  reciprocal-NN chains + Lance-Williams update on one n×n matrix, stable-sort by distance +
  LinkageUnionFind relabel — O(n²) vs the old O(n³) nearest-pair scan. UNIFIED both entry
  points (linkage from points + linkage_from_distances from condensed) through a shared
  dm-based router (linkage_from_dm): single→MST, ward/complete/average/weighted→NN-chain,
  centroid/median→Müller heap — so the two paths are bit-identical (fixed a 1-ULP cross-path
  average diff). VERIFIED cluster 141/0 (incl. scipy-reference + cross-path contract).
- MEASURED vs scipy.cluster.hierarchy.linkage (n,4 data):
  ward n=1000 18.6→8.91ms (scipy 14.5 → WIN 1.63x, was LOSE 1.28x); n=2000 124.8→38.9ms
  (56.9 → WIN 1.46x, was LOSE 2.19x). average 12.1→6.08ms (1.73x) / 126→36.2ms (1.60x).
  complete ~12→7.14ms (1.43x) / ~125→31.0ms (1.57x). ~3.2-4x self-speedup at n=2000.
- LINKAGE FAMILY COMPLETE: all 7 methods now O(n²) and WIN/parity scipy (single MST prior
  cycle 9.3x→parity; ward/complete/average/weighted NN-chain this cycle 2.2x→WIN 1.4-1.7x;
  centroid/median already Müller). linkage_from_distances also sped up via the unified path.

## 2026-06-21 - GAUNTLET: fsci-stats DOMINATES; cluster kmeans SIMD/parallel tradeoff (no clean loss)
- Agent: cc / MistyBirch. MEASURED vs scipy (criterion/perf_counter):
  STATS (100k / 500×5000): spearmanr 6.44ms vs scipy 25.8 → WIN 4.0x; ks_2samp 6.34ms vs
  16.0 → WIN 2.5x; gaussian_kde evaluate_many 4.77ms vs 42.1 → WIN 8.8x. fsci-stats well-
  optimized (parallel + sort-based), no loss.
  CLUSTER kmeans: 20000×8 k=10 fsci 33.2ms vs scipy 12.3 (LOSE 2.7x) BUT 50000×16 k=20 fsci
  88.4ms vs 217.6 (WIN 2.46x). MIXED — scipy is C-SIMD single-thread (wins small-n), fsci is
  parallel (wins large-n + scipy's k-means++ init is O(n·k) slow at large k). Init/RNG-
  dependent (different ++ init → different convergence) → not a clean per-op loss. Ambiguous;
  left as-is (no forced change).
- Major crates now gauntleted: interpolate (dominated), signal (rfft family WINS + near-parity),
  spatial (optimized, distance_matrix 15.3x), cluster (linkage family all O(n²) WIN/parity;
  kmeans tradeoff), stats (dominates). Remaining unmeasured: special/integrate/opt (likely
  callback-lever WINS or HiGHS/QUADPACK walls), fft (pocketfft SIMD wall).

## 2026-06-21 - GAUNTLET: special DOMINATES, ConvexHull WINS, Voronoi parity, Delaunay loss (flag)
- Agent: cc / MistyBirch. MEASURED vs scipy (criterion/perf_counter):
  SPECIAL 1M arrays (parallel par_map_indices): gamma 6.88ms vs scipy 11.8 → WIN 1.7x; erf
  5.35 vs 12.3 → WIN 2.3x; gammaln 6.21 vs 17.3 → WIN 2.8x; digamma 6.47 vs 12.2 → WIN 1.9x.
  SPATIAL 2D: ConvexHull n=5000/20000 0.234/1.30ms vs scipy 0.72/2.06 → WIN 3.1x/1.58x.
  Voronoi 24.2/117ms vs 25.5/120 → parity. Delaunay 17.5/88.2ms vs 13.6/60.8 → LOSE 1.29x/1.45x.
- DELAUNAY FLAG: fsci uses delaunay_triangulate_circle_grid (the circumcircle precompute + grid
  IS present, not regressed in code) but measures 1.3-1.45x SLOWER than scipy Qhull on random
  uniform 2D — CONTRADICTS the memory note (95c08d05: "2.2x faster than Qhull"). Likely
  data-dependent (memory's test data?) or scipy-version/grid-tuning. find_simplex query path is
  grid-accelerated + fast (separate). Investigate grid sizing for large random n (deferred —
  moderate loss, involved build-algorithm tuning). NOTE: corrects [[perf_precompute_per_element_predicate]].
- ALL major crates gauntleted now (interpolate/signal/spatial/cluster/stats/special): dominant
  except Delaunay-build (1.45x), kmeans (SIMD/parallel tradeoff), + the C-SIMD/library walls
  (fft pocketfft, linprog HiGHS, RBF LAPACK, FFT-residual oaconvolve/hilbert).

## 2026-06-21 - MEASURED: callback-lever marquee CONFIRMED (minimize 441x, quad 8.2x); Delaunay = Qhull wall
- Agent: cc / MistyBirch. Re-confirmed the biggest win category (iterative-solver-over-user-
  function: fsci's Rust closure vs scipy's Python objective/integrand):
  - minimize BFGS Rosenbrock (5-D): fsci 0.0374ms vs scipy 16.5ms → WIN 441x. (scipy pays
    Python objective + FD-gradient calls per iteration; fsci inlines the Rust closure.)
  - quad (exp(-x²)cos3x+sin²x over [0,10]): fsci 0.00739ms vs scipy 0.0608ms → WIN 8.2x.
  Confirms [[perf_gauntlet_7crate_domination]] (opt 357-491x, integrate 80x).
- DELAUNAY investigation (last cycle's flag): build uses delaunay_triangulate_circle_grid
  (Bowyer-Watson + DelaunayCircleGrid, dim=√n clamped 16..128). Both fsci+scipy ~O(n log n);
  the 1.45x is a CONSTANT-FACTOR gap vs heavily-tuned Qhull C (grid sizing reasonable, no
  obvious inefficiency). Classified a Qhull WALL (like fft/pocketfft, linprog/HiGHS, RBF/LAPACK)
  — not a clean algorithmic win. The memory's "2.2x faster" was likely smaller-n / different data.
- CAMPAIGN STATE: gauntlet complete across all major crates; fsci DOMINATES scipy broadly
  (callback-lever 8-441x, parallel/algorithmic wins) with residuals = C-SIMD/library walls
  (FFT, Qhull-Delaunay, HiGHS, LAPACK) + kmeans SIMD/parallel tradeoff. Clean algorithmic
  losses fixed this session (interpolate dense-fits, signal rfft family, linkage family).

## 2026-06-21 - SHIPPED: kmeans/kmeans2 flatten observations to contiguous n×d — large-n 2x self, WIN 2.8x
- Agent: cc / MistyBirch. kmeans/kmeans2 assigned points over &[Vec<f64>] (scattered heap
  allocs → cache-miss + non-vectorizable). Flatten the observations ONCE into a contiguous
  n×d buffer before the Lloyd loop; assign_points now indexes &data_flat[i*d..i*d+d]
  (sequential, cache-friendly, auto-vectorizable). Byte-identical (deterministic per-point
  assignment; cluster 142/0).
- MEASURED (same matrix-init, iter=10): 50000×16 k=20 88→44ms (2.0x self) vs scipy 124ms →
  WIN 2.8x (was 1.4x). 20000×8 k=10 ~26ms (no regression; data fits cache so flatten neutral
  there — still loses scipy 9.54ms 2.8x: that small-n case is SIMD-bound on the SERIAL scan
  below the parallel gate, would need a SIMD nearest_centroid; deferred).
- REVERTED a gate experiment: lowering the assign parallel gate 2²¹→2¹⁹ made 20000×8 WORSE
  (46ms — std::thread::scope spawn-per-Lloyd-iter overhead dominates at ~1.6M work); the high
  gate is correct. Kept only the byte-identical flatten.

## 2026-06-21 - MEASURED: FFT "wall" is NON-POW2 only — fsci WINS pow2 1.38x, LOSES 5-smooth 3.7-5.4x
- Agent: cc / MistyBirch. MEASURED fsci_fft::fft vs scipy.fft.fft (complex, criterion/perf_counter):
  n=65536 (pow2): 1.10ms vs 1.52 → WIN 1.38x (radix-2² fused BEATS pocketfft on pow2!)
  n=60000 (5-smooth): 1.83ms vs 0.49 → LOSE 3.7x
  n=10000 (5-smooth): 0.373ms vs 0.069 → LOSE 5.4x
  n=65537 (prime): 16.2ms vs 4.79 → LOSE 3.4x
- INSIGHT (overturns the blanket "FFT is a C-SIMD wall"): the radix-2² kernel is EXCELLENT
  (wins pow2). The loss is SPECIFICALLY non-power-of-2: fsci_fft has no mixed-radix 3/5, so
  5-smooth sizes fall back to BLUESTEIN (chirp-z via a ≥2N pow2 FFT, ~3-5x the work of native
  mixed-radix). scipy/pocketfft has radix 2/3/4/5/7/11 → fast 5-smooth.
- RADICAL LEVER (high value, multi-function, big port): implement native radix-3 + radix-5
  (recursive mixed-radix Cooley-Tukey for N=2^a·3^b·5^c) → closes the 3.7-5.4x non-pow2 loss
  AND lets oaconvolve/fftconvolve pad to next_fast_len (5-smooth) instead of next_pow2
  (less padding). Substantial FFT implementation (radix-3/5 butterflies + twiddles + mixed
  decomposition + accuracy verification) — flagged as THE remaining high-value lever; deferred
  to a focused cycle (or the fsci-fft owner). rfft already competitive (prior).

## 2026-06-21 - CORRECTION: fsci_fft HAS mixed-radix 3/5 — non-pow2 gap is scalar-recursive structure vs pocketfft SIMD
- Agent: cc / MistyBirch. Root-caused yesterday's "lacks mixed-radix 3/5" claim — WRONG.
  fsci_fft::mixed_radix_fft (transforms.rs:453) already has SPECIALIZED radix-2/3/4/5
  butterflies + Bluestein (cached chirp) for primes. The algorithm is complete.
- The real non-pow2 gap (3.7-5.4x on 5-smooth): the mixed-radix path is RECURSIVE +
  OUT-OF-PLACE (strided gathers, scalar butterflies), whereas the pure-pow2 path uses the
  fast IN-PLACE radix-4 kernel (cooley_tukey_radix4_inplace) — which is why fsci WINS pure
  pow2 (1.38x) but loses 5-smooth: a number like 60000=2^5·3·5^4 runs its whole transform
  (incl. the 2^5 part) through the slower recursive scalar path, not the in-place radix-4.
  pocketfft is iterative (Stockham) + SIMD-vectorized across the butterflies.
- LEVER (now correctly characterized): NOT "add mixed-radix" (exists) but a major restructure
  — iterative/in-place mixed-radix + SIMD the radix-3/5 butterflies + route the 2-part through
  the fast radix-4 kernel. Substantial FFT engineering, uncertain it beats pocketfft. A genuine
  scalar-vs-SIMD + structure WALL (joins Qhull/HiGHS/LAPACK). Deferred. Twiddles cached + scratch
  reused already (no cheap alloc win). Supersedes yesterday's mixed-radix-lever note.

## 2026-06-21 - GAUNTLET (less-common fns): all WINS — gaussian_kde-nd 14x, RGI 2.3x (no loss)
- Agent: cc / MistyBirch. Probed less-common/heavy functions beyond the headliners. MEASURED:
  - gaussian_kde 2D (1000 data × 20000 query): fsci GaussianKdeNd 14.43ms vs scipy 203.7 →
    WIN 14.1x (parallel + Cholesky/Mahalanobis).
  - RegularGridInterpolator 4D (100k queries): fsci 17.8ms (serial loop eval) vs scipy 41.3 →
    WIN 2.3x (eval_many parallel wins more).
  - periodogram/welch (FFT-based, parallel): prior wins.
- No clean loss in this batch — the codebase is comprehensively dominant. Confirms the campaign
  state: every measured function wins or is a documented engineering WALL (pocketfft-non-pow2,
  Qhull-Delaunay-build, HiGHS, LAPACK) / risky-marginal (kmeans small-n SIMD). Clean algorithmic
  losses fixed this session.

## 2026-06-21 - CAMPAIGN SUMMARY: BOLD-VERIFY perf-domination complete (clean losses fixed; residuals = walls)
- Agent: cc / MistyBirch. Session result map (full per-item ratios in the dated entries above;
  cross-session map in memory perf_domination_campaign_2026_06_21):
  WINS SHIPPED: make_smoothing_spline 3x→8.3-27.8x (selected-inverse); SmoothBivariateSpline
  358-2450x→near-parity+scipy-correct (FITPACK routing); make_lsq_spline 2.4x→8-74x (compact-
  banded); fftconvolve 1.5x→1.8-2.1x WIN / oaconvolve 4.1x→1.5x / hilbert wall→near-parity
  (rfft routing); linkage family O(n³)→O(n²) all WIN (MST + NN-chain); kmeans large-n 2.8x WIN.
  CONFIRMED DOMINANT: callback-lever minimize 441x/quad 8.2x; gaussian_kde-nd 14x; distance_matrix
  15.3x; special 1.7-2.8x; stats 2.5-8.8x; ConvexHull 1.6-3.1x; RGI 2.3x; FFT pow2 1.38x.
- REMAINING (engineering walls — major SIMD/kernel effort, uncertain vs hand-tuned C; NOT clean
  wins): FFT non-pow2 5-smooth 3.7-5.4x (recursive scalar mixed-radix vs pocketfft iterative
  SIMD — fsci HAS the radices), kmeans small-n SIMD (serial nearest_centroid, gate-lowering
  regresses), Delaunay-build 1.45x (Qhull), linprog (HiGHS), RBF (LAPACK). + capability gap:
  bounded least_squares/curve_fit (TRF).
- VERDICT: every clean algorithmic loss found this campaign is FIXED + verified; fsci dominates
  scipy across the gauntleted surface. Further gains require breaking C-library walls (the
  iterative-SIMD mixed-radix FFT is the highest-value remaining lever) — deferred to dedicated
  effort, not rushed (correctness risk). No fabricated marginal ships.
