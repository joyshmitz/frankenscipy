# Performance Negative-Evidence Ledger

This ledger records every code-first performance attempt, including attempts that
are still awaiting the batch benchmark wave. Entries must name the retry
condition so dead ends are not repeated casually.

## 2026-06-20 - frankenscipy-8l8r1.133 - cluster linkage compact active frontier

- Agent: cod-a / BlackThrush
- Lever kept: replace the boolean `active[k]` range scans in the flat NN-array
  linkage core with a sorted compact `active_ids` frontier. The frontier is
  maintained in ascending cluster-id order, so the pair selection and
  nearest-successor tie behavior stay byte-identical while update/refresh loops
  skip inactive clusters directly.
- Graveyard/artifact route tested: branchless/compact active-set maintenance
  at the nearest-neighbour frontier, after the earlier row-pack keep and
  lazy-arena reject left Average/Ward as measured SciPy losses.
- Decision: KEEP. The same-machine SciPy gauntlet improves Average by `1.87x`
  and Ward by `2.00x` versus current, moving both tracked rows from clear SciPy
  losses to near-parity/slight median wins.
- Artifact:
  `tests/artifacts/perf/2026-06-20-cod-a-linkage-133/EVIDENCE.md`
- Baseline command:
  `AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a-local cargo bench -p fsci-cluster --bench cluster_bench -- va60h_gauntlet_linkage --noplot`
- Candidate command:
  `AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a-local cargo bench -p fsci-cluster --bench cluster_bench -- va60h_gauntlet_linkage --noplot`

Benchmark evidence:

| Workload / route | Mean | Interval / value | Ratio / verdict |
| --- | ---: | ---: | --- |
| Current Rust `linkage(Average)` | 8.5503 ms | [7.9590, 8.9969] ms | baseline; 1.65x slower than same-run SciPy |
| Active-frontier Rust `linkage(Average)` | 4.5727 ms | [4.3609, 4.8932] ms; Criterion change -40.851% | 1.87x faster than current; 1.05x faster than SciPy median |
| SciPy `linkage(Average)` | 4.8204 ms | [4.5636, 4.9797] ms | oracle |
| Current Rust `linkage(Ward)` | 10.831 ms | [8.9537, 12.918] ms | baseline; 2.06x slower than same-run SciPy |
| Active-frontier Rust `linkage(Ward)` | 5.4267 ms | [5.1571, 5.6559] ms; Criterion change -48.487% | 2.00x faster than current; 1.04x faster than SciPy median |
| SciPy `linkage(Ward)` | 5.6168 ms | [5.2357, 6.1779] ms | oracle |

Win/loss/neutral:

- Same-machine candidate versus current: `2/0/0`.
- Candidate median versus same-machine SciPy oracle: `2/0/0`, with overlapping
  independent intervals, so treat as near-parity/slight wins rather than a
  broad dominance claim.

Correctness/conformance guards:

- PASS: rch focused bit-contract:
  `cargo test -p fsci-cluster linkage_flat_core_matches_precomputed_condensed_contract -- --nocapture`
  = 1 passed / 0 failed.
- PASS: rch broader linkage filter:
  `cargo test -p fsci-cluster linkage -- --nocapture` = 28 linkage unit tests
  and 9 linkage metamorphic tests passed.
- PASS: local live SciPy conformance:
  `cargo test -p fsci-conformance --test diff_cluster_linkage_from_distances -- --nocapture`
  = 1 passed / 0 failed.
- PASS: local live SciPy conformance:
  `cargo test -p fsci-conformance --test diff_cluster_linkage_helpers -- --nocapture`
  = 1 passed / 0 failed.
- PASS: rch per-crate release build:
  `cargo build --release -p fsci-cluster`; unrelated existing `perf_kmeans.rs`
  warning remained.
- PASS: rch per-crate all-targets check:
  `cargo check -p fsci-cluster --all-targets`; unrelated existing
  `perf_kmeans.rs` warning remained.
- PASS: rch no-deps clippy:
  `cargo clippy -p fsci-cluster --lib --no-deps -- -D warnings`.
- PASS: final local lib check:
  `cargo check -p fsci-cluster --lib`.
- PASS: `git diff --check` on touched files.
- PASS: changed-file UBS scan found 0 critical issues; existing broad
  `fsci-cluster` warning inventory remains.
- BLOCKED: `cargo fmt -p fsci-cluster --check` remains blocked by pre-existing
  crate-wide formatting drift, including files and hunks unrelated to this
  patch.

Retry condition: do not retry full-square arena initialization, zero/lazy-fill
variants, or pure row-packing for this exact Average/Ward frontier. The next
credible linkage route needs a true method-specific NN-chain primitive, smaller
candidate-distance frontier, or another algorithmic change with same-machine
SciPy proof.

## 2026-06-20 - frankenscipy-8l8r1.132 - ndimage gaussian_filter tile-local scratch

- Agent: cod-a / BlackThrush
- Lever kept: change the 2-D `BoundaryMode::Reflect`, order-0 Gaussian fast
  path from a full-image scratch plus two scoped thread barriers into a
  tile-local scratch pass. Each worker row chunk computes the vertical
  cache-planned AXPY pass into a thread-local scratch tile and immediately runs
  the horizontal pass into the output chunk.
- Graveyard/artifact route tested: cache-blocked separable layout and
  communication/data-movement reduction, after scalar reflect tap peeling,
  full-image scratch, and pure AXPY/tap-folding families left a residual SciPy
  loss.
- Decision: KEEP. Same-worker Criterion on `hz2` improves
  `gaussian_sigma2/256` from `1.9819 ms` to `1.2274 ms` (`1.61x` faster) and
  flips the row from `1.34x` slower than SciPy to `1.20x` faster than SciPy.
- Artifact:
  `tests/artifacts/perf/2026-06-20-cod-a-gaussian-tile-scratch/EVIDENCE.md`
- rch baseline command:
  `AGENT_NAME=BlackThrush RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo bench -p fsci-ndimage --bench ndimage_bench -- correlate_gaussian/gaussian_sigma2/256 --sample-size 10 --measurement-time 1 --warm-up-time 1`
- rch candidate command:
  `AGENT_NAME=BlackThrush RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz2 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo bench -p fsci-ndimage --bench ndimage_bench -- correlate_gaussian/gaussian_sigma2/256 --sample-size 10 --measurement-time 1 --warm-up-time 1`
- Current-route A/B profile command:
  `AGENT_NAME=BlackThrush RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz2 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo test -p fsci-ndimage gaussian_2d_axpy_ab_timing --release -- --ignored --nocapture`
- SciPy oracle command:
  `AGENT_NAME=BlackThrush python3 docs/perf_oracle_ndimage.py`

Benchmark evidence:

| Workload / route | Worker | Mean | Interval / value | Ratio / verdict |
| --- | --- | ---: | ---: | --- |
| Current Rust `ce1857ab` | `hz2` | 1.9819 ms | [1.8484, 2.2050] ms | baseline; 1.34x slower than SciPy |
| Tile-local scratch candidate | `hz2` | 1.2274 ms | [1.1564, 1.2960] ms; Criterion change -40.721% | 1.61x faster than current; 1.20x faster than SciPy |
| Same-process gather toggle | `hz2` | 2760.0 us | one binary, interleaved | current-route profile arm |
| Same-process AXPY toggle | `hz2` | 2430.3 us | one binary, interleaved | 1.14x faster than gather |
| SciPy `ndimage.gaussian_filter` | local SciPy oracle | 1.47367 ms | p50 | oracle |

Win/loss/neutral:

- Same-worker candidate versus current: `1/0/0`.
- Final candidate versus SciPy oracle: `1/0/0`.
- Prior final-source Gaussian score was `0/1/0`; this closes the tracked
  `gaussian_sigma2/256` loss.

Correctness/conformance guards:

- PASS: rch focused Gaussian suite:
  `cargo test -p fsci-ndimage gaussian --lib -- --nocapture` =
  31 passed / 0 failed / 1 ignored.
- PASS: local live SciPy conformance:
  `FSCI_REQUIRE_SCIPY_ORACLE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a-local cargo test -p fsci-conformance --test diff_ndimage_gaussian_filter -- --nocapture`
  = 1 passed / 0 failed.
- PASS: rch per-crate compile:
  `cargo check -p fsci-ndimage --all-targets` passed on `vmi1149989`;
  unrelated existing warnings remained in `fsci-interpolate` and `diff_geom`.
- PASS: `git diff --check`.
- PASS: changed-file UBS scan exited 0 with 0 critical findings; existing
  broad warnings in `crates/fsci-ndimage/src/lib.rs` remain inventory.
- BLOCKED: `cargo fmt -p fsci-ndimage -- --check` remains blocked by
  pre-existing formatting drift in `ndimage_bench.rs`, `diff_fourier.rs`, and
  older `src/lib.rs` hunks outside this change.
- BLOCKED: `cargo clippy -p fsci-ndimage --all-targets -- -D warnings` stopped
  before this patch on existing `fsci-linalg` dependency lints
  (`needless_range_loop` and `needless_borrow`).

Retry condition: do not retry full-image scratch plus two-barrier separable
layout for this path. Future work must target smaller residual overhead such
as fixed-radius/source-plan specialization or plan caching, and must retain
same-worker proof.

## 2026-06-20 - frankenscipy-8l8r1.131 - sparse eigsh projected-residual certificate

- Agent: cod-a / BlackThrush
- Lever kept: trust the symmetric Arnoldi/Lanczos projected residual certificate
  from the tridiagonal/Hessenberg problem for `eigsh` when `k<=6`, avoiding `k`
  explicit post-hoc sparse residual matvecs.
- Lever rejected: using the projected certificate unconditionally. The `k=8`
  row regressed on the same worker despite fewer matvecs, so final source keeps
  the older explicit residual check for `k>6`.
- Graveyard/artifact route tested: communication-avoiding residual validation
  and specialization by small target subspace size; this is a certificate/layout
  lever, not another row-major basis arena.
- Artifact:
  `tests/artifacts/perf/2026-06-20-cod-a-sparse-eigsh-tridiag/EVIDENCE.md`
- rch command:
  `AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo run --release -p fsci-sparse --bin perf_eigsh`
- Local SciPy oracle command: Python 3.13.7 / NumPy 2.4.3 / SciPy 1.17.1
  reimplementation of `perf_eigsh`'s deterministic matrix generator.
- Same-worker internal A/B on `vmi1227854`:

  | Workload | Baseline Rust | Candidate/final `k<=6` route | Internal delta | Prior SciPy oracle | Candidate vs SciPy |
  | --- | ---: | ---: | ---: | ---: | --- |
  | `eigsh n=2000 k=6` | 1.169 ms, 26 matvecs | 1.024 ms, 20 matvecs | 1.14x faster | 3.000 ms | Rust 2.93x faster |
  | `eigsh n=8000 k=6` | 4.789 ms, 26 matvecs | 4.003 ms, 20 matvecs | 1.20x faster | 2.768 ms | Rust 1.45x slower |
  | `eigsh n=20000 k=8` raw projected candidate | 10.672 ms, 28 matvecs | 12.289 ms, 20 matvecs | 1.15x slower | 43.023 ms | reject raw k=8 route |

- Final-source sanity on rch `vmi1152480`:

  | Workload | Final Rust | Matvecs | Converged | Max residual |
  | --- | ---: | ---: | --- | ---: |
  | `eigsh n=2000 k=6` | 1.091 ms | 20 | true | 1.96e-11 |
  | `eigsh n=8000 k=6` | 4.797 ms | 20 | true | 3.45e-11 |
  | `eigsh n=20000 k=8` | 12.042 ms | 28 | true | 2.57e-11 |

- Fresh local SciPy oracle:

  | Workload | SciPy median | Final remote Rust vs fresh SciPy |
  | --- | ---: | --- |
  | `eigsh n=2000 k=6` | 1.538154 ms | Rust 1.41x faster |
  | `eigsh n=8000 k=6` | 3.424127 ms | Rust 1.40x slower |
  | `eigsh n=20000 k=8` | 7.874257 ms | Rust 1.53x slower on this cross-host oracle |

- Win/loss/neutral:
  - Same-worker raw candidate versus current: `2/1/0`; final source keeps the
    two `k=6` wins and guards off the `k=8` loss.
  - Prior-ledger SciPy score for the final guarded route: `2/1/0`; the live
    mid-size loss improves from `4.789/2.768 = 1.73x` slower to
    `4.003/2.768 = 1.45x` slower on the acceptance worker.
  - Fresh local SciPy oracle versus final remote Rust: `1/2/0`; recorded as
    routing evidence because Rust and SciPy were not same-host.
- Correctness/conformance guards:
  - PASS: rch `cargo check -p fsci-sparse --all-targets`.
  - PASS: rch focused `cargo test -p fsci-sparse eigsh --lib -- --nocapture`.
  - PASS: rch `cargo clippy -p fsci-sparse --all-targets --no-deps -- -D warnings`.
  - PASS: local live SciPy
    `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test diff_sparse_eigsh_largest -- --nocapture`.
  - PASS: `git diff --check`.
  - PASS: `ubs crates/fsci-sparse/src/linalg.rs` completed with 0 critical
    findings and no new hunk-specific finding.
  - BLOCKED: local same-host Rust release timing with the shared target stopped
    on `E0514` because the release target contained worker-built artifacts from
    rustc `beae78130` while local rustc was `f20a92ec0`; the target was not
    cleaned.
  - BLOCKED: `cargo fmt -p fsci-sparse -- --check` reports pre-existing sparse
    crate formatting drift in `sparse_bench.rs`, `src/lib.rs`, and older
    `src/linalg.rs` hunks outside this patch.
- Retry condition: do not retry the row-major Arnoldi basis arena, mutable
  operator scratch, or unconditional residual removal for `k>6`. Route the
  remaining `n=8000, k=6` loss to implicit/thick restart, tridiagonal-only
  eigensolve specialization, or deterministic warm-start subspace reuse.

## 2026-06-20 - frankenscipy-8l8r1.128 - cluster linkage row-pack keep + lazy-arena reject

- Agent: cod-a / BlackThrush
- Lever kept: pack nested observation rows once in `linkage` and build pairwise
  distances from contiguous row slices. The Lance-Williams update path, tie
  order, error behavior, and full row-major inter-cluster arena are unchanged.
- Lever rejected: initialize the full inter-cluster arena with zeroes instead
  of `f64::INFINITY`, relying on later active-pair writes. This was intended to
  remove a full-memory fill, but it did not pay in the real `linkage` group.
- Graveyard/artifact route tested: data-layout pressure at the pairwise
  distance frontier and allocator/fill removal in the full arena, after the
  earlier compact triangular arena lost to row-major locality.
- Decision: KEEP the row-pack lever and REJECT/REVERT lazy arena zeroing. Ward
  gets a measured internal win; Average is neutral/slightly better. Both rows
  remain SciPy losses, so this is not a parity claim.
- Artifact:
  `tests/artifacts/perf/2026-06-20-cod-a-linkage-lazy-arena-EVIDENCE.md`
- rch baseline command:
  `AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo bench -p fsci-cluster --bench cluster_bench -- va60h_gauntlet_linkage --noplot`
- Local SciPy head-to-head command:
  `AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a-local cargo bench -p fsci-cluster --bench cluster_bench -- va60h_gauntlet_linkage --noplot`
- rch gate commands:
  `cargo build --release -p fsci-cluster`,
  `cargo test -p fsci-cluster linkage_flat_core_matches_precomputed_condensed_contract -- --nocapture`,
  `cargo test -p fsci-cluster linkage_ -- --nocapture`, and
  `cargo clippy -p fsci-cluster --lib --no-deps -- -D warnings`.
- Baseline/head-to-head evidence:

  | Workload | Baseline Rust | SciPy oracle | Baseline vs SciPy |
  | --- | ---: | ---: | --- |
  | `linkage(Average)`, n=800 d=4 | 7.1834 ms | 5.0346 ms | 1.427x slower |
  | `linkage(Ward)`, n=800 d=4 | 8.2387 ms | 5.3080 ms | 1.552x slower |

- Rejected candidate evidence:

  | Workload | Current baseline | Lazy-arena candidate | SciPy oracle | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | `linkage(Average)`, n=800 d=4 | 7.1834 ms | 7.6203 ms | 4.5097 ms | 1.061x slower than current; 1.690x slower than SciPy |
  | `linkage(Ward)`, n=800 d=4 | 8.2387 ms | 8.2002 ms | 5.2550 ms | 1.005x faster than current; still 1.560x slower than SciPy |

- Final source evidence:

  | Workload | Baseline Rust | Final Rust | Internal delta | Same-run SciPy oracle | Final vs SciPy |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | `linkage(Average)`, n=800 d=4 | 7.1834 ms | 7.1304 ms | 1.007x faster | 4.3843 ms | 1.626x slower |
  | `linkage(Ward)`, n=800 d=4 | 8.2387 ms | 6.9591 ms | 1.184x faster | 4.8687 ms | 1.429x slower |

- Win/loss/neutral:
  - Baseline current versus SciPy: `0/2/0`.
  - Lazy-arena candidate versus current: `0/1/1`; source reverted.
  - Final row-pack source versus current baseline: `1/0/1`.
  - Final row-pack source versus SciPy: `0/2/0`.
- Correctness/conformance guards:
  - PASS: `perf_linkage` isomorphism printed **0 mismatches / 7200 linkage
    matrices** after the row-pack change.
  - PASS: focused linkage contract test via rch:
    `cargo test -p fsci-cluster linkage_flat_core_matches_precomputed_condensed_contract -- --nocapture`.
  - PASS: broader filtered linkage tests via rch:
    `cargo test -p fsci-cluster linkage_ -- --nocapture`.
  - PASS: release build via rch:
    `cargo build --release -p fsci-cluster`; an unrelated warning remained in
    `crates/fsci-cluster/src/bin/perf_kmeans.rs`.
  - PASS: no-deps clippy via rch:
    `cargo clippy -p fsci-cluster --lib --no-deps -- -D warnings`.
  - PASS: changed-file UBS scan exited 0 with 0 critical findings; existing
    warning inventory remains in `fsci-cluster/src/lib.rs`.
  - BLOCKED: rch SciPy benchmark rows failed because the selected worker could
    not import `scipy`; local SciPy 1.17.1 supplied the head-to-head oracle.
  - BLOCKED: dependency-inclusive clippy stops before this patch on existing
    `fsci-linalg` lints.
  - BLOCKED: `rustfmt --edition 2024 --check crates/fsci-cluster/src/lib.rs`
    reports pre-existing file-wide drift outside this patch; this patch did not
    run a broad format churn.
- Negative evidence: do not retry zero/lazy initialization of the full
  inter-cluster arena on this route. Also avoid another full-square-to-
  triangular storage tweak unless a fresh profile shows the row-major scan cost
  has moved. The next SciPy-gap attack should change the primitive: NN-chain,
  method-specific nearest-neighbour maintenance, MST/single-linkage style
  specialization where applicable, or a tighter compiled-distance frontier.
## 2026-06-20 - frankenscipy-8l8r1.129 - ndimage gaussian_filter 2D reflect cache-planned separable pass

- Agent: cod-b / MistyBirch
- Lever: specialize the common `gaussian_filter(input, sigma, Reflect, cval)`
  2-D, order-0, all-axes path by precomputing reflect source-index plans for
  rows and columns, then applying separable vertical and horizontal passes over
  row chunks. All other modes, dimensions, orders, and axis subsets retain the
  generic route.
- Graveyard/artifact route tested: cache-aware separable source plans and
  branch/index hoisting at the first-axis generic N-D convolution layer. This
  is intentionally deeper than the previously rejected scalar row-contiguous
  border/interior tap split.
- Decision: KEEP. Same-worker `vmi1152480` proof improves
  `gaussian_sigma2/256` from `3.2989 ms` to `1.9680 ms` (`1.68x` faster).
  Final Rust is still `1.34x` slower than the SciPy oracle, so the residual
  loss remains routed to vectorized/tiled stencil constants.
- Artifact:
  `tests/artifacts/perf/frankenscipy-8l8r1.128-gaussian-cache-planned/EVIDENCE.md`
- Pre-edit Rust command:
  `AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo bench -p fsci-ndimage --bench ndimage_bench -- correlate_gaussian/gaussian_sigma2/256 --noplot`
- Final Rust command:
  `AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo bench -p fsci-ndimage --bench ndimage_bench -- correlate_gaussian/gaussian_sigma2/256 --noplot`
- Same-worker clean-current command on `vmi1152480`:
  `ssh -i ~/.ssh/contabo_vps_ed25519 -o BatchMode=yes root@109.205.181.92 'cd /data/projects/.scratch/frankenscipy-cod-b-gaussian-baseline-vmi115-20260620 && AGENT_NAME=MistyBirch CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b-vmi115-baseline cargo bench -p fsci-ndimage --bench ndimage_bench -- correlate_gaussian/gaussian_sigma2/256 --noplot'`
- SciPy oracle command:
  `python3 docs/perf_oracle_ndimage.py`
- Benchmark evidence:

  | Workload / route | Worker | Mean | Interval | Ratio / verdict |
  | --- | --- | ---: | ---: | --- |
  | Clean current Rust `ae454655` | `vmi1152480` | 3.2989 ms | [3.1976, 3.4050] ms | 2.25x slower than SciPy |
  | Candidate cache-planned separable pass | `vmi1152480` | 1.9680 ms | [1.8608, 2.0797] ms | 1.68x faster than current; 1.34x slower than SciPy |
  | SciPy `ndimage.gaussian_filter` | local SciPy 1.17.1 | 1.46523 ms | p50 | oracle |

- Routing-only evidence:
  - Pre-edit RCH current on `vmi1227854`: `2.8418 ms`, `1.94x` slower than
    SciPy. Not used as keep/reject proof because the candidate proof worker was
    `vmi1152480`.
  - Clean current on RCH `vmi1149989`: `5.8852 ms`, `4.02x` slower than SciPy.
    Not used as keep/reject proof because no candidate sample landed there.
  - First candidate sample on RCH `vmi1152480`: `2.1316 ms`, `1.45x` slower
    than SciPy.
- Win/loss/neutral:
  - Same-worker candidate versus clean current: `1/0/0`.
  - Final candidate versus SciPy oracle: `0/1/0`.
- Correctness/conformance guards:
  - PASS: `cargo check -p fsci-ndimage --all-targets` via RCH `hz1`; unrelated
    warnings remained in `fsci-interpolate` and `diff_geom`.
  - PASS: `cargo test -p fsci-ndimage gaussian_filter --lib -- --nocapture`
    via RCH `hz1` = **13 passed / 0 failed / 230 filtered**, including
    `gaussian_filter_reflect_2d_fast_path_matches_generic_sequential_path`.
  - PASS: local live SciPy conformance:
    `FSCI_REQUIRE_SCIPY_ORACLE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b-local cargo test -p fsci-conformance --test diff_ndimage_gaussian_filter -- --nocapture`
    = **1 passed / 0 failed**.
  - BLOCKED: RCH live SciPy conformance on `vmi1152480` failed before
    comparison because the worker has no `scipy` module installed.
  - PASS: `git diff --check`.
  - PASS: UBS on the changed code/docs/bead/artifact set; exit 0, no
    criticals, broad pre-existing warning inventory in `lib.rs`.
  - BLOCKED: `cargo fmt --check -p fsci-ndimage` remains blocked by
    pre-existing formatting drift in `ndimage_bench.rs` and `diff_fourier.rs`;
    touched `src/lib.rs` was formatted directly.
  - BLOCKED: `cargo clippy -p fsci-ndimage --all-targets -- -D warnings`
    remains blocked before ndimage by pre-existing `fsci-linalg` dependency
    lints.
- Negative evidence: cache-planned reflect source tables are worthwhile, but
  still not SciPy parity. Do not retry the reverted scalar row-contiguous
  border/interior split or always-line-walk outer-axis split. The next route
  should be a vector-friendly row/column dot kernel, transposed scratch for the
  vertical pass, or cache-blocked separable tiles with the same reflect plan.

## 2026-06-20 - frankenscipy-8l8r1.130 - ndimage gaussian_filter folded AXPY reflect pass

- Agent: cod-b / MistyBirch
- Lever: keep the 2-D `BoundaryMode::Reflect`, order-0 `gaussian_filter` fast
  path from `.129`, but replace the first-axis strided gather-dot with folded
  symmetric row AXPY. The Gaussian kernel is symmetric, so each output can be
  computed as center tap plus paired reflected rows. The hot first pass now
  streams over contiguous rows instead of gathering at stride `cols`.
- Graveyard/artifact route tested: vectorized execution and polyhedral/stencil
  loop restructuring. This is the non-repeated route after scalar reflect tap
  peeling and outer-axis line-walk variants lost.
- Decision: KEEP as an internal win. Same-worker `vmi1167313` proof improves
  `gaussian_sigma2/256` from `6.9394 ms` to `3.3918 ms` (`2.05x` faster), and
  the in-binary A/B toggle moves gather `3585.0 us` to AXPY `2943.3 us`
  (`1.22x` faster). Final Rust remains `0/1/0` versus SciPy, so the residual
  loss stays routed deeper.
- Artifact:
  `tests/artifacts/perf/frankenscipy-8l8r1.130-gaussian-axpy/EVIDENCE.md`
- Clean baseline command:
  `git worktree add --detach /data/projects/.scratch/frankenscipy-cod-b-gaussian-axpy-baseline-20260620T1044 origin/main`
- Same-worker clean-current command:
  `AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1152480 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo bench -p fsci-ndimage --bench ndimage_bench -- correlate_gaussian/gaussian_sigma2/256 --noplot --sample-size 10 --measurement-time 1 --warm-up-time 1`
- Same-worker candidate command:
  `AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1167313 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo bench -p fsci-ndimage --bench ndimage_bench -- correlate_gaussian/gaussian_sigma2/256 --noplot --sample-size 10 --measurement-time 1 --warm-up-time 1`
- Same-process A/B command:
  `AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1167313 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo test -p fsci-ndimage gaussian_2d_axpy_ab_timing --release -- --ignored --nocapture`
- SciPy oracle command:
  `python3 docs/perf_oracle_ndimage.py`
- Benchmark evidence:

  | Workload / route | Worker | Mean | Interval / value | Ratio / verdict |
  | --- | --- | ---: | ---: | --- |
  | Clean current Rust `0cf3cc42` | `vmi1167313` | 6.9394 ms | [5.1048, 9.1535] ms | baseline; noisy row with severe outlier |
  | Candidate AXPY Rust | `vmi1167313` | 3.3918 ms | [2.9580, 3.7948] ms | 2.05x faster than current; 2.91x slower than SciPy |
  | Same-process gather toggle | `vmi1167313` | 3585.0 us | one binary, interleaved | baseline arm |
  | Same-process AXPY toggle | `vmi1167313` | 2943.3 us | one binary, interleaved | 1.22x faster than gather |
  | Final-source routing sanity | `vmi1149989` | 3.0285 ms | [2.8418, 3.3639] ms | 2.59x slower than SciPy; not paired |
  | SciPy `ndimage.gaussian_filter` | local SciPy oracle | 1.16724 ms | p50 | oracle |

- Win/loss/neutral:
  - Same-worker candidate versus clean current: `1/0/0`.
  - Same-process A/B candidate versus gather path: `1/0/0`.
  - Final candidate versus SciPy oracle: `0/1/0`.
- Correctness/conformance guards:
  - PASS: exact final local source:
    `cargo test -p fsci-ndimage gaussian_2d_axpy_matches_gather_dot --lib -- --nocapture`
    = **1 passed / 0 failed**.
  - PASS: focused Gaussian suite via RCH:
    `cargo test -p fsci-ndimage gaussian --lib -- --nocapture`
    = **31 passed / 0 failed / 1 ignored**.
  - PASS: local live SciPy conformance:
    `FSCI_REQUIRE_SCIPY_ORACLE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo test -p fsci-conformance --test diff_ndimage_gaussian_filter -- --nocapture`
    = **1 passed / 0 failed**.
  - PASS: exact final local source crate check:
    `cargo check -p fsci-ndimage --all-targets`.
  - PASS: `git diff --check`.
  - PASS: changed-file UBS scan exited 0 with 0 criticals; the scan reported
    the existing broad warning inventory in `crates/fsci-ndimage/src/lib.rs`.
  - BLOCKED: `cargo fmt --check -p fsci-ndimage` remains blocked by
    pre-existing formatting drift in `ndimage_bench.rs`, `diff_fourier.rs`, and
    unrelated existing `src/lib.rs` sections.
  - BLOCKED: `cargo clippy -p fsci-ndimage --all-targets -- -D warnings`
    remains blocked before ndimage by pre-existing `fsci-linalg` dependency
    lints.
  - PRE-EXISTING WARNINGS: `fsci-interpolate`, `fsci-special`, and
    `crates/fsci-ndimage/src/bin/diff_geom.rs` warnings appeared during gates.
- Negative evidence: folded row AXPY pays, but does not close the SciPy gap.
  Do not retry scalar reflect tap peeling or always-line-walk outer-axis
  variants. Next route should make the horizontal pass stride-1 too, via
  transposed scratch or cache-blocked separable tiles, then remove the runtime
  toggle if the no-toggle production path is measurably faster.

## 2026-06-20 - frankenscipy-8l8r1.127 - EDT feature-transform line starts

- Agent: cod-b / MistyBirch
- Lever: tighten `distance_transform_edt(return_indices=True)` after the
  feature-transform complexity keep by enumerating exact separable line starts
  directly instead of scanning every flat index with division/modulo, then
  materializing 2-D output indices with flat row/column arithmetic and reusing a
  coordinate scratch buffer for generic ndim. The feature-transform math,
  winner propagation, tie behavior, and background-free fallback are unchanged.
- Graveyard/artifact route tested: cache/data-movement and allocation removal
  at the remaining constant-factor layer. This is the non-repeated path after
  the prior catastrophic O(foreground * background) brute-force gap was closed:
  avoid dead line-start scans and per-cell coordinate Vec allocation rather
  than changing the nearest-background algorithm.
- Decision: KEEP. Same-worker rch `vmi1152480` improves every
  `return_indices` row versus the prior feature-transform route, with a strict
  final SciPy score of `1/3/0`. This is still not full parity, so the residual
  loss remains routed to deeper feature-transform constants.
- Artifact:
  `tests/artifacts/perf/frankenscipy-8l8r1.127-edt-line-starts-EVIDENCE.md`
- Baseline Rust command:
  `AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo run --release -p fsci-ndimage --bin perf_edt`
- Final Rust command:
  `AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1152480 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo run --release -p fsci-ndimage --bin perf_edt`
- SciPy oracle command:
  `python3 docs/perf_oracle_edt_indices.py --reps 20`
- Benchmark evidence:

  | Image | Prior Rust `feat` | Final Rust `feat` | Internal speedup | SciPy oracle | Final vs SciPy |
  | --- | ---: | ---: | ---: | ---: | --- |
  | 64x64 | 325.742 us | 216.733 us | 1.50x | 173.434 us | 1.25x slower |
  | 128x128 | 1.380 ms | 1.207 ms | 1.14x | 775.685 us | 1.56x slower |
  | 192x192 | 3.814 ms | 2.107 ms | 1.81x | 2.280155 ms | 1.08x faster |
  | 256x256 | 5.854 ms | 4.855 ms | 1.21x | 4.288605 ms | 1.13x slower |

- Secondary distance-only rows also improved on the same run
  (`259.977 us -> 234.453 us`, `1.094 ms -> 1.038 ms`,
  `2.206 ms -> 2.119 ms`) because the same direct line-start enumeration feeds
  the distance-only separable pass.
- SciPy win/loss/neutral for final source: `1/3/0`.
- Same-worker internal keep/loss/neutral versus previous feature-transform
  route: `4/0/0`.
- Correctness/conformance guards:
  - PASS: `perf_edt` isomorphism printed **0 mismatches / 10876 cells** on
    baseline and final runs, with identical digest rows.
  - PASS: local live SciPy conformance:
    `FSCI_REQUIRE_SCIPY_ORACLE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo test -p fsci-conformance --test diff_ndimage_distance_transform_edt -- --nocapture`
    = **1 passed / 0 failed**.
  - PASS: focused ndimage EDT tests via rch:
    `cargo test -p fsci-ndimage distance_transform_edt --lib -- --nocapture`
    = **15 passed / 0 failed**.
  - PASS: full ndimage lib tests via rch:
    `cargo test -p fsci-ndimage --lib -- --nocapture` =
    **242 passed / 0 failed**.
  - PASS: `cargo check -p fsci-ndimage --all-targets` via rch `hz1`;
    unrelated warnings remained in `fsci-interpolate` and `diff_geom`.
  - PASS: touched-file rustfmt and `git diff --check`.
  - BLOCKED: `cargo fmt -p fsci-ndimage --check` remains blocked by
    pre-existing rustfmt drift in `crates/fsci-ndimage/benches/ndimage_bench.rs`
    and `crates/fsci-ndimage/src/bin/diff_fourier.rs`.
  - BLOCKED: `cargo clippy -p fsci-ndimage --all-targets -- -D warnings`
    stopped before this patch on existing `fsci-linalg` dependency lints
    (`needless_range_loop`, `needless_borrow`).
- Negative evidence: do not retry the old `for base in 0..n` plus
  `(base / stride).is_multiple_of(len)` line-start filter or per-output
  `input.unravel(nearest_flat)` allocation in this path. The next attempt must
  be below this layer: fused/tiled axis passes, a scratch layout that reduces
  feature-index traffic, SIMD-friendly 1-D lower-envelope work, or a
  specialized 2-D feature-transform kernel with the same nearest-background
  validity proof.

## 2026-06-20 - frankenscipy-6l77z - ndimage gaussian_filter inner1 reflect reject

- Agent: cod-a / MistyBirch
- Lever: specialize `convolve1d_along_axis` for the row-contiguous
  `inner == 1`, `BoundaryMode::Reflect`, odd-kernel, `origin == 0` path used by
  the final axis of `gaussian_filter(..., sigma=2.0)` on 2-D images. The
  candidate kept border samples on the existing `boundary_index_1d` path and
  used direct in-bounds indexing only for the interior tap dot.
- Graveyard/artifact route tested: cache-line contiguous stencil specialization
  and branch removal inside the dense dot, after the previous always-line-walk
  and outer-axis row-splitting route was rejected.
- Decision: REJECT AND REVERT. Same-worker Criterion on rch `hz2` regressed the
  realistic gaussian row by `1.17x` versus current, so no source change shipped.
- Artifact:
  `tests/artifacts/perf/2026-06-20-ndimage-gaussian-inner1-reflect-reject/EVIDENCE.md`
- Baseline/current Rust command:
  `AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo bench -p fsci-ndimage --bench ndimage_bench -- correlate_gaussian/gaussian_sigma2/256 --noplot`
- Candidate Rust command:
  `AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz2 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo bench -p fsci-ndimage --bench ndimage_bench -- correlate_gaussian/gaussian_sigma2/256 --noplot`
- SciPy oracle command:
  `AGENT_NAME=MistyBirch python3 docs/perf_oracle_ndimage.py`
- Benchmark evidence:

  | Workload / route | Mean | Interval | Ratio / verdict |
  | --- | ---: | ---: | --- |
  | Current Rust `gaussian_sigma2/256`, rch `hz2` | 3.4399 ms | [3.3426, 3.5375] ms | 3.03x slower than SciPy |
  | Candidate inner1 reflect direct interior dot, rch `hz2` | 4.0213 ms | [3.8424, 4.1989] ms | 1.17x slower than current; 3.54x slower than SciPy |
  | SciPy `ndimage.gaussian_filter`, local SciPy 1.17.1 | 1.13557 ms | p50 | oracle |

- Win/loss/neutral:
  - Same-worker candidate versus current: `0/1/0`.
  - Final restored current versus SciPy oracle: `0/1/0`.
- Correctness/conformance guards:
  - PASS: focused gaussian guard via rch `hz2`:
    `cargo test -p fsci-ndimage gaussian_filter1d_matches_scipy_axis1_reflect --lib -- --nocapture`
    = **1 passed / 0 failed**.
  - PASS: local live SciPy conformance:
    `FSCI_REQUIRE_SCIPY_ORACLE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo test -p fsci-conformance --test diff_ndimage -- --nocapture`
    = **5 passed / 0 failed**.
  - PASS: candidate source was reverted before staging; the remaining production
    `crates/fsci-ndimage/src/lib.rs` diff belongs to a separate live EDT worker
    and is not part of this commit.
- Negative evidence: do not retry another scalar `inner == 1` reflect-only
  interior/border split for this gaussian workload without a new profile proving
  the old closure/boundary branch is still dominant. The next plausible route is
  a deeper separable-layout primitive: transpose/cache-tile between axes so both
  passes are contiguous, or introduce a shared vector-friendly dot kernel with a
  same-worker A/B against the restored current route.

## 2026-06-20 - frankenscipy-8l8r1.126 - label mean one-based contiguous index

- Agent: cod-b / MistyBirch
- Lever: specialize `ndimage.mean(input, labels, index)` for the common
  `index == [1, 2, ..., K]` case. This bypasses dense label-table allocation and
  maps exact positive integer labels directly to `label - 1`, while all
  duplicate, sparse, zero-containing, and arbitrary indexes keep the existing
  dense-table/hash routes.
- Graveyard/artifact route tested: cache/hardware-wall constant reduction after
  the O(N + K) complexity gap was already closed. The useful lever was removing
  a table indirection and a short-lived allocation from the hot scalar
  reduction loop, with an isomorphism proof and same-binary A/B.
- Decision: KEEP. Strict same-host SciPy score improves from `0/4/0` to
  `1/3/0`; the largest-K rows are now near parity, but this is not full release
  speed parity.
- Artifact:
  `tests/artifacts/perf/frankenscipy-8l8r1.126-label-mean-one-based-EVIDENCE.md`
- Baseline/final Rust command:
  `AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz2 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo run --release -p fsci-ndimage --bin perf_label_stats`
- Same-host Rust command:
  `/data/projects/.rch-targets/frankenscipy-cod-b/release/perf_label_stats`
- SciPy oracle command:
  `python3 docs/perf_oracle_label_stats.py --reps 50`
- Benchmark evidence:

  | N | K | Prior dense table | New one-based | SciPy oracle | Verdict |
  | ---: | ---: | ---: | ---: | ---: | --- |
  | 65536 | 512 | 154.810 us | 153.257 us | 0.189 ms | internal 1.01x; Rust 1.23x faster than SciPy |
  | 262144 | 1024 | 646.632 us | 634.996 us | 0.585 ms | internal 1.02x; Rust 1.09x slower than SciPy |
  | 262144 | 2048 | 857.822 us | 687.054 us | 0.576 ms | internal 1.25x; Rust 1.19x slower than SciPy |
  | 589824 | 4096 | 1.782 ms | 1.423 ms | 1.380 ms | internal 1.25x; Rust 1.03x slower than SciPy |

- rch `hz2` same-binary A/B independently showed `1.08x / 1.08x / 1.36x /
  1.33x` speedups over the reconstructed prior dense-table route with
  `mism=0/0/0/0/0`.
- SciPy win/loss/neutral for final source: `1/3/0` strict.
- Same-host internal keep/loss/neutral versus previous dense-table route:
  `4/0/0`.
- Correctness/conformance guards:
  - PASS: focused one-based semantics test via rch:
    `cargo test -p fsci-ndimage mean_one_based_contiguous_lookup_preserves_exact_label_semantics --lib -- --nocapture`
    = **1 passed / 0 failed**.
  - PASS: full ndimage lib tests via rch:
    `cargo test -p fsci-ndimage --lib -- --nocapture` =
    **242 passed / 0 failed**.
  - PASS: local live SciPy conformance:
    `FSCI_REQUIRE_SCIPY_ORACLE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo test -p fsci-conformance --test diff_ndimage -- --nocapture`
    = **5 passed / 0 failed**.
  - PASS: `cargo check -p fsci-ndimage --all-targets` via rch `hz1`;
    unrelated warnings remained in `fsci-interpolate` and `diff_geom`.
  - PASS: touched-file rustfmt and `git diff --check`.
  - PASS: changed-file UBS exited 0 with 0 critical issues; warning inventory
    remains non-blocking.
  - BLOCKED: `cargo clippy -p fsci-ndimage --all-targets -- -D warnings`
    stopped before this patch on existing `fsci-linalg` lints
    (`needless_range_loop`, `needless_borrow`).
- Negative evidence: do not retry another dense-table, `fract()`,
  `is_finite()`, HashMap, or `Vec<Vec<f64>>` grouping variant for this workload.
  The next attempt must be a deeper reduction primitive: parallel/cache-tiled
  sum/count accumulation, vector-friendly label ingestion, or sorted/run-grouped
  spans with a same-binary A/B against this one-based route.

## 2026-06-20 - frankenscipy-va60h - linkage triangular arena reject

- Agent: cod-a / MistyBirch
- Lever: replace the retained flat full inter-cluster linkage arena with a
  compact upper-triangular arena. The graveyard/artifact route was aggressive
  cache/data-movement reduction: store half the distances, keep successor rows
  contiguous, and remove mirrored writes while preserving the same strict `<`
  ascending successor scan and Lance-Williams operand order.
- Decision: REJECT and REVERT. Exact-output correctness passed, but the
  candidate regressed both SciPy head-to-head rows. The production code was
  restored to the prior flat row-major full arena before commit.
- Artifact:
  `tests/artifacts/perf/2026-06-20-va60h-triangular-arena-reject-EVIDENCE.md`
- Baseline/head-to-head command:
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a-local cargo bench -p fsci-cluster --bench cluster_bench -- va60h_gauntlet_linkage --noplot`
- rch correctness command:
  `RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo run -p fsci-cluster --release --bin perf_linkage`
- Correctness evidence: `perf_linkage` printed `isomorphism: 0 mismatches /
  7200 linkage matrices`; rch returned exit 102 only after the already-finished
  remote command because artifact retrieval timed out.
- Final restored-route guards: `cargo test -p fsci-cluster
  linkage_from_distances --lib -- --nocapture` passed 2/2 tests, and
  `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test
  diff_cluster_linkage_from_distances -- --nocapture` passed 1/1 tests against
  local SciPy 1.17.1.
- Benchmark evidence:

  | Workload | Restored current | Triangular candidate | SciPy oracle | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | `linkage(Average)`, n=800 d=4 | 7.5772 ms | 8.8260 ms | 4.2755 ms | candidate 1.165x slower than current; 2.064x slower than SciPy |
  | `linkage(Ward)`, n=800 d=4 | 7.4597 ms | 9.9240 ms | 5.4866 ms | candidate 1.330x slower than current; 1.809x slower than SciPy |

- SciPy win/loss/neutral for candidate: `0/2/0`.
- Internal keep/loss/neutral versus restored current: `0/2/0`.
- Retry condition: do not retry full-square-to-triangular arena layout for this
  NN-array linkage path unless profiling shows index arithmetic and new-cluster
  scatter are no longer the hot cost, or the algorithm changes to avoid the
  repeated arbitrary triangular lookups entirely. Route next work to the
  algorithmic gap with SciPy's compiled linkage implementation: NN-chain/MST
  specializations, lower-constant nearest-neighbour maintenance, or method-
  specific kernels.

## 2026-06-20 - frankenscipy-wh8ac - jnjnp_zeros Cephes j1 seed evaluator

- Agent: cod-b / MistyBirch
- Lever: replace the local `j1_core` series/asymptotic evaluator with the
  Cephes fixed rational J1 kernels used by SciPy's `j1` routine. This removes
  the variable small-series/generic-asymptotic split from the hot
  `jnjnp_zeros` seed/refinement path.
- Graveyard/artifact route tested: match the incumbent's constant rational
  kernel, eliminate branchier convergence work from a repeated evaluator, and
  stop spending effort on top-k/frontier micro-variants once the approximation
  itself is measurably slower.
- Decision: KEEP. The previous `frankenscipy-9l5oo` near-parity SciPy loss is
  now a measured SciPy win. No revert.
- Artifact:
  `tests/artifacts/perf/2026-06-20-wh8ac-jnjnp-cephes-j1/EVIDENCE.md`
- Baseline/final Rust command:
  `RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1149989 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo bench -p fsci-special --bench special_bench -- acoco_gauntlet_jnjnp_zeros --noplot`
- SciPy oracle command:
  local Python timing of `scipy.special.jnjnp_zeros(nt)` with SciPy 1.17.1.
- Benchmark evidence:

  | Workload | Baseline Rust | Final Rust | SciPy oracle | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | `jnjnp_zeros(nt=64)` | 608.21 us | 381.89 us | 463.913 us | Rust 1.59x faster internally, 1.22x faster than SciPy |
  | `jnjnp_zeros(nt=128)` | 1.1970 ms | 742.06 us | 832.786 us | Rust 1.61x faster internally, 1.12x faster than SciPy |

- SciPy win/loss/neutral for final source: `2/0/0`.
- Same-worker internal keep/loss/neutral versus the prior local J1
  series/asymptotic route: `2/0/0`.
- Correctness/conformance guards:
  - PASS: `cargo test -p fsci-special jnjnp -- --nocapture` via rch `hz1`:
    `3 passed; 0 failed`.
  - PASS: `cargo test -p fsci-special j1_matches_scipy_reference_values -- --nocapture`
    via rch `ovh-a`: `1 passed; 0 failed`.
  - PASS: `cargo test -p fsci-special complex_kve_matches_scipy -- --nocapture`
    via rch `hz1`: `1 passed; 0 failed` after replacing a pre-existing
    test-only `panic!` in touched `bessel.rs`.
  - PASS: `FSCI_REQUIRE_SCIPY_ORACLE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo test -p fsci-conformance --test diff_special_bessel_zeros -- --nocapture`:
    `1 passed; 0 failed`.
  - PASS: `cargo check -p fsci-special --all-targets` via rch `hz1` with
    existing unrelated warnings.
  - PASS: `git diff --check`.
  - PASS: changed-file `ubs` reported 0 critical issues; warning inventory
    remains non-blocking.
  - PARTIAL: rch Criterion SciPy rows were skipped because worker Python could
    not import `scipy.special`; local SciPy supplied the head-to-head rows.
  - BLOCKED: `cargo fmt -p fsci-special --check` remains blocked by
    pre-existing rustfmt drift outside this patch.
  - BLOCKED: `cargo clippy -p fsci-special --all-targets -- -D warnings`
    stopped before this patch on existing `fsci-integrate` and `fsci-linalg`
    dependency lints.
- Negative evidence: do not retry local J1 power-series/asymptotic split
  tuning for `jnjnp_zeros`. The Cephes fixed rational evaluator is both faster
  and on the SciPy oracle surface. Next work should profile below the remaining
  root-generation/frontier path, not another Bessel J1 approximation
  micro-variant.

## 2026-06-20 - frankenscipy-oi8hq - ndimage zoom order=1 no-prefilter fast path

- Agent: cod-b / MistyBirch
- Lever: for 2-D `BoundaryMode::Reflect`, `order=1` `zoom`, skip
  `prefilter_spline_coefficients` and the padded coefficient image entirely.
  Precompute row/column linear supports in the original image coordinate domain
  and run the existing fixed four-load bilinear sum directly over `input.data`.
- Graveyard/artifact route tested: remove constant-factor setup work from a hot
  small-kernel path, exploit separability, eliminate redundant copy/padding, and
  preserve the same basis weights/operation order as the generic padded sampler.
- Decision: KEEP. The previous `frankenscipy-wm14d` SciPy loss is now a
  measured SciPy win. No revert.
- Artifact:
  `tests/artifacts/perf/frankenscipy-oi8hq-zoom-order1-no-prefilter-EVIDENCE.md`
- Baseline/final Rust command:
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo bench -p fsci-ndimage --bench ndimage_bench -- zoom/2x_256/1 --noplot --sample-size 10 --measurement-time 1 --warm-up-time 1`
- SciPy oracle command:
  `python3 docs/perf_oracle_zoom.py`
- Benchmark evidence:

  | Workload | Time | Host / worker | Verdict |
  | --- | ---: | --- | --- |
  | Prepatch residual current, `zoom/2x_256/order=1` | 8.8419 ms mean (`[7.5432, 8.8419, 9.7257] ms`) | rch `hz2` | baseline |
  | Final source, `zoom/2x_256/order=1` | 1.2219 ms mean (`[1.0624, 1.2219, 1.5189] ms`) | rch `vmi1149989` | keep |
  | SciPy `ndimage.zoom(256x256, 2x, order=1)` | 4.86171 ms median | local SciPy 1.17.1 | Rust 3.98x faster |

- SciPy win/loss/neutral for final source: `1/0/0`.
- Internal baseline note: the final Rust row is `7.24x` faster than the
  prepatch rerun (`8.8419 / 1.2219`) and `6.52x` faster than the previous
  `frankenscipy-wm14d` residual row (`7.9684 / 1.2219`), but those are
  cross-worker routing-strength comparisons, not same-worker A/B claims.
- Correctness/conformance guards:
  - PASS: `rustfmt --edition 2024 --check crates/fsci-ndimage/src/lib.rs`.
  - PASS: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo check -p fsci-ndimage --all-targets`
    on rch `hz1` with existing unrelated warnings.
  - PASS: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo test -p fsci-ndimage zoom_ --lib -- --nocapture`
    on rch `ovh-a`: `6 passed; 0 failed; 235 filtered out`.
  - PASS: focused bit-equivalence guard
    `zoom_order_one_reflect_fast_path_matches_generic_sampler_bits`.
  - PASS: `FSCI_REQUIRE_SCIPY_ORACLE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo test -p fsci-conformance --test diff_ndimage_zoom -- --nocapture`:
    `1 passed; 0 failed`.
  - PASS: changed-file `ubs` exited 0; broad pre-existing
    `fsci-ndimage` warnings remained inventory only.
  - BLOCKED: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo clippy -p fsci-ndimage --all-targets -- -D warnings`
    stopped before this patch on existing `fsci-linalg` dependency lints
    (`needless_range_loop`, `needless_borrow`).
- Negative evidence: padding/copying spline coefficients for order-1 reflect
  zoom is not worth keeping on the hot path. Do not retry scheduler-only or
  padded-coefficient variants for this row without a fresh profile; route next
  to order-3 zoom, SIMD/tiled row interpolation, or another measured ndimage
  loss.

## 2026-06-20 - frankenscipy-fa62u - label mean cheap dense probe

- Agent: cod-b / MistyBirch
- Lever: replace the dense integer-label hot probe in `ndimage.mean(input,
  labels, index)` from `is_finite() + fract()` to a bounded cast plus exact
  `usize -> f64` round-trip check. This preserves NaN, fractional, negative,
  out-of-table, duplicate first-wins, and `-0.0` label behavior while removing
  expensive scalar classification from the per-element path.
- Decision: KEEP as a measured internal win. The same-host SciPy score remains
  a measured loss, so this is not release-speed parity.
- Artifact:
  `tests/artifacts/perf/frankenscipy-fa62u-label-mean-fast-probe-EVIDENCE.md`
- Correctness guards:
  - Focused dense-label semantics test via rch:
    `cargo test -p fsci-ndimage mean_dense_label_lookup_preserves_exact_label_semantics --lib -- --nocapture`
    = **1 passed / 0 failed**.
  - Full ndimage lib tests via rch:
    `cargo test -p fsci-ndimage --lib -- --nocapture` =
    **241 passed / 0 failed**.
  - Local live SciPy conformance:
    `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test diff_ndimage -- --nocapture`
    = **5 passed / 0 failed**.
  - rch conformance filter: conformance lib-side ndimage tests passed 5/0, then
    live `diff_ndimage` failed because worker `hz2` lacked Python SciPy.
  - `cargo check -p fsci-ndimage --all-targets` via rch: **PASS** with
    unrelated dependency warnings.
  - touched-file rustfmt, `git diff --check`, and UBS: **PASS**.
  - no-deps clippy `-D warnings` remains blocked by pre-existing unrelated
    `fsci-ndimage` lints outside this patch.
- Candidate A/B (`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b`
  on rch `hz2`, same binary reconstructing the previous dense `fract` probe):

  | N | K | previous dense_fract | new dense_fast | internal speedup | mismatches |
  | ---: | ---: | ---: | ---: | ---: | ---: |
  | 65536 | 512 | 353.495 us | 156.520 us | 2.26x | 0/0/0/0 |
  | 262144 | 1024 | 1.444 ms | 640.903 us | 2.25x | 0/0/0/0 |
  | 262144 | 2048 | 1.454 ms | 698.663 us | 2.08x | 0/0/0/0 |
  | 589824 | 4096 | 3.593 ms | 1.696 ms | 2.12x | 0/0/0/0 |

- Same-host final source head-to-head (transferred rch release binary
  `/data/projects/.rch-targets/frankenscipy-cod-b/release/perf_label_stats`;
  SciPy oracle `docs/perf_oracle_label_stats.py`, local SciPy 1.17.1):

  | N | K | Rust dense_fast | SciPy `ndimage.mean` | dense_fast vs SciPy |
  | ---: | ---: | ---: | ---: | ---: |
  | 65536 | 512 | 278.230 us | 0.210 ms | 1.33x slower |
  | 262144 | 1024 | 1.122 ms | 0.749 ms | 1.50x slower |
  | 262144 | 2048 | 1.186 ms | 0.751 ms | 1.58x slower |
  | 589824 | 4096 | 3.169 ms | 1.781 ms | 1.78x slower |

- SciPy win/loss/neutral for final source: `0/4/0`.
- Same-binary internal keep/loss/neutral versus previous dense `fract` probe:
  `4/0/0`.
- Negative evidence: cheapening the dense probe roughly halves the dense-label
  Rust route again, but SciPy's C path still wins every same-host measured row.
- Retry condition: do not retry another `fract()`, `is_finite()`, HashMap, or
  `Vec<Vec<f64>>` grouping variant for this workload. The next attempt must be
  a deeper reduction primitive: parallel/cache-tiled sum/count accumulation,
  vector-friendly integer-label generation, sorted/run-grouped label spans, or
  another profiled route that beats this dense-fast path in the same binary
  while preserving SciPy-observable label semantics.

## 2026-06-20 - frankenscipy-zpunl - Radau diagonal stage solve

- Agent: cod-a / MistyBirch
- Lever: specialize Radau's exactly diagonal Jacobian path. The `3n x 3n`
  collocation system splits into `n` independent `3 x 3` systems and the
  real-shift error solve becomes scalar division. The dense assembly/LU fallback
  remains for non-diagonal Jacobians.
- Decision: KEEP. This flips the previous Radau64 stiff-suite SciPy loss into a
  large measured win.
- Artifact:
  `tests/artifacts/perf/frankenscipy-zpunl-radau64-stage-lu/EVIDENCE.md`
- Correctness guards:
  - Focused Radau tests via rch:
    `cargo test -p fsci-integrate radau --lib -- --nocapture` =
    **3 passed / 0 failed**.
  - New diagonal solve guard compares the diagonal `3 x 3` block route against
    the dense `3n x 3n` LU route to `1e-12`.
  - `cargo check -p fsci-integrate --all-targets` via rch: **PASS**.
  - Touched-file rustfmt and `git diff --check`: **PASS**.
- Gate caveats:
  - `cargo clippy -p fsci-integrate --all-targets -- -D warnings` is still
    blocked by pre-existing non-Radau lint debt after the touched helper issue
    was fixed: `api.rs`, `rk.rs`, and `quad.rs`.
  - `cargo test -p fsci-conformance --test e2e_ivp -- --nocapture` is blocked
    before IVP tests by unrelated `fsci-cluster` compile errors for missing
    `fsci_linalg::{randomized_svd, randomized_eigh}` symbols plus one ambiguous
    float.
- Same-worker A/B on `ovh-a`, repeats=20:

  | Route | per-call | nfev | njev | nlu | checksum |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | baseline dense Radau stage LU | 70047.428100 us | 20900 | 20 | 5600 | 3.234169482298e4 |
  | diagonal stage/scalar solve | 1124.530700 us | 20900 | 20 | 5600 | 3.234169482298e4 |

- Final-source head-to-head vs local SciPy 1.17.1:

  | Workload | Rust final | SciPy oracle | Ratio | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | `radau-stiff64`, repeats=20 | 1687.783900 us | 36545.873049 us | 21.65x faster | SciPy win |
  | `bdf-stiff64` | 2306.348100 us | 25448.461401 us | 11.03x faster | SciPy win |
  | `bdf-stiff128` | 12041.547000 us | 29874.872195 us | 2.48x faster | SciPy win |
  | `radau-stiff32` | 591.275600 us | 33708.498604 us | 57.01x faster | SciPy win |
  | `radau-stiff64`, stiff-suite repeats=10 | 1306.515700 us | 36488.462600 us | 27.93x faster | SciPy win |

- SciPy win/loss/neutral for final stiff suite: `4/0/0`.
- Negative evidence: the previous Radau scaled-RMS stream candidate remains a
  no-retry rejection. The profitable target was dense stage linear algebra for
  structured Jacobians, not norm materialization.
- Retry condition: future Radau attempts should move to banded/non-diagonal
  structure detection, step-to-step LU reuse, or analytic Jacobian plumbing. Do
  not spend another pass on collected-vs-streamed norm micro-variants unless a
  fresh profile shows norm work back on the hot path.

## 2026-06-20 - frankenscipy-klb7o - label mean dense lookup

- Agent: cod-a / MistyBirch
- Lever: specialize `ndimage.mean(input, labels, index)` for compact
  non-negative integer labels by building a dense `Vec<usize>` label-to-first
  position table. This removes the per-element `HashMap` probe from the prior
  flat sum/count path while preserving the HashMap fallback for huge/sparse
  indexes and preserving first-position duplicate-index, signed-zero, NaN, and
  fractional-label semantics.
- Decision: KEEP as a measured internal win. The route is still a measured
  SciPy loss, so this is not release-speed parity.
- Artifact:
  `tests/artifacts/perf/2026-06-20-label-stats-dense-mean/EVIDENCE.md`
- Correctness guards:
  - `cargo test -p fsci-ndimage --lib -- --nocapture` via rch:
    **241 passed / 0 failed**.
  - `cargo test -p fsci-conformance ndimage -- --nocapture` via rch:
    conformance lib-side ndimage filter **5 passed / 0 failed**; rch live
    `diff_ndimage` integration failed only because worker `hz2` lacked SciPy
    while `FSCI_REQUIRE_SCIPY_ORACLE=1`.
  - `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test
    diff_ndimage -- --nocapture` locally with SciPy 1.17.1:
    **5 passed / 0 failed**.
  - Same-binary `perf_label_stats` asserted **0 bit-mismatches** versus the
    old linear-scan route, previous bucketed O(N+K) route, and previous flat
    HashMap route on every row.
- Benchmark (`/data/projects/.rch-targets/frankenscipy-cod-a/release/perf_label_stats`
  for same-host Rust A/B; SciPy oracle `docs/perf_oracle_label_stats.py`,
  local SciPy 1.17.1):

  | N | K | previous flat HashMap | dense mean | internal speedup | SciPy `ndimage.mean` | dense vs SciPy |
  | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 65536 | 512 | 625.594 us | 301.407 us | 2.08x | 158 us | 1.91x slower |
  | 262144 | 1024 | 2.652 ms | 1.263 ms | 2.10x | 538 us | 2.35x slower |
  | 262144 | 2048 | 2.798 ms | 1.383 ms | 2.02x | 550 us | 2.51x slower |
  | 589824 | 4096 | 6.772 ms | 3.351 ms | 2.02x | 1.271 ms | 2.64x slower |

- SciPy win/loss/neutral for final source: `0/4/0`.
- Same-binary internal keep/loss/neutral versus prior flat HashMap route:
  `4/0/0`.
- Negative evidence: dense integer label lookup halves the remaining Rust mean
  cost on compact label workloads and narrows the prior 3.7-4.7x SciPy gap to
  1.9-2.6x slower, but SciPy's compiled C implementation still wins every
  measured row.
- Retry condition: do not retry another per-element `HashMap`, `Vec<Vec<f64>>`
  grouping, or scalar-only dense lookup variant for this workload. Next attempts
  need sorted-label remapping, fused integer-label generation from `label()`,
  SIMD accumulation over contiguous label spans, or cache-tiled sum/count
  reductions that beat this dense route in the same binary while preserving
  SciPy-observable label semantics.

## 2026-06-20 - frankenscipy-8l8r1.125 - label mean flat accumulator

- Agent: cod-a / MistyBirch
- Lever: specialize `ndimage.mean(input, labels, index)` so it streams once into
  flat `sum` and `count` arrays instead of materializing one `Vec<f64>` bucket
  per requested label before dividing. The duplicate-index and exact label
  equality semantics remain the same: the shared canonical label key preserves
  `+0.0 == -0.0`, `NaN` bit identity, and first-position wins for duplicate
  `index` entries.
- Decision: KEEP as a measured internal win. The route is still a measured
  SciPy loss, so this is not release-speed parity.
- Artifact:
  `tests/artifacts/perf/2026-06-20-label-stats-flat-mean/EVIDENCE.md`
- Correctness guards (GREEN):
  - `cargo test -p fsci-ndimage measurement_reduction_wrappers -- --nocapture`
    via rch: **2 passed / 0 failed**.
  - `cargo test -p fsci-ndimage --lib -- --nocapture` via rch:
    **240 passed / 0 failed**.
  - Same-binary `perf_label_stats` asserted **0 bit-mismatches** versus both the
    old linear-scan route and the previous bucketed O(N+K) route on every row.
- Benchmark (`/data/projects/.rch-targets/frankenscipy-cod-a/release/perf_label_stats`
  for same-host Rust A/B; SciPy oracle `docs/perf_oracle_label_stats.py`,
  local SciPy 1.17.1):

  | N | K | previous bucketed O(N+K) | flat mean | internal speedup | SciPy `ndimage.mean` | flat vs SciPy |
  | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 65536 | 512 | 1.047 ms | 590.978 us | 1.77x | 159 us | 3.72x slower |
  | 262144 | 1024 | 3.695 ms | 2.568 ms | 1.44x | 622 us | 4.13x slower |
  | 262144 | 2048 | 4.140 ms | 2.713 ms | 1.53x | 581 us | 4.67x slower |
  | 589824 | 4096 | 11.760 ms | 6.951 ms | 1.69x | 1.688 ms | 4.12x slower |

- SciPy win/loss/neutral for final source: `0/4/0`.
- Same-binary internal keep/loss/neutral versus prior bucketed route: `4/0/0`.
- Negative evidence: removing bucket materialization for `mean` cuts another
  30-44% from the Rust path, but the compiled SciPy implementation is still
  3.7-4.7x faster. The remaining cost is now per-element HashMap lookup plus
  scalar accumulation overhead, not group allocation.
- Retry condition: do not retry another `Vec<Vec<f64>>` grouping variant for
  `mean`. Next attempts need a materially different constant-factor lever:
  dense lookup tables for small contiguous labels, sorted-label remapping,
  specialized integer-label paths, SIMD/cache-tiled accumulation, or another
  profiled route that beats the flat accumulator in the same binary while
  preserving SciPy-observable label semantics.

## 2026-06-19 - frankenscipy-label-stats - label-indexed measurement O(N*K)->O(N+K)

- Agent: cc / MistyBirch
- Lever: the shared core `measurement_label_groups` / `measurement_label_value_positions`
  (used by `sum_labels`, `mean`, `variance`, `standard_deviation`, `minimum`,
  `maximum`, `median`, `labeled_comprehension`) bucketed each of the N input
  elements with an O(K) linear `index.iter().position(...)` scan over the K
  requested labels — O(N*K). Replace with a one-time `HashMap<label_bits, pos>`
  built from `index` (O(K)), then an O(1) lookup per element — O(N+K).
  Canonical ±0.0 key + `or_insert` (keep first) make it byte-identical to the
  old `position` first-match semantics.
- Decision: KEEP. Byte-identical (the change only reorders the same grouping),
  measured large self-speedup that grows with K. Still a SciPy loss in absolute
  terms (see below).
- Correctness guards (GREEN): full `cargo test -p fsci-ndimage --lib` = **240
  passed / 0 failed** (covers `sum_labels`, `mean`, `variance`,
  `labeled_comprehension`, scipy-reference fixtures); the `perf_label_stats`
  bin asserts **0 bit-mismatches** vs a verbatim old linear-scan grouping on
  every row.
- Benchmark (`cargo run --release -p fsci-ndimage --bin perf_label_stats`,
  same-binary old-vs-new A/B; SciPy oracle `docs/perf_oracle_label_stats.py`
  `ndimage.mean`, both same local host, scipy 1.17.1):

  | N | K | old O(N*K) | new O(N+K) | self-speedup | SciPy `ndimage.mean` | new vs SciPy |
  | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 65536 | 512 | 16.900 ms | 1.024 ms | 16.5x | 0.205 ms | 5.0x slower |
  | 262144 | 1024 | 105.53 ms | 3.644 ms | 29.0x | 0.541 ms | 6.7x slower |
  | 262144 | 2048 | 212.80 ms | 3.841 ms | 55.4x | 0.556 ms | 6.9x slower |
  | 589824 | 4096 | 932.40 ms | 10.516 ms | 88.7x | 1.274 ms | 8.3x slower |

- Negative evidence: even after closing the O(N*K) blowup, fsci is still
  5-8x slower than SciPy's compiled C. The remaining cost is the
  `Vec<Vec<f64>>` per-label group materialization (every value pushed into a
  per-label bucket) plus the per-element HashMap hashing; SciPy accumulates
  sum/count per label in a flat array without materializing groups. This is a
  measured SciPy loss recorded as such.
- Retry condition: do not revert to the linear `position` scan. The next lever
  is a reduction-specific path that accumulates sum/count (mean/sum/variance)
  or running min/max directly into a flat per-position array WITHOUT
  materializing `Vec<Vec<f64>>` — measured same-host vs SciPy `ndimage.mean`,
  keeping byte-identity (median/comprehension still need the materialized
  values). A dense lookup table (Vec instead of HashMap) when labels are small
  and contiguous would also cut the per-element constant.

## 2026-06-19 - frankenscipy-edt-indices - distance_transform_edt feature transform

- Agent: cc / MistyBirch
- Lever: replace the `distance_transform_edt(return_indices=True)` brute-force
  nearest-background scan (O(foreground · background) via `nearest_edt_background`)
  with the exact separable Euclidean **feature transform** O(N · ndim). The
  existing Felzenszwalb 1-D pass (`edt_1d_squared`) already finds the winning
  vertex per output position; I expose it (`w` out-param) and propagate a
  parallel nearest-background flat-index array (`feat`) through the separable
  passes in `edt_squared_felzenszwalb_with_indices`. Background-free and
  non-finite-sampling inputs stay on the brute-force/sentinel path.
- Decision: KEEP. A measured, byte-verified complexity win that closes the
  largest single gap from the ndimage recon. Resolves bead
  `frankenscipy-edt-indices-feature-transform-xsudx`.
- Correctness guards (all GREEN):
  - `edt_feature_transform_distances_byte_identical_and_indices_valid` (new
    property test): over multi-background/tie 2-D/3-D grids with non-unit
    sampling, squared distances are byte-identical to the shipped distance-only
    fast path AND every returned index is a genuine nearest background (its
    squared distance equals the exact EDT at that cell).
  - `perf_edt` isomorphism harness: **0 mismatches / 10876 cells**.
  - All 18 `distance_transform` unit tests pass, including the scipy-pinned
    `..._indices_match_scipy` fixtures (single-background → unique nearest → no
    ties → unchanged output).
  - Live SciPy conformance `diff_ndimage_distance_transform_edt` PASS (local
    scipy 1.17.1, `FSCI_REQUIRE_SCIPY_ORACLE=1`).
- Benchmark (`cargo run --release -p fsci-ndimage --bin perf_edt`, same-binary
  brute-vs-feature A/B; SciPy oracle `docs/perf_oracle_edt_indices.py`). All
  rows below are **same local host** (scipy 1.17.1), so both the self-speedup
  and the vs-SciPy ratio are directly comparable:

  | Image | brute O(f·b) | feature transform | self-speedup | SciPy `return_indices` | fsci vs SciPy |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 64x64 | 17.799 ms | 295.9 us | 60.2x | 169.95 us | 1.74x slower |
  | 128x128 | 334.07 ms | 1.353 ms | 246.9x | 769.50 us | 1.76x slower |
  | 192x192 | 1.527 s | 3.581 ms | 426.5x | 2.263 ms | 1.58x slower |
  | 256x256 | 4.764 s | 5.705 ms | 835.1x | 4.108 ms | 1.39x slower |

- Negative evidence: fsci remains ~1.4-1.8x slower than SciPy's compiled C
  feature transform (the gap narrows with size: 1.39x at 256x256), so in
  absolute terms this is still a measured SciPy loss — but the catastrophic
  O(f·b) brute force is gone (60-835x self-speedup, growing with N).
- Retry condition: do not revert to the brute-force indices scan. Further gains
  must tighten the feature-transform constant factor (inner separable line loop
  / index propagation), measured same-host vs SciPy, without distance
  byte-identity or nearest-background-validity drift.

## 2026-06-19 - frankenscipy-8l8r1.124 - jnjnp_zeros top-k select

- Agent: cod-a / MistyBirch
- Lever tested: replace full candidate sorting inside the cutoff-driven
  `jnjnp_zeros` generator with `select_nth_unstable_by` plus sorting only the
  retained prefix.
- Decision: REJECT and revert. The final source remains the full-sort
  cutoff-driven generator from `frankenscipy-8l8r1.123`.
- Artifact:
  `tests/artifacts/perf/2026-06-19-8l8r1-jnjnp-topk-select/EVIDENCE.md`
- Baseline command:
  `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-b CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo bench -p fsci-special --bench special_bench -- acoco_gauntlet_jnjnp_zeros --noplot`
- Same-binary probe command:
  `RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo test -p fsci-special jnjnp_topk_select_perf_probe --release -- --ignored --nocapture`
- SciPy oracle command: local Python timing of
  `scipy.special.jnjnp_zeros(nt)` because RCH workers could not import
  `scipy.special`.

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| `nt=64` restored full-sort current, RCH `vmi1153651` | 1.5407 ms | 3.65x slower than SciPy | final-source SciPy loss |
| `nt=128` restored full-sort current, RCH `vmi1153651` | 3.5199 ms | 4.54x slower than SciPy | final-source SciPy loss |
| SciPy `jnjnp_zeros(nt=64)`, local SciPy 1.17.1 | 421.59 us | 1.00x oracle | reference |
| SciPy `jnjnp_zeros(nt=128)`, local SciPy 1.17.1 | 774.75 us | 1.00x oracle | reference |
| Same-binary top-k probe `nt=64`, longer RCH `hz1` run | 0.911730 ms vs 0.928939 ms full-sort | 1.019x | neutral |
| Same-binary top-k probe `nt=128`, longer RCH `hz1` run | 1.715855 ms vs 1.737776 ms full-sort | 1.013x | neutral |

- SciPy win/loss/neutral for final source: `0/2/0`.
- Candidate internal keep/loss/neutral: `0/0/2`; the first short same-binary
  probe showed one 1.13x `nt=64` win and one neutral/loss `nt=128` row, but the
  longer probe collapsed to near-noise on both rows.
- Correctness guard: the ignored release probe asserted bit-identical
  `(zo, n, m, t)` outputs before timing. Focused `fsci-special jnjnp` tests
  passed via RCH during the probe.
- Negative evidence: candidate ordering/selection is not the next bottleneck.
  Do not retry top-k partitioning, partial sorting, or output-prefix sorting
  without a fresh profile showing candidate sorting dominates and a same-binary
  gate above 10 percent on both `nt=64` and `nt=128`.
- Retry condition: route deeper to lower-cost Bessel/root generation,
  recurrence constants, or a SciPy-style compiled root-polishing strategy.

## 2026-06-19 - frankenscipy-wm14d - ndimage zoom order=1 residual fast path

- Agent: cod-b / MistyBirch
- Lever: add a narrow 2D Reflect/order=1 `zoom` fast path that precomputes
  separable row and column linear supports once, then evaluates each output
  pixel with a fixed four-load bilinear sum over the existing padded spline
  coefficient image.
- Graveyard/artifact route tested: separable support caching, fixed-kernel
  output-sensitive interpolation, cache-friendly row/column support reuse, and
  removal of the generic recursive sampler from the hot pixel loop.
- Decision: KEEP as a measured internal win, route residual SciPy loss deeper.
  No revert.
- Artifact:
  `tests/artifacts/perf/frankenscipy-wm14d-residual-20260619/EVIDENCE.md`
- Baseline/candidate command:
  `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-b CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo bench -p fsci-ndimage --bench ndimage_bench -- zoom/2x_256/1 --noplot --sample-size 10 --measurement-time 1 --warm-up-time 1`
- SciPy oracle command:
  `python3 docs/perf_oracle_zoom.py`
- Same-worker internal A/B on rch worker `ovh-b`:

  | Workload | Baseline current mean | Candidate current mean | Candidate/baseline | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | `zoom/2x_256/order=1` | 34.034 ms | 7.9684 ms | 0.234x time, 4.27x faster | keep |

- Local SciPy oracle (`scipy.ndimage.zoom`, Python 3.13.7, NumPy 2.4.3,
  SciPy 1.17.1):

  | Workload | Candidate Rust mean | SciPy median | Candidate/SciPy | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | `ndimage.zoom(256x256, 2x, order=1)` | 7.9684 ms | 3.88937 ms | 2.05x slower | residual loss |

- SciPy win/loss/neutral: `0/1/0`.
- Same-worker internal keep/loss/neutral: `1/0/0`.
- Correctness/conformance guards:
  - PASS: `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-b CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo test -p fsci-ndimage zoom_order_one_reflect_fast_path_matches_generic_sampler_bits -- --nocapture`
    (`1 passed; 0 failed; 238 filtered out`).
  - PASS: `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-b CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo test -p fsci-ndimage zoom_ -- --nocapture`
    (`6 passed; 0 failed; 233 filtered out`; metamorphic
    `mr_zoom_by_one_is_identity` also passed).
  - PASS: `FSCI_REQUIRE_SCIPY_ORACLE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo test -p fsci-conformance --test diff_ndimage_zoom -- --nocapture`
    (`1 passed; 0 failed` against live local SciPy).
  - PASS: `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-b CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo check -p fsci-ndimage --all-targets`.
  - PASS: `git diff --check`.
  - PASS: `ubs` on the touched Rust file, scorecard/ledger docs, and this
    evidence bundle found no critical issues.
  - BLOCKED: `cargo fmt -p fsci-ndimage --check` and direct
    Rust 2024 rustfmt remain blocked by pre-existing formatting drift outside
    this patch.
  - BLOCKED: strict clippy remains blocked by existing `fsci-linalg`
    dependency lints and existing `fsci-ndimage` library lint debt unrelated
    to this fast path.
- Rejected sub-variant: serial fill inside the new fixed-order fast path
  measured 9.6976 ms versus 7.9684 ms for the kept parallel fill, a 1.22x
  regression. Reverted to `fill_pixels_parallel`.
- Negative evidence: the direct bilinear fast path is worthwhile but not
  sufficient; Rust remains 2.05x slower than SciPy on the local oracle. Do not
  repeat serial-fill or other scheduler-only probes without a fresh profile
  showing parallel overhead dominates after this patch.
- Retry condition: route deeper to a proof-backed no-prefilter order=1 path,
  code-generated/tiled geometric order=1 kernels, or SIMD contiguous row
  interpolation while preserving the existing Reflect boundary and tolerance
  contracts.

## 2026-06-19 - frankenscipy-96n2y - jnjnp_zeros tighter frontier seed

- Agent: cod-b / MistyBirch
- Lever: tighten the adaptive `jnjnp_zeros` frontier seed and expand the serial
  root count (`per`) and order envelope (`n_max`) independently based on the
  existing exact frontier proof. The prior seed overgenerated the gauntlet
  cases: `nt=128` only needs max serial `m=7` and max order `n=19`, while the
  old first pass evaluated `per=16` through `n_max=30`.
- Graveyard/artifact route tested: output-sensitive enumeration, asymmetric
  frontier growth, constant-factor candidate pruning before root polishing.
- Decision: KEEP as a measured internal win, route residual SciPy loss deeper.
  No revert.
- Artifact:
  `tests/artifacts/perf/frankenscipy-cod-b-jnjnp-stream/EVIDENCE.md`
- Baseline/candidate command:
  `RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo bench -p fsci-special --bench special_bench -- acoco_gauntlet_jnjnp_zeros --noplot`
- SciPy oracle command:
  local Python timing of `scipy.special.jnjnp_zeros` because rch worker `hz1`
  could not import `scipy.special`.
- Same-worker internal A/B on rch worker `hz1`:

  | Workload | Baseline current mean | Candidate current mean | Candidate/baseline | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | `jnjnp_zeros(nt=64)` | 4.7127 ms | 2.2230 ms | 0.472x time, 2.12x faster | keep |
  | `jnjnp_zeros(nt=128)` | 8.5181 ms | 6.1605 ms | 0.723x time, 1.38x faster | keep |

- Local SciPy oracle (`scipy.special.jnjnp_zeros`, Python 3.13.7, NumPy 2.4.3,
  SciPy 1.17.1):

  | Workload | Candidate Rust mean | SciPy median | Candidate/SciPy | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | `jnjnp_zeros(nt=64)` | 2.2230 ms | 424.10 us | 5.24x slower | residual loss |
  | `jnjnp_zeros(nt=128)` | 6.1605 ms | 799.97 us | 7.70x slower | residual loss |

- SciPy win/loss/neutral: `0/2/0`.
- Same-worker internal keep/loss/neutral: `2/0/0`.
- Correctness/conformance guards:
  - PASS: `RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo test -p fsci-special jnjnp -- --nocapture`
    (`3 passed; 0 failed; 1109 filtered out`).
  - PASS: `RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo check -p fsci-special --all-targets`.
- Negative evidence: frontier seed retuning is worthwhile but not enough; Rust
  remains 5.24-7.70x slower than SciPy on the local oracle. Do not repeat
  small envelope constant tuning without a fresh profile showing candidate
  overgeneration still dominates.
- Retry condition: route deeper to lower-constant Bessel root generation,
  cached cross-order recurrences, or a true heap-streamed global zero
  enumerator that preserves the exact SciPy output ordering.

## 2026-06-19 - frankenscipy-8l8r1.123 - jnjnp_zeros cutoff-driven generator

- Agent: cod-b / MistyBirch
- Lever: add a cutoff-driven triangular generator for `jnjnp_zeros(nt >= 16)`.
  Instead of growing a rectangular order-by-serial root grid, it estimates a
  global retained-zero cutoff, emits only `J_n` and `J_n'` roots below that
  cutoff, and accepts only when both omitted-root frontiers are past the actual
  retained cutoff. The old rectangular frontier stays as fallback.
- Graveyard/artifact route tested: output-sensitive generation, monotone
  frontier certificate, candidate-count collapse before root polishing, and
  rollback of a public-helper refactor after a non-shipping comparator row
  looked suspicious.
- Decision: KEEP as a measured internal win, route residual SciPy loss deeper.
  No revert.
- Artifact:
  `tests/artifacts/perf/frankenscipy-cod-b-jnjnp-cutoff/EVIDENCE.md`
- Baseline/candidate command:
  `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-b CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo bench -p fsci-special --bench special_bench -- acoco_gauntlet_jnjnp_zeros --noplot`
- SciPy oracle command:
  local Python timing of `scipy.special.jnjnp_zeros` because rch worker `ovh-b`
  could not import `scipy.special`.
- Same-worker internal A/B on rch worker `ovh-b`:

  | Workload | Baseline current mean | Candidate current mean | Candidate/baseline | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | `jnjnp_zeros(nt=64)` | 3.6486 ms | 1.5856 ms | 0.435x time, 2.30x faster | keep |
  | `jnjnp_zeros(nt=128)` | 6.6226 ms | 2.9035 ms | 0.438x time, 2.28x faster | keep |

- Local SciPy oracle (`scipy.special.jnjnp_zeros`, Python 3.13.7, NumPy 2.4.3,
  SciPy 1.17.1):

  | Workload | Candidate Rust mean | SciPy median | Candidate/SciPy | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | `jnjnp_zeros(nt=64)` | 1.5856 ms | 427.47 us | 3.71x slower | residual loss |
  | `jnjnp_zeros(nt=128)` | 2.9035 ms | 789.23 us | 3.68x slower | residual loss |

- SciPy win/loss/neutral: `0/2/0`.
- Same-worker internal keep/loss/neutral: `2/0/0`.
- Correctness/conformance guards:
  - PASS: `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-b CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo test -p fsci-special jnjnp -- --nocapture`
    (`3 passed; 0 failed; 1109 filtered out`).
  - PASS: `FSCI_REQUIRE_SCIPY_ORACLE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo test -p fsci-conformance --test diff_special_bessel_zeros -- --nocapture`
    (`1 passed; 0 failed`).
  - PASS: `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-b CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo check -p fsci-special --all-targets`.
  - BLOCKED: scoped clippy `-D warnings` stops on existing `fsci-integrate`
    and `fsci-linalg` dependency lints before reaching this patch.
  - BLOCKED: broad rustfmt and touched-file rustfmt remain blocked by
    pre-existing formatting drift outside this patch.
- Negative evidence: output-sensitive cutoff generation is worthwhile but not
  sufficient; Rust still loses to SciPy by about 3.7x. Do not repeat envelope
  constant tuning without a fresh profile proving candidate generation still
  dominates.
- Retry condition: route deeper to lower-constant Bessel/derivative evaluation,
  cross-order recurrence caches, or a Specfun-style global zero enumerator or
  code-generated root kernel that preserves the exact SciPy ordering contract.

## 2026-06-19 - frankenscipy-8l8r1.117 - sparse random rounded-empty path

- Agent: cod-b / MistyBirch
- Lever: compute `round(density * rows * cols)` before sampling in
  `fsci_sparse::random`; if the rounded cardinality is zero, return an empty COO
  after overflow validation. This was already implemented by `f037b1da`; this
  entry is the measured closeout.
- Graveyard/artifact route tested: output-sensitive cardinality short-circuit,
  constant-time empty artifact construction, and zero scan of impossible output.
- Decision: KEEP. No revert.
- Artifact: `tests/artifacts/perf/frankenscipy-8l8r1.117/EVIDENCE.md`
- Rust command:
  `RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo bench -p fsci-sparse --bench sparse_bench -- sparse_random_tiny_density --sample-size 10 --measurement-time 1 --warm-up-time 1`
- SciPy oracle command:
  `RCH_WORKER=hz1 rch exec -- python3 -c '... scipy.sparse.random(..., format="coo", rng=reused_rng) ...'`
- Same-worker head-to-head on `hz1`:

  | Workload | Rust Criterion point | SciPy median | SciPy/Rust | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | `random(1e9 x 1e9, density=1e-19)` | 39.405 ns | 46,453 ns | 1,179x faster | win |
  | `random(2e9 x 2e9, density=1e-20)` | 39.408 ns | 55,310 ns | 1,403x faster | win |

- SciPy win/loss/neutral: `2/0/0`.
- Correctness/conformance guards:
  - PASS: `RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo test -p fsci-sparse random_ -- --nocapture`
    (`10 passed; 0 failed; 338 filtered out`).
- Negative evidence: do not retry dense Bernoulli scanning for rounded-empty
  sparse random. The current cardinality-first path is both SciPy-compatible and
  orders of magnitude faster.
- Retry condition: only revisit this specific regime if SciPy changes the
  rounded-cardinality contract or a future profile shows non-empty sparse random
  collision handling, direct CSR/CSC construction, or data generation dominates.

## 2026-06-19 - frankenscipy-x9ckc - jnjnp_zeros root-cost refinement

- Agent: cod-b / MistyBirch
- Lever: reduce the residual per-root cost after `frankenscipy-01lxz` by
  replacing generic strict-mode `J_n`/`J_n'` zero evaluation with direct
  integer-order kernels plus bracket-safe secant/Newton refinement. The Newton
  path uses Bessel derivative identities and falls back to bracketed refinement
  when the correction is not finite or would leave the current bracket.
- Graveyard/artifact route tested: lower-constant scalar kernel, guarded
  Newton/root-polishing, branch-constrained fast path with deterministic
  bracket fallback.
- Decision: KEEP as a measured internal win, route residual SciPy loss deeper.
  No revert.
- Artifact:
  `tests/artifacts/perf/frankenscipy-x9ckc-jnjnp-rootcost/EVIDENCE.md`
- Baseline/candidate command:
  `RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo bench -p fsci-special --bench special_bench -- acoco_gauntlet_jnjnp_zeros --noplot`
- SciPy oracle command:
  local Python timing of `scipy.special.jnjnp_zeros` because rch worker `hz1`
  could not import `scipy.special`.
- Same-worker internal A/B on rch worker `hz1`:

  | Workload | Baseline current mean | Candidate current mean | Candidate/baseline | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | `jnjnp_zeros(nt=64)` | 5.4622 ms | 4.6666 ms | 0.854x time, 1.17x faster | keep |
  | `jnjnp_zeros(nt=128)` | 9.9958 ms | 8.3620 ms | 0.837x time, 1.20x faster | keep |

- Local SciPy oracle (`scipy.special.jnjnp_zeros`, SciPy 1.17.1):

  | Workload | Candidate Rust mean | SciPy mean | Candidate/SciPy | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | `jnjnp_zeros(nt=64)` | 4.6666 ms | 439.49 us | 10.62x slower | residual loss |
  | `jnjnp_zeros(nt=128)` | 8.3620 ms | 787.18 us | 10.62x slower | residual loss |

- SciPy win/loss/neutral: `0/2/0`.
- Same-worker internal keep/loss/neutral: `2/0/0`.
- Correctness/conformance guards:
  - PASS: `RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo test -p fsci-special jnjnp -- --nocapture`.
- Negative evidence: guarded root polishing and direct integer-order derivative
  evaluation are worthwhile but not enough; Rust is still roughly 10.6x slower
  than SciPy on the local oracle. Do not repeat the same secant/Newton guard
  tuning without a fresh profile showing the same loop is still dominant.
- Retry condition: route deeper to a SciPy-style global zero enumerator,
  batched recurrence/evaluation cache across neighboring orders, or a generated
  lower-constant integer-order Bessel kernel that preserves the existing
  SciPy-order output contract.

## 2026-06-19 - frankenscipy-01lxz - jnjnp_zeros output frontier

- Agent: cod-b / MistyBirch
- Lever: replace `jnjnp_zeros` fixed `nt + 2` by `nt + 2` candidate generation
  with an output-sensitive monotone frontier. The new route starts near
  `sqrt(nt)`, sorts the candidate subset, and expands only when the retained
  cutoff is not below both the generated serial-tail frontier and the first
  omitted-order frontier.
- Graveyard/artifact route tested: output-sensitive enumeration, monotone
  frontier certification, and constant-factor collapse before lower-level SIMD
  work.
- Decision: KEEP as a major internal win, route residual SciPy loss deeper. No
  revert.
- Artifact:
  `tests/artifacts/perf/frankenscipy-cod-b-jnp-frontier/EVIDENCE.md`
- Baseline/candidate command:
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo bench -p fsci-special --bench special_bench -- acoco_gauntlet_jnjnp_zeros --noplot`
- Candidate same-worker command:
  `RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo bench -p fsci-special --bench special_bench -- acoco_gauntlet_jnjnp_zeros --noplot`
- SciPy oracle command:
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo bench -p fsci-special --bench special_bench -- acoco_gauntlet_jnjnp_zeros --noplot`
  because rch worker `hz1` could not import `scipy.special`.
- Same-worker internal A/B on rch worker `hz1`:

  | Workload | Baseline current mean | Candidate current mean | Candidate/baseline | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | `jnjnp_zeros(nt=64)` | 97.861 ms | 5.4922 ms | 0.0561x time, 17.82x faster | keep |
  | `jnjnp_zeros(nt=128)` | 513.89 ms | 10.121 ms | 0.0197x time, 50.77x faster | keep |

- Local SciPy oracle (`scipy.special.jnjnp_zeros`, SciPy available on the local
  host):

  | Workload | Candidate Rust mean | SciPy mean | Candidate/SciPy | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | `jnjnp_zeros(nt=64)` | 4.3372 ms | 486.57 us | 8.91x slower | residual loss |
  | `jnjnp_zeros(nt=128)` | 7.5415 ms | 792.81 us | 9.51x slower | residual loss |

- SciPy win/loss/neutral: `0/2/0`.
- Same-worker internal keep/loss/neutral: `2/0/0`.
- Correctness/conformance guards:
  - PASS: `rch exec -- cargo test -p fsci-special jnjnp_adaptive_envelope_matches_oversized_reference --lib -- --nocapture`
    on worker `ovh-a`.
  - PASS: `rch exec -- cargo test -p fsci-special jnyn_and_jnjnp_zeros_match_scipy --lib -- --nocapture`
    on worker `hz1`.
  - PASS: `rch exec -- cargo check -p fsci-special --all-targets` on worker
    `ovh-b`.
  - BLOCKED: `rch exec -- cargo clippy -p fsci-special --all-targets -- -D warnings`
    stopped in dependency crates `fsci-integrate` (`too_many_arguments`) and
    `fsci-linalg` (`needless_range_loop`, `needless_borrow`) before reaching
    this patch.
  - PARTIAL: broad `cargo fmt --check` and touched-file `rustfmt --check`
    report pre-existing rustfmt drift outside this patch; no broad formatting
    churn was applied.
- Negative evidence: the frontier removes the known over-generation gap but
  still leaves Rust roughly 9x slower than SciPy. Do not retry fixed envelope
  tuning, duplicate derivative-zero bracketing, or other candidate-count-only
  tweaks on this workload.
- Retry condition: the next credible route is deeper per-root cost reduction:
  SciPy-style global zero enumeration, lower-constant Bessel/derivative
  evaluation, or a batched bracketing/root-polishing primitive that preserves the
  existing SciPy-order output contract.

## 2026-06-19 - frankenscipy-acdq2 - ndimage gaussian_filter line-walk route

- Agent: cod-a / MistyBirch
- Lever: force `gaussian_filter1d_axis` onto the 1-D slab line walker for every
  axis, then add outermost-axis row-splitting and direct interior tap indexing
  to avoid `boundary_index_1d` on non-border rows.
- Graveyard/artifact route tested: cache-aware line walking, branch removal in
  the interior stencil, and parallel row chunks for the low-outer-slab case.
- Decision: REJECT AND REVERT. No source change shipped.
- Artifact:
  `tests/artifacts/perf/2026-06-19-ndimage-gaussian-linewalk-reject/EVIDENCE.md`
- Candidate command:
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo bench -p fsci-ndimage --bench ndimage_bench -- correlate_gaussian/gaussian_sigma2/256 --noplot`
- Candidate result, rch worker `vmi1152480`:

  | Workload | Candidate mean | Candidate interval | Verdict |
  | --- | ---: | ---: | --- |
  | `correlate_gaussian/gaussian_sigma2/256` | 4.2236 ms | [4.0337 ms, 4.4246 ms] | loss |

- Clean current result on `origin/main` commit `96a37a83`, rch worker `ovh-a`:

  | Workload | Current mean | Current interval | Candidate/current |
  | --- | ---: | ---: | ---: |
  | `correlate_gaussian/gaussian_sigma2/256` | 2.4792 ms | [2.4545 ms, 2.5044 ms] | 1.704x slower |

- Existing ledger current result before this attempt was also faster than the
  candidate: 3.238 ms. That makes the route a rejection even with cross-worker
  noise.
- Local SciPy oracle (`python3 docs/perf_oracle_ndimage.py`, SciPy 1.17.1):
  `gaussian_filter sigma=2 256x256` p50 was 1.47107 ms. The rejected candidate
  is 2.87x slower than that oracle; the clean current rch row is still slower
  and remains a gap.
- Correctness/conformance guard: no optimized source was kept, so the
  conformance surface is unchanged from `origin/main`.
- Retry condition: do not retry fallback removal, outermost-axis row-splitting,
  or scalar direct-index boundary peeling for this workload without a fresh
  profile. The next plausible route is a lower-level contiguous-span/SIMD dot
  kernel or a cache-tiled separable scratch/transpose path that preserves the
  existing gaussian tolerance contract.

## 2026-06-19 - frankenscipy-nm8ex - spatial pdist dim-4 fast path

- Agent: cod-a / MistyBirch
- Lever: specialize `pdist` for the measured 4-D Euclidean and Cosine gap by
  bypassing per-pair metric dispatch, generic slice reductions, and SIMD-tail
  setup, then force the now-cheap dim-4 path through the serial gate to avoid
  thread-spawn overhead at n=256/512. The kept path uses direct dim-4
  dot/squared-norm arithmetic while leaving the generic pair-balanced row split
  available for other metrics and shapes.
- Graveyard/artifact route tested: cache/constant-factor collapse for the tight
  O(n^2) kernel, branch removal inside the metric loop, and shape-specific
  codegen without unsafe code.
- Decision: KEEP as an internal win, route residual SciPy loss deeper. No
  revert.
- Artifact:
  `tests/artifacts/perf/2026-06-19-nm8ex-pdist-dim4/EVIDENCE.md`
- Baseline/candidate command:
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo bench -p fsci-spatial --bench spatial_bench -- pdist --noplot`
- SciPy oracle command:
  `python3 docs/perf_oracle_pdist.py`
- Same-worker internal A/B on rch worker `hz2`:

  | Workload | Baseline mean | Candidate mean | Candidate/baseline | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | `pdist/euclidean/256` | 2.9264 ms | 107.45 us | 0.0367x time, 27.24x faster | keep |
  | `pdist/cosine/256` | 3.3658 ms | 114.04 us | 0.0339x time, 29.51x faster | keep |
  | `pdist/euclidean/512` | 3.5554 ms | 425.75 us | 0.1197x time, 8.35x faster | keep |
  | `pdist/cosine/512` | 2.5548 ms | 461.16 us | 0.1805x time, 5.54x faster | keep |

- Local SciPy oracle (`scipy.spatial.distance.pdist`, SciPy 1.17.1):

  | Workload | Candidate mean | SciPy p50 | Candidate/SciPy | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | Euclidean n=256 d=4 | 107.45 us | 94.30 us | 1.14x slower | residual near-parity |
  | Cosine n=256 d=4 | 114.04 us | 87.20 us | 1.31x slower | residual loss |
  | Euclidean n=512 d=4 | 425.75 us | 310.83 us | 1.37x slower | residual loss |
  | Cosine n=512 d=4 | 461.16 us | 283.75 us | 1.63x slower | residual loss |

- Correctness/conformance guards:
  - PASS: `rch exec -- cargo test -p fsci-spatial pdist --lib -- --nocapture`
    (10 passed).
  - PASS: `rch exec -- cargo test -p fsci-spatial --lib -- --nocapture`
    (206 passed, 2 ignored).
  - PASS: `rch exec -- cargo check -p fsci-spatial --all-targets`; existing
    warning remains `point_in_circumcircle` dead code.
  - BLOCKED: `rch exec -- cargo test -p fsci-conformance -- --nocapture`
    fails before this spatial lane in `crates/fsci-conformance/tests/e2e_sparse.rs`
    with `SolveResult` passed where `&[f64]` is expected.
  - BLOCKED: `rch exec -- cargo clippy -p fsci-spatial --all-targets -- -D warnings`
    stops in dependency crate `fsci-linalg` on existing lint debt.
  - BLOCKED: `cargo fmt -p fsci-spatial --check` reports pre-existing
    `fsci-spatial` source/bench rustfmt drift outside this patch; `git diff --check`
    is clean.
- Retry condition: do not retry generic metric-dispatch removal, dim-4 scalar
  helper extraction, or dim-4 serial gating for this workload. The next
  credible route is a deeper layout/kernel change: packed SoA/flat
  contiguous point storage, batch several pair outputs per inner loop, or a
  generated AVX-style dim-specialized kernel that removes `Vec<Vec<f64>>`
  pointer chasing and matches SciPy's C distance loop constants.

## 2026-06-19 - frankenscipy-nm8ex.1 - spatial pdist dim-4 flat row staging

- Agent: cod-a / MistyBirch
- Lever: after the dim-4 direct serial kernel keep, stage validated 4-column
  `Vec<Vec<f64>>` rows into compact `[f64; 4]` points once per `pdist` call and
  run the same Euclidean/Cosine arithmetic over the fixed-width row layout. This
  removes hot-loop pointer chasing and slice-length metadata from the O(n^2)
  all-pairs loop without changing arithmetic order or adding unsafe code.
- Graveyard/artifact route tested: A1 numeric-kernel cache locality and
  constant-factor collapse, flat-vector layout from nested data-parallelism,
  and profile-first "constants kill you" discipline. No SIMD or batch-output
  rewrite was mixed into this commit, so the attribution stays on row layout.
- Decision: KEEP as an internal win, route residual SciPy loss deeper. No
  revert.
- Artifact:
  `tests/artifacts/perf/2026-06-19-nm8ex1-pdist-flatdim4-ovhb/EVIDENCE.md`
- Baseline/candidate command:
  `RCH_WORKER=ovh-b CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo bench -p fsci-spatial --bench spatial_bench -- pdist --noplot`
- SciPy oracle command:
  `python3 docs/perf_oracle_pdist.py`
- Same-worker internal A/B on rch worker `ovh-b`:

  | Workload | Baseline median | Candidate median | Candidate/baseline | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | `pdist/euclidean/256` | 263.00 us | 172.83 us | 0.657x time, 1.52x faster | keep |
  | `pdist/cosine/256` | 381.98 us | 208.89 us | 0.547x time, 1.83x faster | keep |
  | `pdist/euclidean/512` | 794.72 us | 714.58 us | 0.899x time, 1.11x faster | keep |
  | `pdist/cosine/512` | 1.1930 ms | 828.70 us | 0.695x time, 1.44x faster | keep |

- Local SciPy oracle (`scipy.spatial.distance.pdist`, SciPy 1.17.1, NumPy 2.4.3):

  | Workload | Candidate median | SciPy p50 | Candidate/SciPy | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | Euclidean n=256 d=4 | 172.83 us | 88.96 us | 1.94x slower | residual loss |
  | Cosine n=256 d=4 | 208.89 us | 79.69 us | 2.62x slower | residual loss |
  | Euclidean n=512 d=4 | 714.58 us | 309.79 us | 2.31x slower | residual loss |
  | Cosine n=512 d=4 | 828.70 us | 275.14 us | 3.01x slower | residual loss |

- Win/loss/neutral vs SciPy: 0 / 4 / 0. Internal A/B: 4 / 0 / 0.
- Correctness/conformance guards:
  - PASS: `rch exec -- cargo test -p fsci-spatial pdist_dim4_fast_paths_match_metric_helpers -- --nocapture`
    (1 passed; validates bit-identical dim-4 Euclidean/Cosine output against
    `metric_distance`).
  - PASS: `rch exec -- cargo test -p fsci-spatial --lib -- --nocapture`
    (206 passed, 2 ignored).
  - PASS: `rch exec -- cargo check -p fsci-spatial --all-targets`.
  - PASS: `rch exec -- cargo clippy -p fsci-spatial --all-targets --no-deps -- -D warnings`
    after clearing same-file pre-existing lint blockers.
  - PASS: candidate Criterion run completed on `ovh-b`; Criterion reported
    statistically significant improvements for all four rows.
  - PASS: `git diff --check`.
  - PASS: `ubs` on the changed file set exited 0; it reported no critical
    issues and only pre-existing broad warnings in the large spatial module.
  - BLOCKED: `cargo fmt --check -p fsci-spatial` remains red on pre-existing
    `fsci-spatial` bench/source rustfmt drift outside this patch.
- Retry condition: do not retry fixed-width row staging alone. The next
  credible route must change the inner kernel more deeply: batch several
  output pairs per loop, generate dim-specialized SIMD-style kernels, or move
  the public representation toward packed SoA/flat buffers so the copy into
  `[f64; 4]` disappears.

## 2026-06-19 - frankenscipy-8l8r1.115 - randomized_eigh projected sketch

- Agent: cod-b / MistyBirch
- Lever: keep `randomized_eigh` on a projected symmetric sketch: deterministic
  random block, thin modified Gram-Schmidt basis, two power iterations, and a
  full eigensolve only on the small `q^T A q` matrix.
- Decision: KEEP. No revert. The route is a measured head-to-head win against
  SciPy subset `eigh` on the scoped low-rank symmetric top-k workloads.
- Artifact: `tests/artifacts/perf/frankenscipy-8l8r1.115/EVIDENCE.md`
- Remote guard commands:
  - `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo check -p fsci-linalg --all-targets`
  - `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo test -p fsci-linalg randomized_eigh_matches_full_eigh_on_low_rank --lib -- --nocapture`
  - `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo bench -p fsci-linalg --bench linalg_bench randomized_eigh_gauntlet_scipy -- --noplot`
- SciPy oracle command: local same-host run
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo bench -p fsci-linalg --bench linalg_bench randomized_eigh_gauntlet_scipy -- --noplot`
  because rch worker `ovh-a` skipped the SciPy rows with missing
  `scipy.linalg`.

| Workload | Rust randomized mean | Rust full `eigh` mean | SciPy subset `eigh` mean | SciPy/Rust randomized | Full/Rand internal | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 256x256, k=16 | 3.492738 ms | 11.509742 ms | 4.879053 ms | 1.40x faster | 3.30x faster | win |
| 512x512, k=24 | 16.774593 ms | 136.046217 ms | 198.116969 ms | 11.81x faster | 8.11x faster | win |

- Negative evidence: Rust full `eigh` remains 3.30x to 8.11x slower than the
  projected route on these top-k low-rank shapes. Do not route this workload
  back through the full dense eigensolver unless a future conformance failure
  invalidates the randomized projection contract.
- Retry condition: continue only with deeper sketch-quality or blocked-operator
  improvements that preserve the existing low-rank invariant tests and improve
  the same `randomized_eigh_gauntlet_scipy` group; reject scalar housekeeping or
  allocation-only tweaks that do not move this head-to-head SciPy ratio.

## 2026-06-19 - Gauntlet verification - fsci-opt least_squares scratch cluster

- Agent: cod-b / MistyBirch
- Beads verified: `frankenscipy-szky7`, `frankenscipy-y1mzk`
- Decision: KEEP. No revert. All measured head-to-head rows are wins vs the
  original SciPy LM path on warmed single-process realistic Python callback
  workloads.
- Artifact: `tests/artifacts/perf/2026-06-19-opt-least-squares-gauntlet/least_squares_vs_scipy.json`
- Rust bench command: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo bench -p fsci-opt --bench optimize_bench -- least_squares --noplot`
- SciPy oracle: Python 3, SciPy 1.17.1, NumPy 2.4.3, `scipy.optimize.least_squares(method="lm")`.

| Workload | Rust Criterion p50 (us) | SciPy p50 (us) | SciPy p95 (us) | SciPy/Rust p50 | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `least_squares/rosenbrock_residual` | 2.558 | 1404.547 | 2297.906 | 549.08x | win |
| `least_squares/exp_curve_64` | 16.932 | 753.120 | 1020.621 | 44.48x | win |
| `least_squares/exp_linear_curve_128` | 49.724 | 893.946 | 1015.962 | 17.98x | win |

- Correctness/conformance guards:
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo check -p fsci-opt`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo test -p fsci-opt --test metamorphic_tests mr_least_squares -- --nocapture` (2 passed)
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo run -p fsci-opt --release --bin diff_lsq`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo run -p fsci-opt --release --bin diff_leastsq`
- Blocker recorded separately: broad `cargo test -p fsci-opt least_squares`
  still fails before reaching least-squares assertions because unrelated
  `crates/fsci-opt/src/lib.rs` test modules miss imports for numerical helper
  functions and `OptError`; filed follow-up `frankenscipy-uxs8k`.
- Negative evidence: none for this cluster. Do not retry or revert the
  fixed-shape Jacobian scratch reuse or LM normal-equation scratch reuse on the
  basis of this gauntlet; route future opt perf work to a lower-level hotspot or
  a workload that shows a measured neutral/loss row.

## 2026-06-19 - frankenscipy-u0ucw - GAUNTLET measured wide pinv vs SciPy

- Agent: cod-a / MistyBirch
- Workload: full-row-rank wide pseudo-inverse, 500x1000 dense
  Cauchy-like matrix with a strong diagonal, matching the committed
  `make_underdetermined` Criterion generator and SciPy oracle construction.
- Subject commits: `6c139073` wide `pinv` Cholesky TRSM plus `c39e9394`
  diagonal rcond gate, measured at `41bf34a4`.
- Criterion command:
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo bench -p fsci-linalg --bench linalg_bench -- u0ucw_gauntlet_scipy_pinv --noplot`
- Environment: local same-host oracle because rch worker `vmi1152480` lacked
  `scipy` for the Python oracle row; local `python3` 3.13.7, NumPy 2.4.3,
  SciPy 1.17.1, rustc 1.98.0-nightly.
- Results use Criterion mean point estimates from
  `target/criterion/u0ucw_gauntlet_scipy_pinv/*/new/estimates.json`:

  | Route | Mean | Ratio vs SciPy | Internal delta | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | SciPy `scipy.linalg.pinv(check_finite=False)` | 7.257573 s | 1.00x | oracle | reference |
  | Rust current Cholesky + diagonal rcond gate | 183.699926 ms | 39.51x faster | 1.40x vs eigengate, 2.82x vs SVD fallback | keep |
  | Rust Cholesky + old eigenspectrum rcond gate | 257.672693 ms | 28.17x faster | current is 28.71% less time | superseded |
  | Rust SVD fallback route | 517.524097 ms | 14.02x faster | current is 64.50% less time | superseded |

- rch evidence: `rch exec -- cargo bench -p fsci-linalg --bench linalg_bench
  -- u0ucw_gauntlet_scipy_pinv --noplot` built and timed the Rust rows on
  worker `vmi1152480` (`current` 171.97 ms, `eigengate` 268.75 ms,
  `svd_fallback` 416.57 ms), then the SciPy oracle row failed with
  `ModuleNotFoundError: No module named 'scipy'`. That worker result is
  retained only as infrastructure evidence and is not used for the
  head-to-head ratio.
- Correctness guard: no revert. Keep the Cholesky route and diagonal rcond gate
  because the measured route is materially faster than the old in-crate routes
  and 39.51x faster than SciPy on this realistic wide full-row-rank workload.
- Retry condition: do not retry the old eigenspectrum rcond gate or SVD
  fallback for this 500x1000 full-row-rank wide workload unless a future
  conformance failure, condition-audit regression, or same-host SciPy/Criterion
  rerun reverses the current >1.4x internal win.

## 2026-06-19 - frankenscipy-y1mzk - least_squares LM normal-equation scratch

- Agent: cod-b / MistyBirch
- Lever: keep the `least_squares` Levenberg-Marquardt algorithm on the same
  residual/Jacobian path while reusing fixed-shape scratch for `J^T J`,
  `J^T r`, the damped normal-equation matrix, Cholesky factor, forward/backward
  solve vectors, and `J*step` predicted-reduction work.
- Status: measured KEEP. 2026-06-19 gauntlet found wins vs SciPy LM on all
  selected least-squares rows: 549.08x, 44.48x, and 17.98x p50 SciPy/Rust.
- Correctness guard: existing `least_squares`, `curve_fit`, and `leastsq`
  coverage exercises convergence and covariance paths; the helper preserves
  residual evaluation order, finite-difference values, LM damping updates,
  fallback diagonal solve semantics, and `nfev`/`njev`/`nit` accounting.
- Benchmark guard: Criterion `least_squares/rosenbrock_residual`,
  `least_squares/exp_curve_64`, and `least_squares/exp_linear_curve_128`
  rows quantify the scratch-reuse path against SciPy LM in the gauntlet
  artifact above.
- Retry condition: keep only if same-worker focused `fsci-opt` least-squares /
  curve-fit timings improve without convergence, cost, parameter, Jacobian, or
  counter drift; if timings are neutral/slower, reject this exact LM
  normal-equation scratch formulation and do not retry unless allocation
  profiles put damped normal-equation matrix/Cholesky/`J*step` scratch back in
  the top-5 `fsci-opt` hotspots.

## 2026-06-19 - frankenscipy-szky7 - least_squares fixed-shape Jacobian scratch

- Agent: cod-b / MistyBirch
- Lever: keep `fsci_opt::curvefit::least_squares` on the same
  Levenberg-Marquardt control path while reusing the finite-difference
  Jacobian rows and perturbation vector across accepted steps instead of
  allocating a fresh `m x n` `Vec<Vec<f64>>` plus `x_perturbed` for every
  Jacobian rebuild.
- Status: measured KEEP. 2026-06-19 gauntlet found wins vs SciPy LM on all
  selected least-squares rows: 549.08x, 44.48x, and 17.98x p50 SciPy/Rust.
- Correctness guard: existing `least_squares`, `curve_fit`, and `leastsq`
  coverage plus the `diff_lsq` probe exercise the residual evaluation contract.
  The helper preserves forward-difference order, residual call count,
  `nfev`/`njev` accounting, accepted/rejected damping updates, and final
  Jacobian row/column shape.
- Benchmark guard: Criterion `least_squares/rosenbrock_residual`,
  `least_squares/exp_curve_64`, and `least_squares/exp_linear_curve_128`
  rows quantify the allocation-removal path against SciPy LM in the gauntlet
  artifact above.
- Retry condition: keep only if same-worker focused `fsci-opt` least-squares /
  curve-fit timings improve without convergence, `nfev`/`njev`, cost, or
  parameter drift; if timings are neutral or slower, reject this exact
  fixed-shape Jacobian scratch route and do not retry unless allocation
  profiles put curvefit Jacobian matrix allocation back in the top-5
  `fsci-opt` hotspots.

## 2026-06-19 - frankenscipy-8l8r1.122 - L-BFGS-B Wolfe probe scratch reuse

- Agent: cod-b / MistyBirch
- Lever: route unconstrained `L-BFGS-B` Strong-Wolfe finite-difference
  gradient probes through `line_search_wolfe2_with_gradient_probe`, reusing the
  line-search trial buffer and gradient `Vec` instead of allocating a fresh
  `g` and `xp` inside every curvature probe.
- Status: measured REJECT and reverted. The attempted mutable-probe path was
  neutral on 2D Rosenbrock and slower on both larger rows, so the source path
  was restored to the parent `line_search_wolfe2` implementation while keeping
  the new Criterion/SciPy measurement harness.
- Artifact: `tests/artifacts/perf/2026-06-19-opt-lbfgsb-gauntlet/lbfgsb_wolfe_probe_reject.json`
- Optimization commit under test: `b5dbf1244e52632edc9bd0edc2102cb3ff78dfad`
  vs parent `69ae5d214f8e90356789b112cff30a5c69b43d2a`.

Same-worker internal A/B (`ovh-a`, Criterion p50):

| Workload | Parent p50 (us) | Candidate p50 (us) | Candidate time vs parent | Verdict |
| --- | ---: | ---: | ---: | --- |
| `lbfgsb/rosenbrock_unconstrained_fd/2` | 17.491 | 17.405 | 0.995x | neutral |
| `lbfgsb/rosenbrock_unconstrained_fd/10` | 87.087 | 106.440 | 1.222x | loss |
| `lbfgsb/quadratic_unconstrained_fd/32` | 5.246 | 6.055 | 1.154x | loss |

Post-revert current route vs original SciPy (`hz2` Rust Criterion p50, local
SciPy 1.17.1 / NumPy 2.4.3 oracle p50):

| Workload | Current Rust p50 (us) | SciPy p50 (us) | SciPy/Rust p50 | Verdict |
| --- | ---: | ---: | ---: | --- |
| `lbfgsb/rosenbrock_unconstrained_fd/2` | 22.236 | 4585.899 | 206.24x | current route remains fast |
| `lbfgsb/rosenbrock_unconstrained_fd/10` | 105.090 | 18262.642 | 173.78x | current route remains fast |
| `lbfgsb/quadratic_unconstrained_fd/32` | 6.313 | 1447.172 | 229.23x | current route remains fast |

- Correctness/conformance guards:
  - PASS: `cargo fmt -p fsci-opt --check`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo check -p fsci-opt --all-targets`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo test -p fsci-opt lbfgsb -- --nocapture`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test diff_opt_lbfgsb_minimize -- --nocapture`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo clippy -p fsci-opt --all-targets -- -D warnings`
- Retry condition: do not retry this mutable finite-difference probe scratch
  formulation. Reopen only if a fresh allocation profile again puts
  `L-BFGS-B` Wolfe gradient-probe Vec churn in the top-5 `fsci-opt` hotspots
  and a new same-worker parent/candidate A/B shows a material win across the
  10D Rosenbrock and 32D quadratic rows.

## 2026-06-19 - frankenscipy-8l8r1.116 - FFT CSD rfft real-spectrum route

- Agent: cod-b / MistyBirch
- Lever: route `fsci_fft::cross_spectral_density` real input pairs through
  two `rfft` calls instead of promoting both inputs to complex buffers and
  running two full complex FFTs before taking the one-sided half.
- Status: measured REJECT and reverted. The rfft route won the large internal
  row but regressed the 4096 row and remained slower than the fastest
  equivalent SciPy `scipy.fft.rfft` cross-spectrum formula on both measured
  sizes. The release route is restored to the full-complex implementation while
  keeping the Criterion CSD rows, SciPy oracle script, and full-complex
  equivalence guard.
- Artifact: `tests/artifacts/perf/2026-06-19-fft-csd-gauntlet/csd_rfft_reject.json`
- Optimization commit under test: `d027c140bc8a937877e8c018cf7265d1f4bc5049`
  vs parent `55f32c99f69991f4ab252621dd86948dc6e95b20`.

Same-worker internal A/B (`hz1`, Criterion mean; parent scratch had only the
same CSD bench harness added, with parent library code unchanged):

| Workload | Parent full-complex mean | Candidate rfft mean | Candidate time vs parent | Verdict |
| --- | ---: | ---: | ---: | --- |
| `fft_helpers/cross_spectral_density/4096` | 112.08 us | 125.88 us | 1.123x | loss |
| `fft_helpers/cross_spectral_density/65536` | 4.9543 ms | 2.3509 ms | 0.475x | win |

Local original-SciPy oracle (`python3 docs/perf_oracle_fft_csd.py --reps 120
--warmups 5`, Python 3.13.7 / NumPy 2.4.3 / SciPy 1.17.1):

| Workload | SciPy rfft-formula p50 | Candidate Rust mean | SciPy/Rust | Verdict |
| --- | ---: | ---: | ---: | --- |
| `cross_spectral_density/4096` | 72.091 us | 125.88 us | 0.573x | Rust slower |
| `cross_spectral_density/65536` | 1.653584 ms | 2.3509 ms | 0.703x | Rust slower |

- Correctness/conformance guards:
  - PASS: `python3 docs/perf_oracle_fft_csd.py --reps 120 --warmups 5`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo bench -p fsci-fft --bench fft_bench -- fft_helpers/cross_spectral_density --noplot` for the parent/candidate measurement rows above
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo check -p fsci-fft --all-targets`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo test -p fsci-fft cross_spectral_density -- --nocapture`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo clippy -p fsci-fft --all-targets -- -D warnings`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo test -p fsci-fft rfft_matches_exact_numpy_dft_golden_values -- --nocapture`
  - PASS: `git diff --check` and targeted `rustfmt --edition 2024 --check crates/fsci-fft/benches/fft_bench.rs crates/fsci-fft/src/transforms.rs`
  - PASS: `ubs` on the changed file set exited 0.
  - BLOCKED: broad `fsci-fft` rustfmt still reports pre-existing file-wide drift in `src/lib.rs` and `src/helpers.rs`; this commit avoids unrelated formatting churn.
- Retry condition: do not retry this standalone rfft CSD route. Reopen only if
  a fresh same-worker profile attributes a clearly-above-noise share to full
  complex promotion inside `cross_spectral_density` on a workload of at least
  65536 samples, and the replacement beats the fastest equivalent SciPy rfft
  formula while not regressing the 4096 row.

## 2026-06-19 - frankenscipy-fo9cj - sparse Arnoldi row-major basis arena

- Agent: cod-b / MistyBirch
- Lever: replace the `krylov_arnoldi_eigs` `Vec<Vec<f64>>` basis and allocating
  operator return with a row-major basis arena plus a reusable operator scratch
  buffer; switch `eigsh`, `eigs`, and `svds` callers to `csr_matvec_into` /
  `csc_matvec_into`.
- Status: MEASURED REJECT. Same-worker rch A/B on `ovh-a` showed the row-major
  basis arena regressed every `eigsh` row, while `svds` only produced tiny mixed
  movement. The source route was restored to the parent implementation.
- Internal A/B, `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b`,
  `cargo run --profile release-perf -p fsci-sparse --bin perf_eigsh`:

| Workload | Candidate | Parent/restored | Parent/candidate | Result |
| --- | ---: | ---: | ---: | --- |
| `eigsh n=2000 k=6` | 1.667 ms | 1.184 ms | 0.71x | loss |
| `eigsh n=8000 k=6` | 6.594 ms | 5.548 ms | 0.84x | loss |
| `eigsh n=20000 k=8` | 16.147 ms | 11.599 ms | 0.72x | loss |

- Internal A/B, same worker/target dir,
  `cargo run --profile release-perf -p fsci-sparse --bin perf_svds`:

| Workload | Candidate | Parent/restored | Parent/candidate | Result |
| --- | ---: | ---: | ---: | --- |
| `svds 2200x2000 k=6` | 1.203 ms | 1.191 ms | 0.99x | neutral/loss |
| `svds 8200x8000 k=6` | 4.654 ms | 4.929 ms | 1.06x | tiny win |
| `svds 20200x20000 k=8` | 12.362 ms | 12.534 ms | 1.01x | tiny win |

- SciPy head-to-head on the restored route, same deterministic matrix family:

| Workload | Restored Rust | SciPy 1.17.1 | SciPy/Rust | Result |
| --- | ---: | ---: | ---: | --- |
| `eigsh n=2000 k=6` | 1.184 ms | 3.000 ms | 2.53x | Rust win |
| `eigsh n=8000 k=6` | 5.548 ms | 2.768 ms | 0.50x | Rust loss |
| `eigsh n=20000 k=8` | 11.599 ms | 43.023 ms | 3.71x | Rust win |
| `svds 2200x2000 k=6` | 1.191 ms | 17.567 ms | 14.75x | Rust win |
| `svds 8200x8000 k=6` | 4.929 ms | 4.861 ms | 0.99x | neutral |
| `svds 20200x20000 k=8` | 12.534 ms | 42.018 ms | 3.35x | Rust win |

- Ratio summary vs SciPy after revert: 4 wins / 1 loss / 1 neutral. The real
  measured gap to target next is mid-size `eigsh n=8000 k=6`, not the discarded
  basis-arena allocation lever.
- Correctness/conformance guards:
  - PASS: rch `cargo check -p fsci-sparse --all-targets`
  - PASS: rch `cargo test -p fsci-sparse eig -- --nocapture`
  - PASS: rch `cargo test -p fsci-sparse svds -- --nocapture`
  - PASS: local SciPy-backed `cargo test -p fsci-conformance --test
    diff_sparse_eigsh_largest -- --nocapture`
  - PASS: local SciPy-backed `cargo test -p fsci-conformance --test
    diff_sparse_svds -- --nocapture`
  - PASS: rch `cargo clippy -p fsci-sparse --all-targets --no-deps -- -D warnings`
  - PASS: `git diff --check` and `ubs crates/fsci-sparse/src/linalg.rs`
  - BLOCKED: rch SciPy-backed sparse conformance on `ovh-a` because the worker
    image lacks `scipy`; local SciPy 1.17.1 supplied the oracle proof.
  - BLOCKED: touched-file rustfmt check by pre-existing file-wide formatting
    drift outside this sparse revert hunk; no unrelated formatting churn was
    introduced.
- Retry condition: do not retry this row-major `Vec<Vec>` replacement or mutable
  operator scratch route. Reopen only if a fresh allocator/profile run puts
  Arnoldi basis allocation in the top five costs and a new design avoids the
  per-step basis copy cost that made this formulation slower.

## 2026-06-18 - frankenscipy-bpzha - RK step scratch double-buffer

- Agent: cod-b / MistyBirch
- Lever: move `rk_step` rejected-attempt storage into solver-owned reusable
  buffers for `dy`, `y_stage`, `y_new`, and `f_new`; accepted steps swap the
  buffers into live state, while rejected attempts overwrite the same scratch on
  retry.
- Status: measured reject and reverted on 2026-06-19. The scratch variants had
  one promising scalar row, but the broader RK gate failed once Lorenz/vector
  workloads were repeated on fresh rch workers.
- Worker/target: rch `vmi1149989`,
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b`.
  RCH rewrote that to worker-scoped target directories under each scratch
  worktree; later repeats also used `hz2`, `hz1`, `ovh-a`, and `ovh-b` with the
  same target-dir root.

Initial internal A/B against `07ec1ab4^` parent looked keep-worthy but did not
survive repeat/fresh-worker validation:

| Workload | Parent route | Candidate route | Candidate/parent | Invariant |
| --- | ---: | ---: | ---: | --- |
| `vmi1149989 solve_ivp RK45 exponential`, repeats=3000 | 18.589129 us/call | 14.068991 us/call | 1.321x faster | `nfev=741000`, checksum `6.061522287381e5` |
| `vmi1149989 solve_ivp RK45 Lorenz`, repeats=3000 | 28.266539 us/call | 21.940886 us/call | 1.288x faster | `nfev=975000`, checksum `4.576590840398e6` |
| `hz2 solve_ivp RK45 exponential`, repeats=3000 | 17.356838 us/call | 13.863079 us/call | 1.252x faster | `nfev=741000`, checksum `6.061522287381e5` |
| `hz2 solve_ivp RK45 Lorenz`, repeats=3000 | 21.951172 us/call | 23.402816 us/call | 0.938x slower | `nfev=975000`, checksum `4.576590840398e6` |
| `hz1 solve_ivp RK45 Lorenz`, repeats=3000 | 28.621224 us/call | 31.335899 us/call | 0.913x slower | `nfev=975000`, checksum `4.576590840398e6` |
| `ovh-a solve_ivp RK45 Lorenz`, repeats=3000 | 20.597014 us/call | 32.037205 us/call | 0.643x slower | `nfev=975000`, checksum `4.576590840398e6` |

Final scalar/vector helper sanity check was also red: the final candidate's
`ovh-b solve_ivp RK45 exponential` row measured `27.755498 us/call`, slower
than every parent exponential row observed in this lane. That row had no
same-worker parent pair, so it is recorded as a rejection signal rather than a
standalone ratio.

Internal keep-gate summary for the scratch lever: 1 paired win / 3 paired
losses / 0 neutral after excluding the preliminary `vmi1149989` route as
superseded by later final-source validation.

SciPy head-to-head for the restored parent route, local SciPy 1.17.1 oracle with
the same ODE/tolerance shape:

| Workload | Restored Rust | SciPy 1.17.1 | SciPy/Rust | Result |
| --- | ---: | ---: | ---: | --- |
| `solve_ivp RK45 exponential` | 18.589129 us/call | 1443.255860 us/call | 77.64x | Rust win |
| `solve_ivp RK45 Lorenz` | 28.266539 us/call | 2062.735365 us/call | 72.97x | Rust win |

- Ratio summary vs SciPy for restored Rust: 2 wins / 0 losses / 0 neutral.
- Ratio summary for this scratch lever vs restored parent: 1 win / 3 losses /
  0 neutral on paired final-validation rows, so the source change was reverted.
- Correctness/conformance guards:
  - PASS: rch `cargo check -p fsci-integrate --all-targets` on the scratch
    candidate before rejection.
  - PASS: rch `cargo test -p fsci-integrate rk -- --nocapture` on the scratch
    candidate after the `RkStepScratch` cleanup (`17` filtered RK tests plus
    the focused property row passed).
  - PASS: rch `cargo test -p fsci-conformance --test e2e_ivp -- --nocapture`
    on the scratch candidate (`11` IVP scenarios passed on `vmi1227854`).
  - PASS: local `rustfmt --edition 2024 crates/fsci-integrate/src/rk.rs` and
    `git diff --check` before final source revert.
  - BLOCKED: rch `cargo clippy -p fsci-integrate --all-targets --no-deps --
    -D warnings` had no remaining RK scratch API lint, but still failed on
    pre-existing unrelated `api.rs`/`quad.rs` lints:
    `solve_event_equation` too-many-arguments, `quad.rs` excessive precision,
    and `quad.rs` type complexity.
- Decision: reject and revert. The source tree is back to the parent RK route;
  the public SciPy ratios remain wins because the baseline Rust solver already
  avoids Python callback overhead. Do not retry this scratch formulation unless
  a fresh allocation profile puts RK temporary `Vec` churn back in the top five
  costs and the replacement proves both scalar and vector ODE rows on the same
  worker in one run window.

## 2026-06-18 - frankenscipy-6m75u - Wolfe trial-point scratch reuse

- Agent: cod-b / MistyBirch
- Lever: replace per-probe `x + alpha*d` trial `Vec` construction in public
  Wolfe line search with one reusable trial buffer, and thread that buffer
  through the bisection zoom phase.
- Status: measured reject on 2026-06-19. Same-worker `hz2` proof-mode
  Criterion showed only neutral/noisy movement versus `fcbcbaf4^`, so the
  public Wolfe source path was restored to the parent implementation.
- Same-worker Rust gate:
  `RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz2 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b
  rch exec -- cargo bench -p fsci-opt --bench optimize_bench --
  rosenbrock_exact_gradient/10 --sample-size 10 --measurement-time 1
  --warm-up-time 1`.
  BFGS median `64.178 us -> 63.085 us` (`1.017x`, overlapping intervals);
  CG median `170.54 us -> 157.46 us` (`1.083x`, overlapping intervals).
  Repeated current Criterion reported no performance change (`p = 0.48` BFGS,
  `p = 0.85` CG), so this is neutral rather than a keep.
- SciPy scorecard: `2/0/0` final-source wins/losses/neutrals, but Python ran
  on `thinkstation1` because RCH does not remote non-compilation commands here.
  Final restored Rust rows were `88.1x` faster than SciPy BFGS exact-gradient
  Rosenbrock-10 and `121.0x` faster than SciPy CG exact-gradient Rosenbrock-10.
- Artifact: `tests/artifacts/perf/frankenscipy-6m75u/EVIDENCE.md`.
- Retry condition: do not retry public Wolfe trial-point scratch reuse unless a
  fresh allocation profile puts public Wolfe trial construction back in the top
  opt hotspots. Probe-path scratch reuse is separate and not rejected here.

## 2026-06-18 - frankenscipy-va60h - linkage row-major distance arena

- Agent: cod-a / MistyBirch
- Lever: replace the nearest-neighbour linkage core's `(2n-1) x (2n-1)`
  `Vec<Vec<f64>>` inter-cluster distance matrix with one row-major `Vec<f64>`
  arena and stride indexing; fill that arena directly from observations or
  condensed precomputed distances for the non-Centroid/Median methods.
- Status: measured gauntlet complete on 2026-06-19. Decision: KEEP the flat
  arena as an internal win, but record the full routine as a SciPy LOSS on this
  workload. A direct production revert probe showed the nested route would be
  slower on both measured rows, so no revert remains in the release candidate.
- Artifact:
  `tests/artifacts/perf/2026-06-19-va60h-linkage-gauntlet/`
- Correctness guard: benchmark setup asserted current flat rows are
  byte-identical to the benchmark-local legacy nested NN-array route; filtered
  `fsci-cluster linkage` tests passed via rch (28 unit tests, 9 metamorphic
  tests), including `linkage_flat_core_matches_precomputed_condensed_contract`;
  `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test
  diff_cluster_linkage_from_distances -- --nocapture` passed locally against
  SciPy 1.17.1.
- Benchmark command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a
  cargo bench -p fsci-cluster --bench cluster_bench --
  va60h_gauntlet_linkage --noplot`.

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust current flat `linkage(Average)`, n=800 d=4 | 6.1713 ms | 1.385x slower than SciPy | SciPy loss, internal keep |
| Rust legacy nested helper `linkage(Average)`, n=800 d=4 | 6.9616 ms | current flat is 1.128x faster | internal win |
| SciPy `scipy.cluster.hierarchy.linkage(method="average")`, n=800 d=4 | 4.4550 ms | 1.00x oracle | reference |
| Rust current flat `linkage(Ward)`, n=800 d=4 | 7.5250 ms | 1.497x slower than SciPy | SciPy loss, internal neutral/win |
| Rust legacy nested helper `linkage(Ward)`, n=800 d=4 | 7.6707 ms | current flat is 1.019x faster | internal neutral/win |
| SciPy `scipy.cluster.hierarchy.linkage(method="ward")`, n=800 d=4 | 5.0256 ms | 1.00x oracle | reference |

- Revert probe: manually reverting production to nested rows and rerunning the
  same Criterion group showed the flat route faster than the reverted production
  route by 1.290x on Average and 1.251x on Ward. The revert was therefore
  undone before commit.
- Gate notes: `cargo check -p fsci-cluster --benches` passed with an existing
  `perf_kmeans.rs` warning. `cargo fmt -p fsci-cluster --check` is blocked by
  existing `perf_isomap.rs` formatting drift; `cargo clippy -p fsci-cluster
  --benches -- -D warnings` is blocked by existing `fsci-linalg` dependency
  lints before this benchmark file is linted.
- Next route: if release parity against SciPy is required for hierarchical
  clustering, route deeper into the algorithmic gap with SciPy's compiled
  linkage implementation rather than retrying full-square arena layout changes.

## 2026-06-18 - frankenscipy-8l8r1.118 - coherence chunk-local spectra accumulator

- Agent: cod-b / MistyBirch
- Lever: keep `coherence` on the fused Welch segment pass, but replace the
  per-segment `CoherenceSegmentSpectra` materialization with chunk-local
  Pxy/Pxx/Pyy accumulators and reusable windowed `wx`/`wy` scratch buffers. This
  removes one spectra allocation triplet per segment and avoids retaining all
  segment spectra before the final fold on the target
  `spectral/coherence/65536_w1024_o512` workload.
- Status: measured gauntlet complete on 2026-06-19. Decision: KEEP. The fused
  coherence route beats original `scipy.signal.coherence` by 8.65x on the
  scoped 65536-sample Hann-window workload, and beats the internal triple-CSD
  composition by 2.98x locally and 2.80x on rch worker `hz1`.
- Artifact: `tests/artifacts/perf/frankenscipy-8l8r1.118/EVIDENCE.md`.
- Correctness guard: existing `coherence_matches_compositional_csd_formula`
  compares the fused path against `csd(x,y)`, `csd(x,x)`, and `csd(y,y)`;
  existing SciPy-reference coherence coverage anchors the public tolerance
  contract. `cargo check -p fsci-signal --all-targets` and the focused
  `coherence_matches` tests passed via rch.

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust fused `coherence`, local SciPy-oracle host | 2.191980 ms | 8.65x faster than SciPy | SciPy win |
| Rust compositional triple-CSD route, local host | 6.536569 ms | fused is 2.98x faster | internal win |
| SciPy `scipy.signal.coherence`, local host | 18.961613 ms | 1.00x oracle | reference |
| Rust fused `coherence`, rch worker `hz1` | 4.3780 ms | 2.80x faster than compositional | internal win |
| Rust compositional triple-CSD route, rch worker `hz1` | 12.269 ms | 1.00x internal baseline | slower |

- Retry condition: do not retry the triple-Welch/triple-CSD coherence route
  unless a same-host gauntlet reverses both the >=2.8x internal fused win and
  the >8x SciPy win. Future signal work should route below this API into FFT
  staging, windowing, or shared Welch segment infrastructure rather than
  decomposing coherence back into independent `csd` calls.

## 2026-06-18 - frankenscipy-8l8r1.119 - BDF Newton streamed scaled RMS norm

- Agent: cod-a / MistyBirch
- Lever: replace `newton_bdf`'s per-Newton-iteration temporary
  `collect::<Vec<_>>()` for `dy[j] / scale[j]` with an allocation-free streamed
  scaled RMS helper over the LU solve vector and scale slice.
- Status: measured gauntlet complete on 2026-06-20. Decision: KEEP. Same-worker
  `hz2` BDF64 improved from 2390.435500 us to 2298.069000 us (1.040x faster);
  BDF128 was neutral/noise at 12032.349600 us baseline vs 12138.374200 us
  candidate (0.991x). Step counters and checksum stayed unchanged.
- Artifact:
  `tests/artifacts/perf/frankenscipy-8l8r1-119-121-integrate-stiff-stream-norms/EVIDENCE.md`.
- Correctness guard: `rms_norm_scaled_matches_collected_reference` proves the
  streamed helper is bit-identical to the old collect-then-`rms_norm` path for
  the same scaled values. `cargo test -p fsci-integrate bdf --lib --
  --nocapture` passed via rch: 16 passed / 0 failed. `cargo test -p
  fsci-conformance --test e2e_ivp -- --nocapture` passed via rch: 11 passed / 0
  failed.
- SciPy head-to-head for final source after Radau revert: BDF64 1959.286800 us
  Rust on rch `ovh-a` vs 26351.239008 us local SciPy (13.45x faster); BDF128
  11052.293000 us Rust vs 29334.902694 us SciPy (2.65x faster).
- Retry condition: do not retry BDF scaled-vector allocation work unless a fresh
  profile puts it back in the top integrate hotspots. The next BDF work should
  be deeper than this micro-allocation.

## 2026-06-18 - frankenscipy-8l8r1.120 - Radau streamed scaled RMS norms

- Agent: cod-a / MistyBirch
- Lever: replace Radau's per-step scaled-vector materialization for Newton
  correction norms and embedded error norms with streamed scaled RMS
  accumulation over the LU solve output and scale slices.
- Status: measured gauntlet complete on 2026-06-20. Decision: REJECT and
  REVERT. Same-worker `hz2` Radau32 regressed from 12586.934900 us baseline to
  14971.401500 us candidate (0.841x); Radau64 regressed from 78394.827700 us to
  81492.956400 us (0.962x). `nfev/njev/nlu` and checksums stayed unchanged, so
  this was a constant-factor regression in the norm formulation rather than a
  different solver path.
- Artifact:
  `tests/artifacts/perf/frankenscipy-8l8r1-119-121-integrate-stiff-stream-norms/EVIDENCE.md`.
- Correctness guard after revert: `cargo test -p fsci-integrate radau --lib --
  --nocapture` passed via rch: 2 passed / 0 failed. `cargo test -p
  fsci-conformance --test e2e_ivp -- --nocapture` passed via rch: 11 passed / 0
  failed.
- SciPy head-to-head for final source after revert: Radau32 10191.487800 us Rust
  on rch `ovh-a` vs 33444.223704 us local SciPy (3.28x faster); Radau64
  70176.946400 us Rust vs 35156.708304 us SciPy (2.00x slower). Final
  stiff-suite score: `3/1/0` vs SciPy.
- Retry condition: do not retry this streamed scaled-RMS formulation. The
  remaining Radau64 SciPy loss is tracked in `frankenscipy-zpunl` and should
  target Radau stage linear algebra/LU reuse, DMatrix/DVector assembly,
  stage-major cache layout, or structured-Jacobian exploitation.

## 2026-06-18 - frankenscipy-8l8r1.121 - BDF streamed step/order error norms

- Agent: cod-a / MistyBirch
- Lever: reuse the BDF streamed scaled RMS helper for accepted-step error norms
  and order-minus/order-plus selection norms, removing three temporary
  coefficient-scaled error vectors that existed only to call `rms_norm`.
- Status: measured gauntlet complete on 2026-06-20. Decision: KEEP with BDF
  `.119`. Same-worker `hz2` BDF64 improved 1.040x and BDF128 was neutral/noise
  at 0.991x; no step-counter or checksum drift was observed.
- Artifact:
  `tests/artifacts/perf/frankenscipy-8l8r1-119-121-integrate-stiff-stream-norms/EVIDENCE.md`.
- Correctness guard: `rms_norm_scaled_matches_collected_reference` now covers
  both direct scaled vectors and coefficient-scaled error vectors with
  bit-identical results to the old collect-then-`rms_norm` path. `cargo test -p
  fsci-integrate bdf --lib -- --nocapture` passed via rch: 16 passed / 0
  failed.
- SciPy head-to-head for final source after Radau revert: `bdf-stiff64` and
  `bdf-stiff128` both remain SciPy wins (13.45x and 2.65x faster,
  respectively).
- Retry condition: do not retry another BDF order-error norm materialization
  micro-lever without a fresh allocation profile. Future BDF work should focus
  on algorithm/linear-solve costs or larger state layouts.

## 2026-06-18 - frankenscipy-8l8r1.118 - CSD chunk-local cross-spectrum accumulator

- Agent: cod-b / MistyBirch
- Lever: apply the coherence accumulator pattern to `csd_with_scaling` itself:
  each worker chunk now reuses `wx`/`wy` segment scratch and folds
  cross-periodograms directly into one Pxy accumulator instead of allocating
  and retaining one `Vec<(re, im)>` per segment before the final average.
- Status: pending batch-test. This is a code-first commit per campaign
  instruction; local `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b
  cargo check -p fsci-signal` is the only required pre-commit gate.
- Correctness guard: existing `csd_auto_spectrum_matches_welch`,
  `csd_scaling_matches_scipy_density_and_spectrum`, and
  `coherence_matches_compositional_csd_formula` cover the scaled CSD contract
  and downstream coherence use.
- Benchmark guard: compare focused Welch/CSD rows, especially
  `spectral/csd/65536_w1024_o512` if present, plus the coherence target that
  consumes `csd` compositionally in the guard path.
- Retry condition: keep only if same-worker CSD/Welch timings improve without
  SciPy density/spectrum drift; if chunk-local reduction grouping or scratch
  initialization costs erase the allocation savings, reject this CSD accumulator
  formulation and do not retry unless retained segment spectra reappear as a
  top-5 signal allocation hotspot.

## 2026-06-18 - frankenscipy-u0ucw - Tall normal-equation Gram/vector streaming

- Agent: cod-a / MistyBirch
- Lever: compute the full-rank tall `pinv`/`lstsq` normal-equation products
  directly from DMatrix column slices: `A^T A` is formed symmetrically without
  materializing `A^T`, and `lstsq` computes `A^T b` / `A^T r` with the same
  contiguous column-dot helper.
- Status: pending batch-test. This is a code-first commit per campaign
  instruction; only local `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a
  cargo check -p fsci-linalg` is expected before commit.
- Correctness guard: `normal_equation_column_helpers_match_nalgebra_products`
  compares the new Gram/vector helpers against the previous nalgebra transpose
  products, while existing full-rank tall `pinv`/`lstsq` route tests preserve
  the SVD-reference tolerance contract.
- Benchmark guard: compare focused tall `pinv`/`lstsq` normal-equation
  workloads, especially 1000x500 and 3000x1500 shapes from `u0ucw`, against the
  previous same-worker commit.
- Retry condition: keep only if same-worker tall `pinv`/`lstsq` timings improve
  without fast-route acceptance, rank, singular-value, residual, or
  pseudo-inverse tolerance drift; if the hand-rolled symmetric Gram loses to
  nalgebra's generic transpose product, reject this exact column-dot formulation
  and do not retry unless profiling puts `A^T` materialization or Gram formation
  back in the top linalg hotspot list.

## 2026-06-19 - frankenscipy-u0ucw - Wide lstsq row-streamed normal equations

- Agent: cod-a / MistyBirch
- Lever tested: route full-row-rank wide `lstsq` normal-equation products
  through the caller's row-major input: form `A A^T` from contiguous row dot
  products and compute `A^T y` / `A^T dy` by streaming rows once, avoiding the
  materialized `A^T` matrix.
- Decision: REVERT. The row-streamed candidate is slower than the prior
  materialized transpose route on the same `rch` worker. The retained current
  materialized route remains much faster than original SciPy on the realistic
  500x1000 workload, so the regression is in the code-first micro-lever rather
  than in the public algorithm choice.
- Artifact: `tests/artifacts/perf/2026-06-19-u0ucw-wide-lstsq-gauntlet/`
- Commands:
  - `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo bench -p fsci-linalg --bench linalg_bench -- u0ucw_gauntlet_scipy_lstsq --noplot`
  - `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo bench -p fsci-linalg --bench linalg_bench -- u0ucw_gauntlet_scipy_lstsq --noplot`

| Route | Environment | Criterion mean | Ratio | Verdict |
| --- | --- | ---: | ---: | --- |
| Rust row-streamed `A A^T` + `A^T y` | `rch` worker `vmi1227854` | 139.965 ms | 0.966x vs materialized | loss, reverted |
| Rust materialized `A^T` pre-revert | `rch` worker `vmi1227854` | 135.206 ms | 1.00x internal reference | keep old route |
| Rust current materialized `A^T` after revert | local SciPy host | 109.370 ms | 11.46x faster than SciPy | keep |
| SciPy `scipy.linalg.lstsq(check_finite=False)` | local Python 3.13.7 / NumPy 2.4.3 / SciPy 1.17.1 | 1.253347 s | 1.00x oracle | reference |

- Conformance/correctness guard:
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo check -p fsci-linalg --benches`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo test -p fsci-linalg wide_pinv -- --nocapture`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo test -p fsci-linalg public_wide_min_norm_lstsq_route_perf_probe --release -- --ignored --nocapture`
  - The release probe reported `shape=256x512`, `lstsq_speedup=15.571283`,
    and `lstsq_max_abs_diff=3.38840067115597776e-13` against the reference
    route.
- Infrastructure note: `vmi1227854` did not have SciPy importable, so the
  row-streaming-vs-materialized loss uses remote same-worker Rust A/B evidence,
  while the ratio-vs-SciPy row uses the local host with SciPy installed.
- Retry condition: do not retry this exact row-streamed wide `lstsq`
  formulation unless allocation or cache profiles put wide-route `A^T`
  materialization back in the top linalg hotspot list and a same-worker A/B run
  beats the materialized route by at least 10% without rank, singular-value,
  min-norm, residual, or public `lstsq` tolerance drift.

## 2026-06-19 - frankenscipy-u0ucw - Wide pinv normal-equation Cholesky TRSM

- Agent: cod-a / MistyBirch
- Lever: add a full-row-rank wide `pinv` route using
  `A^+ = A^T (A A^T)^-1`: form the small row Gram from row-major input, solve
  the identity RHS through a new 8-wide batched Cholesky TRSM helper, stream the
  final `A^T G^-1` multiply without materializing `A^T`, and certify
  `A A^+ ~= I_rows` before accepting.
- Status: MEASURED KEEP in the 2026-06-19 gauntlet section above. The
  same-host SciPy oracle row measured the current wide route at 39.51x faster
  than SciPy and 2.82x faster than the in-crate SVD fallback on 500x1000
  full-row-rank input.
- Correctness guard: `wide_pinv_cholesky_matches_svd_reference` locks the new
  route against the SVD pseudo-inverse tolerance contract, while
  `wide_pinv_helpers_match_materialized_products` checks identity-RHS Cholesky
  solve, streamed `A^T G^-1`, and streaming left-inverse certificate against
  materialized nalgebra products.
- Benchmark guard: same-binary Criterion group `u0ucw_wide_pinv` compares
  `_normal_equation_cholesky` against `_svd_fallback` on 500x1000 and
  1000x2000 full-row-rank wide workloads.
- Retry condition: keep only if same-worker `u0ucw_wide_pinv` timings improve
  without rank, pseudo-inverse tolerance, Moore-Penrose left-inverse, or public
  `pinv` certificate drift; if the normal-equation Cholesky route is
  neutral/slower because the left-inverse certificate or `A^T G^-1` multiply
  dominates, reject this exact wide-`pinv` formulation and do not retry unless
  profiles put wide full-row-rank SVD work back in the top linalg hotspot list.

## 2026-06-19 - frankenscipy-u0ucw - Wide pinv diagonal rcond gate

- Agent: cod-a / MistyBirch
- Lever: remove the pre-Cholesky symmetric eigensolve from the full-row-rank
  wide `pinv` route. `pinv` does not expose singular values, so the fast path
  now uses the same diagonal rcond sanity estimate as the tall Cholesky route,
  then relies on Cholesky plus the existing streaming `A A^+ ~= I_rows`
  certificate as the fail-closed acceptance test. The old eigenspectrum gate
  stays behind `FSCI_DISABLE_WIDE_PINV_DIAG_RCOND_GATE` and
  `DISABLE_WIDE_PINV_DIAG_RCOND_GATE` for same-binary A/B.
- Status: MEASURED KEEP in the 2026-06-19 gauntlet section above. The
  same-host Criterion run measured the diagonal rcond gate at 1.40x faster than
  the old eigenspectrum rcond gate and 39.51x faster than SciPy on 500x1000
  full-row-rank input.
- Correctness guard:
  `wide_pinv_diag_rcond_gate_matches_eigen_gate_and_rejects_rank_loss` compares
  the diagonal-gated and eigengated pseudo-inverse outputs on a full-row-rank
  wide matrix and verifies an exact rank-deficient wide matrix still rejects.
- Benchmark guard: same-binary Criterion group `u0ucw_wide_pinv` now compares
  `_normal_equation_cholesky`, `_eigen_rcond_gate`, and `_svd_fallback` on
  500x1000 and 1000x2000 full-row-rank wide workloads.
- Retry condition: keep only if same-worker `u0ucw_wide_pinv` timings improve
  versus `_eigen_rcond_gate` without rank, pseudo-inverse tolerance,
  left-inverse certificate, or rcond-estimate audit regressions; if the
  eigenspectrum was not a measurable cost, reject this exact diagonal-rcond
  shortcut and do not retry unless profiles put `AA^T` symmetric eigensolve
  back in the top wide-`pinv` hotspot list.

## 2026-06-19 - frankenscipy-8l8r1.116 - FFT CSD rfft real-spectrum route

- Agent: cod-a / MistyBirch
- Lever: route `fsci_fft::cross_spectral_density` real input pairs through
  two `rfft` calls instead of promoting both inputs to complex buffers and
  running two full complex FFTs before taking the one-sided half.
- Status: measured REJECT and reverted. The rfft route won the large internal
  row but regressed the 4096 row and remained slower than the fastest
  equivalent SciPy `scipy.fft.rfft` cross-spectrum formula on both measured
  sizes. The release route is restored to the full-complex implementation while
  keeping the Criterion CSD rows, SciPy oracle script, and full-complex
  equivalence guard.
- Artifact:
  `tests/artifacts/perf/2026-06-19-fft-csd-gauntlet/csd_rfft_reject.json`
- Optimization commit under test: `d027c140bc8a937877e8c018cf7265d1f4bc5049`
  vs parent `55f32c99f69991f4ab252621dd86948dc6e95b20`.

Same-worker internal A/B (`hz1`, Criterion mean; parent scratch had only the
same CSD bench harness added, with parent library code unchanged):

| Workload | Parent full-complex mean | Candidate rfft mean | Candidate time vs parent | Verdict |
| --- | ---: | ---: | ---: | --- |
| `fft_helpers/cross_spectral_density/4096` | 112.08 us | 125.88 us | 1.123x | loss |
| `fft_helpers/cross_spectral_density/65536` | 4.9543 ms | 2.3509 ms | 0.475x | win |

Local original-SciPy oracle (`python3 docs/perf_oracle_fft_csd.py --reps 120
--warmups 5`, Python 3.13.7 / NumPy 2.4.3 / SciPy 1.17.1):

| Workload | SciPy rfft-formula p50 | Candidate Rust mean | SciPy/Rust | Verdict |
| --- | ---: | ---: | ---: | --- |
| `cross_spectral_density/4096` | 72.091 us | 125.88 us | 0.573x | Rust slower |
| `cross_spectral_density/65536` | 1.653584 ms | 2.3509 ms | 0.703x | Rust slower |

- Correctness/conformance guards to re-run in this closeout:
  - `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a RUSTFLAGS='-C force-frame-pointers=yes' cargo check -p fsci-fft --all-targets`
  - `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a RUSTFLAGS='-C force-frame-pointers=yes' cargo test -p fsci-fft cross_spectral_density -- --nocapture`
  - `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a RUSTFLAGS='-C force-frame-pointers=yes' cargo clippy -p fsci-fft --all-targets -- -D warnings`
  - `python3 docs/perf_oracle_fft_csd.py --reps 120 --warmups 5`
  - `git diff --check` and `ubs` on the changed file set.
- Retry condition: do not retry this standalone rfft CSD route. Reopen only if
  a fresh same-worker profile attributes a clearly-above-noise share to full
  complex promotion inside `cross_spectral_density` on a workload of at least
  65536 samples, and the replacement beats the fastest equivalent SciPy rfft
  formula while not regressing the 4096 row.

## 2026-06-18 - frankenscipy-va60h - MDS streamed double-centering

- Agent: cod-a / MistyBirch
- Lever: remove internal full squared-distance materializations from
  `classical_mds`, `landmark_mds`, and `landmark_isomap` by streaming the
  double-centering formula from squared-distance callbacks; stream landmark
  triangulation dot products instead of allocating a per-point `dshift` vector.
- Status: pending batch-test. This is a code-first commit per campaign
  instruction; only local `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a
  cargo check -p fsci-cluster` is expected before commit.
- Correctness guard: `double_centered_gram_matches_materialized_delta_reference`
  locks the streamed centering helper against the materialized formula, while
  existing MDS and Isomap recovery tests preserve embedding-distance behavior.
- Benchmark guard: compare focused `classical_mds`, `landmark_mds`, and
  `landmark_isomap` workloads against the previous same-worker commit,
  emphasizing large n with small k/m where allocation traffic competes with the
  randomized eigensolve.
- Retry condition: keep only if same-worker MDS/Isomap timings improve without
  embedding distance, eigenvalue ordering, or validation drift; if recomputing
  squared distances beats no allocation only in noise or regresses, reject this
  streaming double-centering formulation and do not retry unless allocation
  profiles again show `D^2`/`Delta` or `dshift` churn as a top cluster hotspot.

## 2026-06-18 - frankenscipy-acoco - jnjnp_zeros bracket reuse

- Agent: cod-a / MistyBirch
- Lever: reuse the per-order `J_n` zero sequence already computed by
  `jnjnp_zeros` when bracketing `J_n'` roots, avoiding a duplicate
  `jn_zeros(n, per)` call for every positive order.
- Status: MEASURED INTERNAL KEEP / SCIPY LOSS. The bracket-reuse route is
  faster than the benchmark-only recreation of the previous duplicate
  bracketing route, so it stays in tree. The full routine is still much slower
  than original SciPy and must route to a deeper algorithmic optimization.
- Artifact:
  `tests/artifacts/perf/2026-06-19-acoco-jnjnp-zeros-gauntlet/`.
- Head-to-head result, Criterion point-estimate means:

  | Workload | Rust current | Rust legacy duplicate route | SciPy original | Current vs SciPy | Current vs legacy |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | `jnjnp_zeros(nt=64)` | 80.728603 ms | 101.762454 ms | 0.493655 ms | 163.53x slower | 1.26x faster |
  | `jnjnp_zeros(nt=128)` | 410.059973 ms | 544.006333 ms | 0.924456 ms | 443.57x slower | 1.33x faster |
- Correctness guard: `derivative_bessel_zeros_match_scipy_reference_points`
  passed via rch, and the existing
  `jnyn_and_jnjnp_zeros_match_scipy` SciPy anchor also passed.
- Build/format guard: `cargo check -p fsci-special --benches` passed via rch;
  `rustfmt --edition 2024 --check crates/fsci-special/benches/special_bench.rs`
  passed.
- Blocked lint guard: `cargo clippy -p fsci-special --benches -- -D warnings`
  failed before this benchmark file on existing dependency lints in
  `fsci-integrate` and `fsci-linalg`.
- Retry condition: do not retry this exact duplicate-`jn_zeros` lever without a
  fresh profile. Future work should target SciPy's much faster zero
  enumeration/root-finding strategy or another measured special-function
  hotspot.

## 2026-06-19 - frankenscipy-nm8ex - dim-4 pdist SIMD-across-pairs (SoA) — GAP CLOSED, now FASTER than SciPy

- Agent: cod-a / MistyBirch
- Lever: transpose dim-4 points to SoA coordinate columns and process L=8 pairs
  per SIMD chunk (lane k = pair (i, start+j+k)). The dependent per-pair `sqrt`
  (Euclidean) and `divide` (Cosine) — the two bottlenecks that serialized the
  prior one-pair-at-a-time kernels — pipeline across lanes via `vsqrtpd`/`vdivpd`.
- Status: **MEASURED WIN vs SciPy** (flips all four prior nm8ex/nm8ex.1 losses).
  Supersedes the nm8ex serial-gate and nm8ex.1 flat-row-staging internal keeps.
- Bit-identity: per lane the squared-sum+sqrt and `1 - dot/(ni·nj)` with the
  `denom==0 ⇒ NaN` select run in the exact left-to-right order of the scalar
  `sqeuclidean4`/`pair` helpers. `pdist_dim4_fast_paths_match_metric_helpers`
  (to_bits equality) and `pdist_parallel_is_bit_identical` pass; 206 fsci-spatial
  lib tests green; `clippy -p fsci-spatial --lib -D warnings` clean.
- Head-to-head, same-box (local) criterion median + scipy 1.17:

  | Workload | Rust before | Rust after | SciPy | After vs SciPy | Self-speedup |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | `pdist/euclidean/256` | 82.5 us | 41.0 us | 85.3 us | 2.08x faster | 2.01x |
  | `pdist/cosine/256` | 93.4 us | 34.8 us | 76.8 us | 2.21x faster | 2.68x |
  | `pdist/euclidean/512` | 340 us | 162.6 us | 303 us | 1.86x faster | 2.09x |
  | `pdist/cosine/512` | 369 us | 145 us | 275 us | 1.90x faster | 2.54x |
- Commit: `595eceb5`.
- Retry/extend condition: the same SoA SIMD-across-pairs shape applies to the
  parallel (`nthreads>1`, n>512) dim-4 segments and to other small-fixed-dim
  pdist/cdist metrics (sqeuclidean, cityblock) still on the scalar per-pair path.

## 2026-06-19 - frankenscipy-nm8ex - dim-4 pdist parallel vectorization + serial gate raised to n<=2048

- Agent: cod-a / MistyBirch
- Follow-up to `595eceb5`. (1) Extended the SoA SIMD-across-pairs kernel to the
  parallel (n>2048) workers via shared `fill_euclidean4_rows`/`fill_cosine4_rows`;
  new `pdist_dim4_parallel_matches_metric_helpers` gate (n=600, to_bits) covers
  the `r0>0` mid-triangle offset. (2) Raised the dim-4 Euclidean/Cosine serial
  guard 512→2048: the vectorized serial kernel is compute-bound and beats SciPy
  ~1.5-2.2x to n≈2048, below which the parallel path lost to its own ~64-thread
  spawn cost (n=1024: serial 0.67ms vs parallel **2.83ms** — was 2.4x SLOWER than
  SciPy). Above 2048 the memory-bound O(n²) work amortizes the spawn (n=4096:
  parallel 13ms vs serial 43ms vs SciPy 51ms).
- Status: **MEASURED WIN at every size** (same-box criterion median vs scipy 1.17):

  | n | Rust eucl | SciPy eucl | eucl ratio | Rust cos | SciPy cos | cos ratio |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | 256 | 48.5 us | 85.3 us | 1.76x | 40.7 us | 76.8 us | 1.89x |
  | 512 | 167 us | 303 us | 1.82x | 148 us | 275 us | 1.86x |
  | 1024 | 666 us | 1.20 ms | 1.80x | 549 us | 1.08 ms | 1.97x |
  | 2048 | 2.86 ms | 4.65 ms | 1.63x | 2.53 ms | 4.23 ms | 1.67x |
  | 4096 | 13.2 ms | 51.1 ms | **3.87x** | 13.1 ms | 48.3 ms | **3.69x** |
- Commits: `595eceb5` (serial kernel), `aeb5e0c8` (parallel + gate).
- Note: the residual `cdist_thread_count` over-spawn (CrimsonForge's finding,
  `work < 1<<18` spawns ~64 threads) still affects the generic cdist path (no
  bench); coordinate before retuning that shared gate.

## 2026-06-19 - frankenscipy-nm8ex - dim-4 cdist SoA SIMD + 16-thread cap — closes 2.4-2.6x scipy loss to parity/1.15x WIN

- Agent: cod-a / MistyBirch
- cdist Euclidean/Cosine at d=4 had NO fast path (generic scalar `metric_distance`
  arm) AND over-spawned ~64 threads for memory-bandwidth-bound work: na=nb=1000
  d=4 was 5.7ms vs scipy 2.2ms (2.6x SLOWER).
- Lever 1: dim-4 fast paths vectorize each output ROW across xb columns (SoA, lane
  k = column j+k) so the dependent per-pair sqrt/divide pipeline across SIMD lanes
  (same lever as pdist). BIT-identical to the generic arm at d=4.
- Lever 2 (the unlock): cap the dim-4 cdist parallel path at **16 workers**. The
  kernel is bandwidth-bound (~40 bytes traffic per 8-byte output), so ~16 threads
  saturate memory bandwidth; 32/64 only contend. Measured cap sweep on na=nb=1000:
  cap8 2.2ms, **cap16 1.8-2.0ms**, cap32 3.7ms, cap64 4.6ms. This is the cdist
  analogue of pdist's over-spawn finding (CrimsonForge's `cdist_thread_count`).
- Status: **MEASURED WIN** (was 2.4-2.6x loss → now parity-to-1.15x faster):

  | Workload | Rust before | Rust after | SciPy | After vs SciPy |
  | --- | ---: | ---: | ---: | ---: |
  | cdist eucl 1000×1000 d4 | 5.72 ms | 1.99 ms | 2.17 ms | 1.09x faster |
  | cdist eucl 2000×500 d4 | 5.17 ms | 2.23 ms | 2.19 ms | parity |
  | cdist cos 1000×1000 d4 | — | 1.76 ms | 2.03 ms | 1.15x faster |
  | cdist cos 2000×500 d4 | — | 2.20 ms | 2.03 ms | parity |
- Gate `cdist_dim4_fast_paths_match_metric_distance` (serial+parallel+zero-norm NaN,
  to_bits). 208 lib tests green; clippy -D warnings (lib+bins) clean. Commit `02e057ce`.
- REJECTED sub-lever: `with_capacity`+`extend_from_slice` single-write (to avoid the
  `vec![0.0;nb]` zero-init) REGRESSED 4.56→5.59ms — `alloc_zeroed` gives lazy zero
  pages (~free) and `copy_to_slice` is a direct store, while `extend` adds per-chunk
  Vec-growth overhead. Kept `vec![0.0;nb]` + `copy_to_slice`.

## 2026-06-19 - frankenscipy-nm8ex - pdist thread gate (all metrics) + dim-4 SqEuclidean/Cityblock SoA

- Agent: cod-a / MistyBirch
- Found via metric×dim sweep (new perf_pdist_sweep bin): cityblock/sqeuclidean/
  chebyshev at d=4 were ~13x SLOWER than scipy (2.3-2.5ms vs 0.18ms) — only
  Euclidean/Cosine d4 had a serial guard; everything else hit cdist_thread_count's
  1<<18 gate and over-spawned ~64 threads (~2.4ms) for sub-ms work.
- Fix 1 (byte-identical, broad): `pdist_thread_count(n,dim)` — serial unless
  pairs·dim ≥ 1<<20, then cap at 16 (bandwidth-bound kernels). Strict improvement
  over the shipped 64-thread path at every measured size (d16 chebyshev 2.48→1.26ms,
  d64 euclidean 2.49→1.31ms, d4 metrics 2.4→0.7ms serial). Commit `bd09f5d6`.
- Fix 2 (WIN): dim-4 SqEuclidean/Cityblock SoA SIMD-across-pairs fast paths (shared
  `pdist_fill_dim4` wrapper). Bit-identical at d=4. Commit `0278e4b3`:

  | Workload | before | after | scipy | vs scipy |
  | --- | ---: | ---: | ---: | ---: |
  | pdist sqeuclidean d4 n512 | 2.47ms | 0.102ms | 0.179ms | 1.75x faster |
  | pdist cityblock d4 n512 | 2.49ms | 0.109ms | 0.190ms | 1.74x faster |
- Bit-identity gates extended (serial n=32 + parallel n=2100); parallel coverage
  tests bumped (n=1100 d2 / n=2100 d4) to stay above the raised 1<<20 gate.
- DEFERRED: chebyshev d4 (0.79ms vs scipy 0.18ms) — its NaN-propagating max fold
  needs careful SIMD replication (f64::max doesn't propagate NaN; the helper forces
  it). Left on the gate-fixed generic path.

## 2026-06-20 - frankenscipy-9l5oo - Delaunay circumcircle grid closes large-n SciPy loss to parity

- Agent: cod-b / MistyBirch
- Baseline/routing: expanding `bench_delaunay` from n=1000/2000 to
  n=1000/2000/4000/8000 showed the original issue text was stale at small sizes
  but still real at larger sizes. Pre-grid rch probe on `hz1`: n=1000 1.0832 ms,
  n=2000 3.9718 ms, n=4000 14.935 ms, n=8000 55.761 ms. Local SciPy 1.17.1 oracle:
  n=1000 1.93258 ms, n=2000 4.54974 ms, n=4000 9.50086 ms, n=8000 20.62714 ms.
  Result before this lever: wins at 1000/2000, losses of 1.57x/2.70x at 4000/8000.
- Lever: for n>=4096, keep stable triangle IDs and index each active
  circumcircle's bounding box into a fixed grid. Each inserted point checks only
  the candidate circles in its cell, then runs the exact `dist2 < r2` predicate.
  Removed triangles become inactive; stale grid IDs are skipped. If a grid lookup
  ever returns no bad triangle, the code falls back to the full active scan.
- Final head-to-head (`cargo bench -p fsci-spatial --bench spatial_bench -- delaunay
  --sample-size 10 --measurement-time 1 --warm-up-time 1` via rch `ovh-a`; SciPy
  oracle local):

  | n | Rust after | SciPy | Ratio vs SciPy | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | 1000 | 754.03 us | 1.93258 ms | 2.56x faster | win |
  | 2000 | 2.6129 ms | 4.54974 ms | 1.74x faster | win |
  | 4000 | 9.4632 ms | 9.50086 ms | parity | neutral |
  | 8000 | 20.622 ms | 20.62714 ms | parity | neutral |

- Score vs SciPy: **2 wins / 0 losses / 2 neutral**. The measured large-n losses
  are closed to parity on the scoped deterministic 2-D workload. Remaining
  negative evidence: this is still Bowyer-Watson with a fixed-grid candidate
  accelerator, not a full Qhull-class randomized incremental/history-DAG locate;
  re-test beyond n=8000 before claiming asymptotic dominance.
- Gates: `cargo test -p fsci-spatial delaunay -- --nocapture` 8 unit + 1
  metamorphic pass; full `cargo test -p fsci-spatial --lib -- --nocapture` 208
  passed / 0 failed / 2 ignored; `cargo check -p fsci-spatial --all-targets`;
  `cargo clippy -p fsci-spatial --all-targets --no-deps -- -D warnings`;
  `cargo fmt --check -p fsci-spatial`; `ubs` on touched files; and
  `cargo test -p fsci-conformance --test e2e_spatial -- --nocapture` 16/0.

## 2026-06-20 - frankenscipy-2hclc - public CSR SpMV cached+unrolled row sweep closes scale loss

- Agent: cod-b / MistyBirch
- Starting point: the original Krylov-owned allocation issue had already been
  mostly completed in `linalg.rs` (`csr_matvec_into` is used by CG/GMRES/BiCGSTAB/
  LSMR/LSQR/SVDS-style paths), and `frankenscipy-fo9cj` already rejected the
  deeper Arnoldi arena/mutable scratch route. The live measured sparse loss was
  the public `spmv_csr` row loop in `ops.rs`, where the ledger showed 1 win /
  2 losses vs SciPy.
- Lever: cache `shape/indptr/indices/data` once, use an index row loop, and
  unroll the per-row multiply-add by 4. This preserves left-to-right row
  accumulation order and keeps the public API allocation behavior unchanged.
- Same-process old/current proof on rch `ovh-a`:
  `env FSCI_PUBLIC_SPMV_AB=1 cargo run --profile release-perf -p fsci-sparse
  --bin perf_csr_matvec`. The perf binary runs the legacy public row sweep and
  current `spmv_csr` in one process and checks `to_bits` equality.

  | n | nnz | legacy row sweep | current | Internal ratio | Identity |
  | --- | ---: | ---: | ---: | ---: | --- |
  | 100 | 500 | 550 ns | 356 ns | 1.54x faster | true |
  | 1000 | 10000 | 12.074 us | 5.741 us | 2.10x faster | true |
  | 10000 | 100000 | 135.043 us | 63.231 us | 2.14x faster | true |

- Head-to-head vs SciPy (`docs/perf_oracle_spmv.py`, local SciPy 1.17.1; Rust
  `cargo bench -p fsci-sparse --bench sparse_bench -- sparse_spmv --sample-size
  10 --measurement-time 1 --warm-up-time 1` via rch `hz2` after the lever):

  | n | nnz | Rust after | SciPy | Ratio vs SciPy | Verdict |
  | --- | ---: | ---: | ---: | ---: | --- |
  | 100 | 500 | 387.54 ns | 4.63 us | 11.95x faster | win |
  | 1000 | 10000 | 7.077 us | 8.00 us | 1.13x faster | win |
  | 10000 | 100000 | 68.820 us | 96.95 us | 1.41x faster | win |

- Score vs SciPy: **3 wins / 0 losses / 0 neutral**. Prior ledger status was
  **1 win / 2 losses / 0 neutral**; the two scale losses are closed on the
  scoped public CSR SpMV workload.
- Gates: `cargo bench -p fsci-sparse --bench sparse_bench -- sparse_spmv`
  via rch; same-process A/B via rch; `cargo test -p fsci-sparse spmv -- --nocapture`
  via rch (5 unit/property + 4 metamorphic pass); `cargo check -p fsci-sparse
  --all-targets` via rch; `cargo clippy -p fsci-sparse --all-targets --no-deps
  -- -D warnings` via rch; sparse conformance
  `cargo test -p fsci-conformance --test diff_sparse spmv -- --nocapture`
  via rch (11/0); touched-file `rustfmt --edition 2024 --check`;
  `git diff --check`; UBS on touched Rust files (0 critical, existing warnings).
- Remaining route: explicit SIMD or sparse-BLAS-style row blocking only if fresh
  profiles still show public SpMV as a top gap. Do not retry the rejected
  `frankenscipy-fo9cj` row-major Arnoldi arena/scratch family without new
  allocation-profile proof.
