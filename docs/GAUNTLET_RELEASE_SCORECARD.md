# Gauntlet Release Scorecard

Last updated: 2026-06-19 by cod-a/cod-b / MistyBirch.

This scorecard tracks code-first performance work that has been converted into
measured head-to-head evidence against the SciPy original. The detailed
win/loss/neutral ledger lives in `docs/progress/perf-negative-results.md`.

## Measured Keeps

| Bead | Cluster | Realistic workload | Rust result | SciPy result | Ratio | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `frankenscipy-8l8r1.118` | Fused signal coherence | `scipy.signal.coherence`, 65536 samples, Hann window 1024/512 overlap | 2.191980 ms | 18.961613 ms | 8.65x faster | keep |
| `frankenscipy-u0ucw` | Wide `pinv` Cholesky TRSM + diagonal rcond gate | 500x1000 full-row-rank dense `scipy.linalg.pinv` equivalent | 183.699926 ms | 7.257573 s | 39.51x faster | keep |
| `frankenscipy-u0ucw` | Wide `lstsq` current materialized normal equations after row-stream revert | 500x1000 full-row-rank dense `scipy.linalg.lstsq` equivalent | 109.369915 ms | 1.253347 s | 11.46x faster | keep current, reject row-stream lever |

## Measured Losses / Internal Keeps

| Bead | Cluster | Realistic workload | Rust result | SciPy result | Ratio | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `frankenscipy-96n2y` | `jnjnp_zeros` tighter frontier seed | `scipy.special.jnjnp_zeros(nt=64)` equivalent | 2.2230 ms | 424.10 us | 5.24x slower | keep 2.12x internal win; route deeper |
| `frankenscipy-96n2y` | `jnjnp_zeros` tighter frontier seed | `scipy.special.jnjnp_zeros(nt=128)` equivalent | 6.1605 ms | 799.97 us | 7.70x slower | keep 1.38x internal win; route deeper |
| `frankenscipy-x9ckc` | `jnjnp_zeros` guarded root-cost refinement | `scipy.special.jnjnp_zeros(nt=64)` equivalent | 4.6666 ms | 439.49 us | 10.62x slower | keep 1.17x internal win; route deeper |
| `frankenscipy-x9ckc` | `jnjnp_zeros` guarded root-cost refinement | `scipy.special.jnjnp_zeros(nt=128)` equivalent | 8.3620 ms | 787.18 us | 10.62x slower | keep 1.20x internal win; route deeper |
| `frankenscipy-01lxz` | `jnjnp_zeros` output-sensitive frontier | `scipy.special.jnjnp_zeros(nt=64)` equivalent | 4.3372 ms | 486.57 us | 8.91x slower | keep 17.82x internal win; route deeper |
| `frankenscipy-01lxz` | `jnjnp_zeros` output-sensitive frontier | `scipy.special.jnjnp_zeros(nt=128)` equivalent | 7.5415 ms | 792.81 us | 9.51x slower | keep 50.77x internal win; route deeper |
| `frankenscipy-acoco` | `jnjnp_zeros` bracket reuse | `scipy.special.jnjnp_zeros(nt=64)` equivalent | 80.728603 ms | 0.493655 ms | 163.53x slower | keep internal bracket reuse; route deeper |
| `frankenscipy-acoco` | `jnjnp_zeros` bracket reuse | `scipy.special.jnjnp_zeros(nt=128)` equivalent | 410.059973 ms | 0.924456 ms | 443.57x slower | keep internal bracket reuse; route deeper |
| `frankenscipy-nm8ex` | Spatial `pdist` dim-4 fast path + serial gate | Euclidean n=256 d=4 | 107.45 us | 94.30 us | 1.14x slower | keep 27.24x internal win; route deeper |
| `frankenscipy-nm8ex` | Spatial `pdist` dim-4 fast path + serial gate | Cosine n=256 d=4 | 114.04 us | 87.20 us | 1.31x slower | keep 29.51x internal win; route deeper |
| `frankenscipy-nm8ex` | Spatial `pdist` dim-4 fast path + serial gate | Euclidean n=512 d=4 | 425.75 us | 310.83 us | 1.37x slower | keep 8.35x internal win; route deeper |
| `frankenscipy-nm8ex` | Spatial `pdist` dim-4 fast path + serial gate | Cosine n=512 d=4 | 461.16 us | 283.75 us | 1.63x slower | keep 5.54x internal win; route deeper |
| `frankenscipy-va60h` | Linkage flat row-major distance arena | `scipy.cluster.hierarchy.linkage(method="average")`, n=800 d=4 | 6.1713 ms | 4.4550 ms | 1.385x slower | keep internal flat arena; route deeper |
| `frankenscipy-va60h` | Linkage flat row-major distance arena | `scipy.cluster.hierarchy.linkage(method="ward")`, n=800 d=4 | 7.5250 ms | 5.0256 ms | 1.497x slower | keep internal flat arena; route deeper |

## Measured Rejects

| Bead | Rejected lever | Realistic workload | Candidate result | Parent/current result | Decision |
| --- | --- | --- | ---: | ---: | --- |
| `frankenscipy-8l8r1.122` | L-BFGS-B mutable Wolfe finite-difference probe scratch | 10D Rosenbrock finite-difference `L-BFGS-B` | 106.440 us | 87.087 us parent | reject and revert |
| `frankenscipy-8l8r1.122` | L-BFGS-B mutable Wolfe finite-difference probe scratch | 32D quadratic finite-difference `L-BFGS-B` | 6.055 us | 5.246 us parent | reject and revert |
| `frankenscipy-8l8r1.116` | FFT CSD rfft real-spectrum route | 4096-sample CSD helper | 125.88 us | 112.08 us parent | reject and revert |
| `frankenscipy-8l8r1.116` | FFT CSD rfft real-spectrum route | 65536-sample CSD helper vs SciPy rfft formula | 2.3509 ms | 1.653584 ms SciPy | reject and revert |
| `frankenscipy-acdq2` | `gaussian_filter1d` always-line-walk plus outermost row-split/direct interior taps | `ndimage` gaussian sigma=2, 256x256 | 4.2236 ms | 2.4792 ms clean current; prior ledger current 3.238 ms | reject and revert |

## Internal Regression Gates

| Bead | Current route | Superseded route | Mean delta | Decision |
| --- | --- | --- | ---: | --- |
| `frankenscipy-u0ucw` | Cholesky + diagonal rcond gate | Cholesky + eigenspectrum rcond gate | 1.40x faster | keep current |
| `frankenscipy-u0ucw` | Cholesky + diagonal rcond gate | SVD fallback | 2.82x faster | keep current |
| `frankenscipy-u0ucw` | Wide `lstsq` materialized `A^T` | Row-streamed `A A^T` + `A^T y` | 1.035x faster | revert row-streaming |
| `frankenscipy-8l8r1.122` | Parent-style `line_search_wolfe2` gradient closure | Mutable `line_search_wolfe2_with_gradient_probe` path | 1.222x faster on 10D Rosenbrock; 1.154x faster on 32D quadratic | revert mutable-probe route |
| `frankenscipy-01lxz` | `jnjnp_zeros` output-sensitive frontier | Fixed `nt + 2` by `nt + 2` candidate rectangle | 17.82x faster at `nt=64`; 50.77x faster at `nt=128` | keep current despite SciPy loss |
| `frankenscipy-acoco` | `jnjnp_zeros` bracket reuse | Pre-optimization duplicate `jnp_zeros` bracketing route | 1.26x faster at `nt=64`; 1.33x faster at `nt=128` | keep current despite SciPy loss |
| `frankenscipy-96n2y` | `jnjnp_zeros` tighter frontier seed and dimension-specific expansion | Guarded root-cost route from `frankenscipy-x9ckc` | 2.12x faster at `nt=64`; 1.38x faster at `nt=128` on same-worker `hz1` | keep current despite SciPy loss |
| `frankenscipy-8l8r1.116` | Full-complex CSD route | rfft CSD route | 1.123x faster on 4096; rfft wins 2.107x on 65536 but loses to SciPy rfft oracle | revert rfft route |
| `frankenscipy-va60h` | Flat row-major linkage arena | Legacy nested-row NN-array helper | 1.128x faster on Average; 1.019x faster on Ward | keep current despite SciPy loss |
| `frankenscipy-va60h` | Flat row-major linkage arena | Reverted production nested route probe | 1.290x faster on Average; 1.251x faster on Ward | undo revert; keep flat route |
| `frankenscipy-8l8r1.118` | Fused signal coherence | Compositional `csd(x,y)` + `csd(x,x)` + `csd(y,y)` route | 2.98x faster locally; 2.80x faster on rch `hz1` | keep fused route |
| `frankenscipy-nm8ex` | Dim-4 Euclidean/Cosine direct serial `pdist` kernels | Generic metric-dispatch/reduction/threaded path | 5.54-29.51x faster on same-worker `hz2` | keep current despite SciPy loss |
| `frankenscipy-x9ckc` | `jnjnp_zeros` guarded direct integer-order root polishing | Generic strict-mode bracketed zero route after output frontier | 1.17x faster at `nt=64`; 1.20x faster at `nt=128` on same-worker `hz1` | keep current despite SciPy loss |

## Current Readiness

| Area | Status | Evidence |
| --- | --- | --- |
| Signal coherence performance | measured keep | fused Rust coherence is 8.65x faster than `scipy.signal.coherence` on the 65536-sample Hann 1024/512 workload |
| Signal coherence correctness | guarded | focused `coherence_matches_scipy_reference` and `coherence_matches_compositional_csd_formula` tests passed via rch |
| Wide `pinv` performance | measured keep | Criterion mean point estimate vs SciPy 1.17.1 oracle, 39.51x faster |
| Wide `pinv` correctness | guarded | targeted `fsci-linalg` tests cover the diagonal gate, Cholesky route, helper products, and SciPy reference values |
| Wide `lstsq` performance | measured keep plus internal reject | current materialized path is 11.46x faster than SciPy; row-streamed lever was 0.966x vs materialized and was reverted |
| Wide `lstsq` correctness | guarded | `public_wide_min_norm_lstsq_route_perf_probe` passed in release with max abs diff `3.38840067115597776e-13` |
| L-BFGS-B performance | measured reject plus current-route keep | mutable Wolfe probe scratch regressed same-worker rows; reverted route is still 173.78-229.23x faster than SciPy on measured callback workloads |
| L-BFGS-B correctness | guarded | `fsci-opt lbfgsb` tests and SciPy-backed `diff_opt_lbfgsb_minimize` conformance passed after revert |
| FFT CSD performance | measured reject | rfft route regressed 4096 internally and was 1.42-1.75x slower than the equivalent SciPy rfft formula; full-complex route restored |
| FFT CSD correctness | guarded | full-complex equivalence guard retained; final fsci-fft gates recorded in `docs/progress/perf-release-readiness-scorecard.md` |
| `fsci-opt` lint/build gate | guarded | `cargo check -p fsci-opt --all-targets`, `cargo fmt -p fsci-opt --check`, and `cargo clippy -p fsci-opt --all-targets -- -D warnings` passed |
| `fsci-special` `jnjnp_zeros` performance | measured loss plus internal keep | tighter frontier seed is 2.12x faster at `nt=64` and 1.38x faster at `nt=128` than the guarded root-cost route on same-worker `hz1`, but still 5.24-7.70x slower than SciPy |
| `fsci-special` `jnjnp_zeros` correctness | guarded | `jnjnp_adaptive_envelope_matches_oversized_reference`, `jnjnp_frontier_matches_scipy_bench_cutoffs`, and `jnyn_and_jnjnp_zeros_match_scipy` passed via rch |
| `fsci-special` lint/build gate | partial | `cargo check -p fsci-special --all-targets` passed; clippy `-D warnings` stopped on existing `fsci-integrate`/`fsci-linalg` dependency lints; broad rustfmt/touched-file rustfmt remain blocked by pre-existing formatting drift outside this patch |
| `fsci-fft` lint/build gate | guarded | `cargo check -p fsci-fft --all-targets`, focused CSD/rfft tests, and `cargo clippy -p fsci-fft --all-targets -- -D warnings` passed; broad rustfmt remains blocked by pre-existing file-wide drift |
| `fsci-cluster` linkage performance | measured loss plus internal keep | flat arena is 1.128x faster than the legacy nested helper on Average and 1.019x on Ward, but 1.385-1.497x slower than SciPy |
| `fsci-cluster` linkage correctness | guarded | filtered linkage tests passed via rch (28 unit, 9 metamorphic); SciPy-backed `diff_cluster_linkage_from_distances` conformance passed locally |
| `fsci-cluster` lint/build gate | partial | `cargo check -p fsci-cluster --benches` passed; fmt blocked on existing `perf_isomap.rs` drift and clippy blocked on existing `fsci-linalg` dependency lints |
| `fsci-ndimage` gaussian_filter route | measured reject | always-line-walk plus outermost row-split regressed `gaussian_sigma2/256`; source reverted, current remains a SciPy loss routed to inner dot SIMD/tiled contiguous-span work |
| `fsci-spatial` `pdist` performance | measured loss plus internal keep | dim-4 direct serial kernels are 5.54-29.51x faster than the generic threaded route on same-worker `hz2`, but still 1.14-1.63x slower than SciPy |
| `fsci-spatial` `pdist` correctness | guarded | focused `pdist` tests passed via rch (10 passed), full `fsci-spatial` lib suite passed via rch (206 passed, 2 ignored) |
| `fsci-spatial` lint/build gate | partial | `cargo check -p fsci-spatial --all-targets` passed; conformance is blocked by unrelated `e2e_sparse` compile error, clippy by existing `fsci-linalg` lints, and fmt by pre-existing `fsci-spatial` rustfmt drift |
| rch SciPy oracle parity | blocked on worker image | `vmi1152480` and `vmi1227854` lacked `scipy`; local same-host oracle supplied the head-to-head ratios |
| Release readiness | partial | two linalg perf clusters, one `fsci-signal` coherence win, one `fsci-opt` L-BFGS-B reject, one `fsci-special` measured SciPy-loss/internal-keep frontier win, one `fsci-fft` CSD reject, one `fsci-cluster` linkage measured SciPy loss, one `fsci-ndimage` gaussian reject, and one `fsci-spatial` `pdist` internal keep verified; other code-first perf ledger entries still need gauntlet conversion |

## Pending Gauntlet Backlog

Continue converting `pending batch-test` entries in the negative-evidence ledger
one cluster at a time. Each conversion must record the SciPy ratio, internal
A/B route deltas, conformance status, and keep/revert decision before release
readiness can be raised.
