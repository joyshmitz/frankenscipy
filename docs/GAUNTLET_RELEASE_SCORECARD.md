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
| `frankenscipy-8l8r1.123` | `jnjnp_zeros` cutoff-driven generator | `scipy.special.jnjnp_zeros(nt=64)` equivalent | 1.5856 ms | 427.47 us | 3.71x slower | keep 2.30x internal win; route deeper |
| `frankenscipy-8l8r1.123` | `jnjnp_zeros` cutoff-driven generator | `scipy.special.jnjnp_zeros(nt=128)` equivalent | 2.9035 ms | 789.23 us | 3.68x slower | keep 2.28x internal win; route deeper |
| `frankenscipy-wm14d` | `ndimage.zoom` 2D linear Reflect fast path | `scipy.ndimage.zoom(256x256, 2x, order=1)` equivalent | 7.9684 ms | 3.88937 ms | 2.05x slower | keep 4.27x internal win; route deeper |
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
| `frankenscipy-nm8ex.1` | Spatial `pdist` dim-4 flat row staging | Euclidean n=256 d=4 | 172.83 us | 88.96 us | 1.94x slower | keep 1.52x internal win; route deeper |
| `frankenscipy-nm8ex.1` | Spatial `pdist` dim-4 flat row staging | Cosine n=256 d=4 | 208.89 us | 79.69 us | 2.62x slower | keep 1.83x internal win; route deeper |
| `frankenscipy-nm8ex.1` | Spatial `pdist` dim-4 flat row staging | Euclidean n=512 d=4 | 714.58 us | 309.79 us | 2.31x slower | keep 1.11x internal win; route deeper |
| `frankenscipy-nm8ex.1` | Spatial `pdist` dim-4 flat row staging | Cosine n=512 d=4 | 828.70 us | 275.14 us | 3.01x slower | keep 1.44x internal win; route deeper |
| `frankenscipy-va60h` | Linkage flat row-major distance arena | `scipy.cluster.hierarchy.linkage(method="average")`, n=800 d=4 | 6.1713 ms | 4.4550 ms | 1.385x slower | keep internal flat arena; route deeper |
| `frankenscipy-va60h` | Linkage flat row-major distance arena | `scipy.cluster.hierarchy.linkage(method="ward")`, n=800 d=4 | 7.5250 ms | 5.0256 ms | 1.497x slower | keep internal flat arena; route deeper |

## Measured Rejects

| Bead | Rejected lever | Realistic workload | Candidate result | Parent/current result | Decision |
| --- | --- | --- | ---: | ---: | --- |
| `frankenscipy-8l8r1.122` | L-BFGS-B mutable Wolfe finite-difference probe scratch | 10D Rosenbrock finite-difference `L-BFGS-B` | 106.440 us | 87.087 us parent | reject and revert |
| `frankenscipy-8l8r1.122` | L-BFGS-B mutable Wolfe finite-difference probe scratch | 32D quadratic finite-difference `L-BFGS-B` | 6.055 us | 5.246 us parent | reject and revert |
| `frankenscipy-8l8r1.116` | FFT CSD rfft real-spectrum route | 4096-sample CSD helper | 125.88 us | 112.08 us parent | reject and revert |
| `frankenscipy-8l8r1.116` | FFT CSD rfft real-spectrum route | 65536-sample CSD helper vs SciPy rfft formula | 2.3509 ms | 1.653584 ms SciPy | reject and revert |
| `frankenscipy-8l8r1.124` | `jnjnp_zeros` top-k candidate partition | `jnjnp_zeros(nt=64)` same-binary probe | 0.911730 ms | 0.928939 ms full-sort | reject: 1.019x near-noise |
| `frankenscipy-8l8r1.124` | `jnjnp_zeros` top-k candidate partition | `jnjnp_zeros(nt=128)` same-binary probe | 1.715855 ms | 1.737776 ms full-sort | reject: 1.013x near-noise |
| `frankenscipy-acdq2` | `gaussian_filter1d` always-line-walk plus outermost row-split/direct interior taps | `ndimage` gaussian sigma=2, 256x256 | 4.2236 ms | 2.4792 ms clean current; prior ledger current 3.238 ms | reject and revert |
| `frankenscipy-fo9cj` | Sparse Arnoldi row-major basis arena plus mutable operator scratch | `eigsh n=2000 k=6` | 1.667 ms | 1.184 ms parent/restored | reject and revert |
| `frankenscipy-fo9cj` | Sparse Arnoldi row-major basis arena plus mutable operator scratch | `eigsh n=8000 k=6` | 6.594 ms | 5.548 ms parent/restored | reject and revert |
| `frankenscipy-fo9cj` | Sparse Arnoldi row-major basis arena plus mutable operator scratch | `eigsh n=20000 k=8` | 16.147 ms | 11.599 ms parent/restored | reject and revert |
| `frankenscipy-fo9cj` | Sparse Arnoldi row-major basis arena plus mutable operator scratch | `svds 2200/8200/20200 sweep` | 1.203 / 4.654 / 12.362 ms | 1.191 / 4.929 / 12.534 ms parent/restored | reject: mixed near-noise cannot offset `eigsh` regression |
| `frankenscipy-bpzha` | Solver-owned RK scratch double-buffer variants | `solve_ivp RK45` Lorenz on rch `hz2`, repeats=3000 | 23.402816 us/call | 21.951172 us/call parent | reject and revert |
| `frankenscipy-bpzha` | Solver-owned RK scratch double-buffer variants | `solve_ivp RK45` Lorenz on rch `hz1`, repeats=3000 | 31.335899 us/call | 28.621224 us/call parent | reject and revert |
| `frankenscipy-bpzha` | Solver-owned RK scratch double-buffer variants | `solve_ivp RK45` Lorenz on rch `ovh-a`, repeats=3000 | 32.037205 us/call | 20.597014 us/call parent | reject and revert |

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
| `frankenscipy-8l8r1.123` | `jnjnp_zeros` cutoff-driven generator | Tighter rectangular frontier seed from `frankenscipy-96n2y` | 2.30x faster at `nt=64`; 2.28x faster at `nt=128` on same-worker `ovh-b` | keep current despite SciPy loss |
| `frankenscipy-8l8r1.123` | `jnjnp_zeros` cutoff-driven full-sort prefix | Top-k candidate partition from `frankenscipy-8l8r1.124` | top-k only 1.019x at `nt=64` and 1.013x at `nt=128` in longer same-binary probe | reject top-k; keep full-sort |
| `frankenscipy-wm14d` | 2D Reflect/order=1 direct bilinear zoom fast path with parallel fill | Generic per-pixel sampler path | 4.27x faster on `zoom/2x_256/order=1` on same-worker `ovh-b` | keep current despite SciPy loss |
| `frankenscipy-wm14d` | 2D Reflect/order=1 direct bilinear zoom fast path with parallel fill | Serial fill probe inside the same fast path | 1.22x faster than serial on same-worker `ovh-b` | revert serial probe |
| `frankenscipy-8l8r1.116` | Full-complex CSD route | rfft CSD route | 1.123x faster on 4096; rfft wins 2.107x on 65536 but loses to SciPy rfft oracle | revert rfft route |
| `frankenscipy-va60h` | Flat row-major linkage arena | Legacy nested-row NN-array helper | 1.128x faster on Average; 1.019x faster on Ward | keep current despite SciPy loss |
| `frankenscipy-va60h` | Flat row-major linkage arena | Reverted production nested route probe | 1.290x faster on Average; 1.251x faster on Ward | undo revert; keep flat route |
| `frankenscipy-8l8r1.118` | Fused signal coherence | Compositional `csd(x,y)` + `csd(x,x)` + `csd(y,y)` route | 2.98x faster locally; 2.80x faster on rch `hz1` | keep fused route |
| `frankenscipy-nm8ex` | Dim-4 Euclidean/Cosine direct serial `pdist` kernels | Generic metric-dispatch/reduction/threaded path | 5.54-29.51x faster on same-worker `hz2` | keep current despite SciPy loss |
| `frankenscipy-x9ckc` | `jnjnp_zeros` guarded direct integer-order root polishing | Generic strict-mode bracketed zero route after output frontier | 1.17x faster at `nt=64`; 1.20x faster at `nt=128` on same-worker `hz1` | keep current despite SciPy loss |
| `frankenscipy-fo9cj` | Restored sparse Arnoldi `Vec<Vec>` basis and allocating matvec closure | Row-major basis arena plus mutable operator scratch | 1.19-1.41x faster on `eigsh`; candidate `svds` movement only 0.99-1.06x | reject arena/scratch route |
| `frankenscipy-nm8ex.1` | Dim-4 `pdist` fixed-width `[f64; 4]` row staging | Prior direct serial dim-4 `Vec<Vec<f64>>` fast path | 1.11-1.83x faster on same-worker `ovh-b`; still 0/4/0 vs SciPy | keep current despite SciPy loss |
| `frankenscipy-bpzha` | Restored parent per-attempt RK temporaries | Solver-owned scratch double-buffer variants | paired rows finished 1 win / 3 losses / 0 neutral; Lorenz regressed 1.067x-1.555x on fresh rch workers | reject scratch route |

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
| `fsci-special` `jnjnp_zeros` performance | measured loss plus internal keep and reject | cutoff-driven generator is 2.30x faster at `nt=64` and 2.28x faster at `nt=128` than the tighter rectangular frontier on same-worker `ovh-b`, but still 3.68-3.71x slower than SciPy; top-k candidate partition was rejected as 1.019x/1.013x near-noise |
| `fsci-special` `jnjnp_zeros` correctness | guarded | `jnjnp_adaptive_envelope_matches_oversized_reference`, `jnjnp_frontier_matches_scipy_bench_cutoffs`, and `jnyn_and_jnjnp_zeros_match_scipy` passed via rch; live SciPy `diff_special_bessel_zeros` conformance passed locally |
| `fsci-special` lint/build gate | partial | `cargo check -p fsci-special --all-targets` passed; clippy `-D warnings` stopped on existing `fsci-integrate`/`fsci-linalg` dependency lints; broad rustfmt/touched-file rustfmt remain blocked by pre-existing formatting drift outside this patch |
| `fsci-ndimage` zoom order=1 performance | measured loss plus internal keep | 2D Reflect/order=1 direct bilinear zoom path is 4.27x faster than the generic sampler on same-worker `ovh-b`, but still 2.05x slower than SciPy |
| `fsci-ndimage` zoom order=1 correctness | guarded | focused bit-equivalence guard against the generic sampler passed via rch; broader `zoom_` test filter passed via rch; live SciPy `diff_ndimage_zoom` conformance passed locally |
| `fsci-ndimage` lint/build gate | partial | `cargo check -p fsci-ndimage --all-targets`, `git diff --check`, and UBS passed; fmt remains blocked by pre-existing drift and clippy by existing `fsci-linalg`/`fsci-ndimage` lint debt outside this patch |
| `fsci-fft` lint/build gate | guarded | `cargo check -p fsci-fft --all-targets`, focused CSD/rfft tests, and `cargo clippy -p fsci-fft --all-targets -- -D warnings` passed; broad rustfmt remains blocked by pre-existing file-wide drift |
| `fsci-cluster` linkage performance | measured loss plus internal keep | flat arena is 1.128x faster than the legacy nested helper on Average and 1.019x on Ward, but 1.385-1.497x slower than SciPy |
| `fsci-cluster` linkage correctness | guarded | filtered linkage tests passed via rch (28 unit, 9 metamorphic); SciPy-backed `diff_cluster_linkage_from_distances` conformance passed locally |
| `fsci-cluster` lint/build gate | partial | `cargo check -p fsci-cluster --benches` passed; fmt blocked on existing `perf_isomap.rs` drift and clippy blocked on existing `fsci-linalg` dependency lints |
| `fsci-ndimage` gaussian_filter route | measured reject | always-line-walk plus outermost row-split regressed `gaussian_sigma2/256`; source reverted, current remains a SciPy loss routed to inner dot SIMD/tiled contiguous-span work |
| `fsci-spatial` `pdist` performance | measured loss plus internal keep | dim-4 direct serial kernels were 5.54-29.51x faster than the generic threaded route; follow-up flat row staging is another 1.11-1.83x same-worker `ovh-b` internal win but still 0/4/0 vs SciPy, 1.94-3.01x slower |
| `fsci-spatial` `pdist` correctness | guarded | focused dim-4 bit-exact `pdist` test and full `fsci-spatial --lib` suite passed after flat row staging |
| `fsci-spatial` lint/build gate | partial | `cargo check -p fsci-spatial --all-targets`, `cargo clippy -p fsci-spatial --all-targets --no-deps -- -D warnings`, `git diff --check`, and UBS passed; `cargo fmt --check -p fsci-spatial` remains blocked by pre-existing `fsci-spatial` rustfmt drift |
| `fsci-sparse` `eigsh`/`svds` performance | measured reject plus current-route SciPy wins | row-major Arnoldi arena regressed all `eigsh` rows and was reverted; restored route is 4 wins / 1 loss / 1 neutral vs SciPy, with the remaining loss at `eigsh n=8000 k=6` |
| `fsci-sparse` `eigsh`/`svds` correctness | guarded | focused sparse eig/svds tests passed via rch; SciPy-backed `diff_sparse_eigsh_largest` and `diff_sparse_svds` conformance passed locally |
| `fsci-sparse` lint/build gate | partial | `cargo check -p fsci-sparse --all-targets`, focused sparse tests, `cargo clippy -p fsci-sparse --all-targets --no-deps -- -D warnings`, UBS, and `git diff --check` passed; rch SciPy conformance blocked by missing worker SciPy and rustfmt by pre-existing file-wide drift |
| `fsci-integrate` RK performance | measured reject plus restored SciPy wins | RK scratch double-buffer was reverted after paired same-worker/fresh-worker rows finished 1 win / 3 losses / 0 neutral; restored parent route remains 77.64x faster than SciPy on exponential and 72.97x faster on Lorenz |
| `fsci-integrate` RK correctness | guarded | scratch candidate passed focused RK and IVP `e2e_ivp` checks before rejection; final source ships no RK code change |
| `fsci-integrate` lint/build gate | partial | scratch candidate `cargo check -p fsci-integrate --all-targets` passed; final source ships no RK lint surface, while crate clippy remains blocked by unrelated existing `api.rs`/`quad.rs` lints |
| rch SciPy oracle parity | blocked on worker image | `vmi1152480` and `vmi1227854` lacked `scipy`; local same-host oracle supplied the head-to-head ratios |
| Release readiness | partial | two linalg perf clusters, one `fsci-signal` coherence win, one `fsci-opt` L-BFGS-B reject, one `fsci-special` measured SciPy-loss/internal-keep cutoff win plus one top-k reject, one `fsci-fft` CSD reject, one `fsci-cluster` linkage measured SciPy loss, one `fsci-ndimage` gaussian reject, one `fsci-spatial` `pdist` internal keep, one `fsci-sparse` Arnoldi reject, and one `fsci-integrate` RK scratch reject verified; other code-first perf ledger entries still need gauntlet conversion |

## Pending Gauntlet Backlog

Continue converting `pending batch-test` entries in the negative-evidence ledger
one cluster at a time. Each conversion must record the SciPy ratio, internal
A/B route deltas, conformance status, and keep/revert decision before release
readiness can be raised.
