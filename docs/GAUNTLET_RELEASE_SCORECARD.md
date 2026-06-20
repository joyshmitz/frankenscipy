# Gauntlet Release Scorecard

Last updated: 2026-06-20 by cod-a/cod-b / MistyBirch.

This scorecard tracks code-first performance work that has been converted into
measured head-to-head evidence against the SciPy original. The detailed
win/loss/neutral ledger lives in `docs/progress/perf-negative-results.md`.

## Measured Keeps

| Bead | Cluster | Realistic workload | Rust result | SciPy result | Ratio | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `frankenscipy-8l8r1.118` | Fused signal coherence | `scipy.signal.coherence`, 65536 samples, Hann window 1024/512 overlap | 2.191980 ms | 18.961613 ms | 8.65x faster | keep |
| `frankenscipy-8l8r1.119/.121` | BDF streamed scaled RMS norms | `solve_ivp(method=BDF)` stiff diagonal decay, n=64 | 1.959287 ms | 26.351239 ms | 13.45x faster | keep BDF stream helper; 1.040x internal win on same-worker BDF64 |
| `frankenscipy-8l8r1.119/.121` | BDF streamed scaled RMS norms | `solve_ivp(method=BDF)` stiff diagonal decay, n=128 | 11.052293 ms | 29.334903 ms | 2.65x faster | keep; internal A/B neutral on same-worker BDF128 |
| `frankenscipy-zpunl` | Radau diagonal-Jacobian stage solve | `solve_ivp(method=Radau)` stiff diagonal decay, n=64 | 1.687784 ms | 36.545873 ms | 21.65x faster | keep; same-worker old dense stage LU to diagonal solve was 62.29x faster with unchanged counters/checksum |
| `frankenscipy-u0ucw` | Wide `pinv` Cholesky TRSM + diagonal rcond gate | 500x1000 full-row-rank dense `scipy.linalg.pinv` equivalent | 183.699926 ms | 7.257573 s | 39.51x faster | keep |
| `frankenscipy-u0ucw` | Wide `lstsq` current materialized normal equations after row-stream revert | 500x1000 full-row-rank dense `scipy.linalg.lstsq` equivalent | 109.369915 ms | 1.253347 s | 11.46x faster | keep current, reject row-stream lever |
| `frankenscipy-nm8ex` | Dim-4 `pdist` SIMD-across-pairs (SoA), serial n≤2048 / parallel above | `scipy.spatial.distance.pdist` euclidean d=4, n=256→4096 | 48.5us / 167us / 666us / 2.86ms / 13.2ms | 85.3us / 303us / 1.20ms / 4.65ms / 51.1ms | **1.76 / 1.82 / 1.80 / 1.63 / 3.87x faster** | keep — was 1.14-1.37x slower (and 2.4x slower at n=1024 via over-spawn); bit-identical |
| `frankenscipy-nm8ex` | Dim-4 `pdist` SIMD-across-pairs (SoA), serial n≤2048 / parallel above | `scipy.spatial.distance.pdist` cosine d=4, n=256→4096 | 40.7us / 148us / 549us / 2.53ms / 13.1ms | 76.8us / 275us / 1.08ms / 4.23ms / 48.3ms | **1.89 / 1.86 / 1.97 / 1.67 / 3.69x faster** | keep — was 1.31-1.63x slower; bit-identical |
| `frankenscipy-nm8ex` | Dim-4 `cdist` SoA SIMD-across-columns + 16-thread cap | `scipy.spatial.distance.cdist` euclidean d=4, 1000×1000 / 2000×500 | 1.99ms / 2.23ms | 2.17ms / 2.19ms | **1.09x faster / parity** | keep — was 2.6x slower (over-spawn 64 thr); bandwidth-bound, cap16; bit-identical |
| `frankenscipy-nm8ex` | Dim-4 `cdist` SoA SIMD-across-columns + 16-thread cap | `scipy.spatial.distance.cdist` cosine d=4, 1000×1000 / 2000×500 | 1.76ms / 2.20ms | 2.03ms / 2.03ms | **1.15x faster / parity** | keep — 11.3x self-speedup; bit-identical |
| `frankenscipy-nm8ex` | Dim-4 `pdist` SqEuclidean SoA SIMD-across-pairs | `scipy.spatial.distance.pdist` sqeuclidean d=4, n=512 | 0.102ms | 0.179ms | **1.75x faster** | keep — was 2.47ms (13x slower, over-spawn); 24x self; bit-identical |
| `frankenscipy-nm8ex` | Dim-4 `pdist` Cityblock SoA SIMD-across-pairs | `scipy.spatial.distance.pdist` cityblock d=4, n=512 | 0.109ms | 0.190ms | **1.74x faster** | keep — was 2.49ms (13x slower); 23x self; bit-identical |
| `frankenscipy-nm8ex` | `pdist` thread gate (all metrics): serial < 1<<20 work, cap 16 | over-spawn fix, n=512 (cityblock/sqeucl/chebyshev d4/d16/d64) | 0.7-2.2ms | (was 2.3-3.3ms 64-thr) | **1.3-3.6x self** (byte-identical) | keep — fixes 64-thread over-spawn for non-fast-path metrics |
| `frankenscipy-9l5oo` | Delaunay circumcircle grid candidate index for n>=4096 | `scipy.spatial.Delaunay` deterministic 2-D points, n=1000/2000/4000/8000 | 0.754ms / 2.613ms / 9.463ms / 20.622ms | 1.933ms / 4.550ms / 9.501ms / 20.627ms | **2.56x / 1.74x faster; parity at 4000/8000** | keep — 4000/8000 loss closed to parity; score 2 wins / 0 losses / 2 neutral |
| `frankenscipy-2hclc` | Public CSR `spmv_csr` cached-slice + 4-lane unrolled row sweep | `scipy.sparse` CSR `.dot(x)`, n=100/1000/10000 with 500/10k/100k nnz | 0.388us / 7.077us / 68.82us | 4.63us / 8.00us / 96.95us | **11.95x / 1.13x / 1.41x faster** | keep — was 1 win / 2 losses; now 3 wins / 0 losses; bit-identical to legacy public row sweep |
| `frankenscipy-oi8hq` | `ndimage.zoom` 2D Reflect/order=1 no-prefilter fast path | `scipy.ndimage.zoom(256x256, 2x, order=1)` equivalent | 1.2219 ms | 4.86171 ms | **3.98x faster** | keep — closes previous `frankenscipy-wm14d` residual loss |
| `frankenscipy-wh8ac` | `jnjnp_zeros` Cephes `j1` seed evaluator | `scipy.special.jnjnp_zeros(nt=64/128)` equivalent | 381.89 us / 742.06 us | 463.91 us / 832.79 us | **1.22x / 1.12x faster** | keep — same-worker rch internal win 1.59-1.61x; closes `frankenscipy-9l5oo` near-parity loss to `2/0/0` SciPy wins |

## Measured Losses / Internal Keeps

| Bead | Cluster | Realistic workload | Rust result | SciPy result | Ratio | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `frankenscipy-8l8r1.123` | `jnjnp_zeros` cutoff-driven generator | `scipy.special.jnjnp_zeros(nt=64)` equivalent | 1.5856 ms | 427.47 us | 3.71x slower | keep 2.30x internal win; route deeper |
| `frankenscipy-8l8r1.123` | `jnjnp_zeros` cutoff-driven generator | `scipy.special.jnjnp_zeros(nt=128)` equivalent | 2.9035 ms | 789.23 us | 3.68x slower | keep 2.28x internal win; route deeper |
| `frankenscipy-8l8r1.125` | `ndimage.mean(labels,index)` flat sum/count accumulator | N=65536 K=512 label-indexed mean | 590.978 us | 159 us | 3.72x slower | keep 1.77x internal win; route deeper |
| `frankenscipy-8l8r1.125` | `ndimage.mean(labels,index)` flat sum/count accumulator | N=262144 K=1024 label-indexed mean | 2.568 ms | 622 us | 4.13x slower | keep 1.44x internal win; route deeper |
| `frankenscipy-8l8r1.125` | `ndimage.mean(labels,index)` flat sum/count accumulator | N=262144 K=2048 label-indexed mean | 2.713 ms | 581 us | 4.67x slower | keep 1.53x internal win; route deeper |
| `frankenscipy-8l8r1.125` | `ndimage.mean(labels,index)` flat sum/count accumulator | N=589824 K=4096 label-indexed mean | 6.951 ms | 1.688 ms | 4.12x slower | keep 1.69x internal win; route deeper |
| `frankenscipy-fa62u` | `ndimage.mean(labels,index)` cheap dense integer probe | N=65536 K=512 label-indexed mean | 278.230 us | 0.210 ms | 1.33x slower | keep 2.13x same-host internal win vs dense `fract` probe; route deeper |
| `frankenscipy-fa62u` | `ndimage.mean(labels,index)` cheap dense integer probe | N=262144 K=1024 label-indexed mean | 1.122 ms | 0.749 ms | 1.50x slower | keep 2.09x same-host internal win vs dense `fract` probe; route deeper |
| `frankenscipy-fa62u` | `ndimage.mean(labels,index)` cheap dense integer probe | N=262144 K=2048 label-indexed mean | 1.186 ms | 0.751 ms | 1.58x slower | keep 2.13x same-host internal win vs dense `fract` probe; route deeper |
| `frankenscipy-fa62u` | `ndimage.mean(labels,index)` cheap dense integer probe | N=589824 K=4096 label-indexed mean | 3.169 ms | 1.781 ms | 1.78x slower | keep 1.94x same-host internal win vs dense `fract` probe; route deeper |
| `frankenscipy-wm14d` | `ndimage.zoom` 2D linear Reflect fast path | `scipy.ndimage.zoom(256x256, 2x, order=1)` equivalent | 7.9684 ms | 3.88937 ms | 2.05x slower | superseded by `frankenscipy-oi8hq` measured win |
| `frankenscipy-edt-indices` | `distance_transform_edt(return_indices)` separable feature transform (was O(f·b) brute force) | `scipy.ndimage.distance_transform_edt(return_indices=True)` 256x256 | 5.705 ms | 4.108 ms | 1.39x slower | **complexity gap CLOSED** — 835x self-speedup vs brute, byte-identical distances; route deeper to constant factor |
| `frankenscipy-96n2y` | `jnjnp_zeros` tighter frontier seed | `scipy.special.jnjnp_zeros(nt=64)` equivalent | 2.2230 ms | 424.10 us | 5.24x slower | keep 2.12x internal win; route deeper |
| `frankenscipy-96n2y` | `jnjnp_zeros` tighter frontier seed | `scipy.special.jnjnp_zeros(nt=128)` equivalent | 6.1605 ms | 799.97 us | 7.70x slower | keep 1.38x internal win; route deeper |
| `frankenscipy-x9ckc` | `jnjnp_zeros` guarded root-cost refinement | `scipy.special.jnjnp_zeros(nt=64)` equivalent | 4.6666 ms | 439.49 us | 10.62x slower | keep 1.17x internal win; route deeper |
| `frankenscipy-x9ckc` | `jnjnp_zeros` guarded root-cost refinement | `scipy.special.jnjnp_zeros(nt=128)` equivalent | 8.3620 ms | 787.18 us | 10.62x slower | keep 1.20x internal win; route deeper |
| `frankenscipy-01lxz` | `jnjnp_zeros` output-sensitive frontier | `scipy.special.jnjnp_zeros(nt=64)` equivalent | 4.3372 ms | 486.57 us | 8.91x slower | keep 17.82x internal win; route deeper |
| `frankenscipy-01lxz` | `jnjnp_zeros` output-sensitive frontier | `scipy.special.jnjnp_zeros(nt=128)` equivalent | 7.5415 ms | 792.81 us | 9.51x slower | keep 50.77x internal win; route deeper |
| `frankenscipy-acoco` | `jnjnp_zeros` bracket reuse | `scipy.special.jnjnp_zeros(nt=64)` equivalent | 80.728603 ms | 0.493655 ms | 163.53x slower | keep internal bracket reuse; route deeper |
| `frankenscipy-acoco` | `jnjnp_zeros` bracket reuse | `scipy.special.jnjnp_zeros(nt=128)` equivalent | 410.059973 ms | 0.924456 ms | 443.57x slower | keep internal bracket reuse; route deeper |
| `frankenscipy-9l5oo` | `jnjnp_zeros` Cephes `j0` + Newton/secant bracket refinement (cumulative) | `scipy.special.jnjnp_zeros(nt=64)` equivalent | 0.589 ms | 0.498 ms | 1.18x slower | **SUPERSEDED** by `frankenscipy-wh8ac` Cephes `j1` keep; was 163x slower (acoco), 158x self-speedup vs legacy route |
| `frankenscipy-9l5oo` | `jnjnp_zeros` Cephes `j0` + Newton/secant bracket refinement (cumulative) | `scipy.special.jnjnp_zeros(nt=128)` equivalent | 1.239 ms | 0.897 ms | 1.38x slower | **SUPERSEDED** by `frankenscipy-wh8ac` Cephes `j1` keep; was 443x slower (acoco), 381x self-speedup vs legacy route |
| `frankenscipy-nm8ex` | Spatial `pdist` dim-4 (was serial gate / flat row staging) | All 4 dim-4 workloads (eucl/cos × 256/512) | — | — | **GAP CLOSED → now 1.86-2.21x FASTER** | **WIN** — SIMD-across-pairs (SoA) `595eceb5` pipelines the dependent per-pair sqrt/div; bit-identical; promoted to Measured Keeps |
| `frankenscipy-va60h` | Linkage flat row-major distance arena | `scipy.cluster.hierarchy.linkage(method="average")`, n=800 d=4 | 6.1713 ms | 4.4550 ms | 1.385x slower | keep internal flat arena; route deeper |
| `frankenscipy-va60h` | Linkage flat row-major distance arena | `scipy.cluster.hierarchy.linkage(method="ward")`, n=800 d=4 | 7.5250 ms | 5.0256 ms | 1.497x slower | keep internal flat arena; route deeper |
| `frankenscipy-zpunl` | Restored Radau route after stream-norm reject | `solve_ivp(method=Radau)` stiff diagonal decay, n=64 | 70.176946 ms | 35.156708 ms | 2.00x slower | superseded by diagonal stage solve keep above |

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
| `frankenscipy-8l8r1.120` | Radau streamed scaled RMS norms | `radau-stiff32` same-worker `hz2` | 14.971402 ms | 12.586935 ms parent | reject and revert |
| `frankenscipy-8l8r1.120` | Radau streamed scaled RMS norms | `radau-stiff64` same-worker `hz2` | 81.492956 ms | 78.394828 ms parent | reject and revert |

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
| `frankenscipy-wh8ac` | `jnjnp_zeros` Cephes `j1` seed evaluator | Local J1 series/asymptotic seed split | 1.59x faster at `nt=64`; 1.61x faster at `nt=128` on same-worker `vmi1149989` | keep current; final SciPy score `2/0/0` |
| `frankenscipy-wm14d` | 2D Reflect/order=1 direct bilinear zoom fast path with parallel fill | Generic per-pixel sampler path | 4.27x faster on `zoom/2x_256/order=1` on same-worker `ovh-b` | keep current despite SciPy loss |
| `frankenscipy-wm14d` | 2D Reflect/order=1 direct bilinear zoom fast path with parallel fill | Serial fill probe inside the same fast path | 1.22x faster than serial on same-worker `ovh-b` | revert serial probe |
| `frankenscipy-oi8hq` | 2D Reflect/order=1 no-prefilter direct-original zoom path | Padded-coefficient fast path residual | 6.52-7.24x faster across rch residual rows; final 1.2219 ms is 3.98x faster than SciPy | keep; SciPy loss closed |
| `frankenscipy-8l8r1.116` | Full-complex CSD route | rfft CSD route | 1.123x faster on 4096; rfft wins 2.107x on 65536 but loses to SciPy rfft oracle | revert rfft route |
| `frankenscipy-va60h` | Flat row-major linkage arena | Legacy nested-row NN-array helper | 1.128x faster on Average; 1.019x faster on Ward | keep current despite SciPy loss |
| `frankenscipy-va60h` | Flat row-major linkage arena | Reverted production nested route probe | 1.290x faster on Average; 1.251x faster on Ward | undo revert; keep flat route |
| `frankenscipy-8l8r1.118` | Fused signal coherence | Compositional `csd(x,y)` + `csd(x,x)` + `csd(y,y)` route | 2.98x faster locally; 2.80x faster on rch `hz1` | keep fused route |
| `frankenscipy-8l8r1.119/.121` | BDF streamed scaled RMS norms | Pre-stream BDF norm materialization | BDF64 1.040x faster; BDF128 0.991x neutral on same-worker `hz2` | keep BDF stream helper |
| `frankenscipy-8l8r1.120` | Restored Radau collected scaled-vector route | Radau streamed scaled RMS candidate | candidate regressed same-worker `hz2` Radau32 1.19x and Radau64 1.04x | reject and revert streamed Radau norms |
| `frankenscipy-nm8ex` | Dim-4 Euclidean/Cosine direct serial `pdist` kernels | Generic metric-dispatch/reduction/threaded path | 5.54-29.51x faster on same-worker `hz2` | keep current despite SciPy loss |
| `frankenscipy-x9ckc` | `jnjnp_zeros` guarded direct integer-order root polishing | Generic strict-mode bracketed zero route after output frontier | 1.17x faster at `nt=64`; 1.20x faster at `nt=128` on same-worker `hz1` | keep current despite SciPy loss |
| `frankenscipy-fo9cj` | Restored sparse Arnoldi `Vec<Vec>` basis and allocating matvec closure | Row-major basis arena plus mutable operator scratch | 1.19-1.41x faster on `eigsh`; candidate `svds` movement only 0.99-1.06x | reject arena/scratch route |
| `frankenscipy-nm8ex.1` | Dim-4 `pdist` fixed-width `[f64; 4]` row staging | Prior direct serial dim-4 `Vec<Vec<f64>>` fast path | 1.11-1.83x faster on same-worker `ovh-b`; still 0/4/0 vs SciPy | keep current despite SciPy loss |
| `frankenscipy-bpzha` | Restored parent per-attempt RK temporaries | Solver-owned scratch double-buffer variants | paired rows finished 1 win / 3 losses / 0 neutral; Lorenz regressed 1.067x-1.555x on fresh rch workers | reject scratch route |
| `frankenscipy-8l8r1.125` | `ndimage.mean` flat sum/count accumulator | Prior O(N+K) label bucket materialization | 1.44-1.77x faster on same-host same-binary rows; 0 bit mismatches vs old and bucketed routes | keep current despite SciPy loss |
| `frankenscipy-fa62u` | `ndimage.mean` cheap dense integer probe | Prior dense lookup with `is_finite()+fract()` hot probe | 1.94-2.13x faster on same-host replay; 2.08-2.26x faster on rch `hz2`; 0 bit mismatches vs old, bucketed, hashflat, and dense_fract routes | keep current despite SciPy loss |

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
| `fsci-special` `jnjnp_zeros` performance | **measured KEEP (wh8ac)** | Cephes `j1` seed evaluator cuts current `nt=64/128` rows from 608.21 us / 1.1970 ms to 381.89 us / 742.06 us on same-worker `vmi1149989` (1.59-1.61x internal win) and beats local SciPy 1.17.1 at 463.91 us / 832.79 us; score `2/0/0` vs SciPy |
| `fsci-special` `jnjnp_zeros` correctness | guarded | `jnjnp_adaptive_envelope_matches_oversized_reference`, `jnjnp_frontier_matches_scipy_bench_cutoffs`, and `jnyn_and_jnjnp_zeros_match_scipy` passed via rch; `j1_matches_scipy_reference_values` passed via rch; live SciPy `diff_special_bessel_zeros` conformance passed locally |
| `fsci-special` lint/build gate | partial | `cargo check -p fsci-special --all-targets` passed via rch `hz1`; `git diff --check` and changed-file UBS passed; clippy `-D warnings` stopped on existing `fsci-integrate`/`fsci-linalg` dependency lints; broad rustfmt remains blocked by pre-existing formatting drift outside this patch |
| `fsci-ndimage` zoom order=1 performance | measured keep | 2D Reflect/order=1 no-prefilter direct-original path is 1.2219 ms on rch `vmi1149989` vs local SciPy median 4.86171 ms, 3.98x faster; previous `frankenscipy-wm14d` loss is closed |
| `fsci-ndimage` zoom order=1 correctness | guarded | focused bit-equivalence guard against the generic padded sampler passed via rch; broader `zoom_` test filter passed via rch `ovh-a` 6/0; live SciPy `diff_ndimage_zoom` conformance passed locally 1/0 |
| `fsci-ndimage` `distance_transform_edt` indices performance | **complexity gap closed (edt-indices)** | `return_indices` path moved from O(foreground·background) brute force to the exact separable feature transform O(N·ndim): 60-835x self-speedup (same-binary, growing with N), now 1.39-1.76x slower than SciPy's compiled C (was catastrophic); byte-identical distances |
| `fsci-ndimage` `distance_transform_edt` indices correctness | guarded | new property test (distances byte-identical to the distance-only fast path + every index a genuine nearest background on multi-background/tie grids); `perf_edt` isomorphism 0 mismatches/10876 cells; all 18 `distance_transform` unit tests incl scipy-pinned index fixtures; live SciPy `diff_ndimage_distance_transform_edt` conformance passed locally |
| `fsci-ndimage` label-indexed measurements performance | **complexity gap closed; mean constants cut again** | `measurement_label_groups` moved shared label reductions from O(N*K) to O(N+K). Follow-ups first removed per-label bucket materialization, then replaced per-element HashMap probes with compact dense integer-label lookup. `frankenscipy-fa62u` now removes the dense route's `is_finite()+fract()` hot probe: dense_fast is 1.94-2.13x faster than dense_fract on same-host replay and 2.08-2.26x faster on rch `hz2`, but final same-host score remains `0/4/0` vs SciPy at 1.33-1.78x slower |
| `fsci-ndimage` label-indexed measurements correctness | guarded | byte-identical (`mism=0/0/0/0` vs old linear scan, previous bucketed route, previous flat HashMap route, and previous dense_fract route in `perf_label_stats`); full `cargo test -p fsci-ndimage --lib` 241 passed / 0 failed; live SciPy-backed `diff_ndimage` conformance passed locally 5/0 |
| `fsci-ndimage` lint/build gate | partial | `cargo check -p fsci-ndimage --all-targets`, touched-file rustfmt, `git diff --check`, changed-file UBS, focused zoom tests, and local SciPy `diff_ndimage_zoom` conformance passed; explicit `cargo clippy -p fsci-ndimage --all-targets -- -D warnings` is blocked before this patch on existing `fsci-linalg` dependency lints |
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
| `fsci-sparse` public `spmv_csr` performance | measured keep; scale loss closed | cached-slice + 4-lane unrolled row sweep is 1.54-2.14x faster than the legacy public row sweep in same-process rch A/B, all `to_bits` identical; final source scores 3 wins / 0 losses / 0 neutral vs SciPy |
| `fsci-sparse` public `spmv_csr` correctness | guarded | focused `spmv` unit/property/metamorphic tests passed via rch; sparse conformance filter recorded in `docs/progress/perf-release-readiness-scorecard.md` |
| `fsci-sparse` lint/build gate | partial | `cargo check -p fsci-sparse --all-targets`, focused sparse tests, `cargo clippy -p fsci-sparse --all-targets --no-deps -- -D warnings`, UBS, and `git diff --check` passed; rch SciPy conformance blocked by missing worker SciPy and rustfmt by pre-existing file-wide drift |
| `fsci-integrate` RK performance | measured reject plus restored SciPy wins | RK scratch double-buffer was reverted after paired same-worker/fresh-worker rows finished 1 win / 3 losses / 0 neutral; restored parent route remains 77.64x faster than SciPy on exponential and 72.97x faster on Lorenz |
| `fsci-integrate` RK correctness | guarded | scratch candidate passed focused RK and IVP `e2e_ivp` checks before rejection; final source ships no RK code change |
| `fsci-integrate` BDF/Radau stiff performance | measured keeps plus measured reject | BDF streamed RMS helper kept; Radau streamed RMS rejected/reverted; Radau diagonal-Jacobian stage solve kept. Final stiff suite is `4/0/0` vs SciPy: BDF64 11.03x faster, BDF128 2.48x faster, Radau32 57.01x faster, Radau64 27.93x faster |
| `fsci-integrate` BDF/Radau correctness | guarded/partial | focused Radau tests 3/0 and `cargo check -p fsci-integrate --all-targets` passed via rch for the Radau diagonal solve; current `fsci-conformance --test e2e_ivp` is blocked before IVP tests by unrelated `fsci-cluster` compile errors |
| `fsci-integrate` lint/build gate | partial | crate check, touched-file rustfmt, and diff hygiene passed; crate clippy remains blocked by unrelated existing `api.rs`/`rk.rs`/`quad.rs` lints after the touched Radau helper lint was fixed |
| rch SciPy oracle parity | blocked on worker image | `vmi1152480` and `vmi1227854` lacked `scipy`; local same-host oracle supplied the head-to-head ratios |
| Release readiness | partial | two linalg perf clusters, one `fsci-signal` coherence win, one `fsci-opt` L-BFGS-B reject, one `fsci-special` measured SciPy-loss/internal-keep cutoff win plus one top-k reject, one `fsci-fft` CSD reject, one `fsci-cluster` linkage measured SciPy loss, one `fsci-ndimage` zoom win plus gaussian reject, one `fsci-spatial` `pdist` internal keep, one `fsci-sparse` Arnoldi reject, and one `fsci-integrate` RK scratch reject verified; other code-first perf ledger entries still need gauntlet conversion |

## Pending Gauntlet Backlog

Continue converting `pending batch-test` entries in the negative-evidence ledger
one cluster at a time. Each conversion must record the SciPy ratio, internal
A/B route deltas, conformance status, and keep/revert decision before release
readiness can be raised.
