# FrankenSciPy Perf Ledger — CrimsonForge (measured head-to-head vs SciPy/sklearn)

Sidecar to the canonical `docs/NEGATIVE_EVIDENCE.md` (reserved by MistyBirch). Holds
**CrimsonForge's** measured gauntlet results so dead ends are never retried and
regressions are reverted. Entries also routed to MistyBirch for the canonical merge.

- Host: 64 cores, release builds via
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cc rch exec -- cargo bench -p <crate>`.
- Original = SciPy 1.17.1 / scikit-learn 1.9.0. Oracle scripts: `docs/perf_oracle_*.py`.
- "vs serial" A/B isolates a parallelization by forcing its work-gate to `usize::MAX`.
- ⚠️ Some runs taken under concurrent multi-agent bench load → parallel numbers are
  conservative (more core contention hurts the 64-thread path more than serial).

## Scorecard

| Optimization (bead) | Workload | SciPy/orig | fsci | fsci vs orig | parallel vs serial | Verdict |
|---|---|---|---|---|---|---|
| GMM diag E-step parallel (yw7ts) | GMM n=1000 d=3 k=3 | 2.398 ms | 0.592 ms | **4.05× faster** | serial (below gate) | ✅ KEEP |
| GMM diag E-step parallel (yw7ts) | GMM n=5000 d=8 k=5 | 29.75 ms | 3.735 ms | **7.97× faster** | parallel | ✅ KEEP |
| GMM diag E-step parallel (yw7ts) | GMM n=20000 d=16 k=8 | 432.4 ms | 39.04 ms | **11.1× faster** | **3.07×** (119.8→39.0 ms) | ✅ KEEP |
| AP responsibility parallel (yw7ts) | AP n=1000 d=4 | 319.4 ms | 249.9 ms | 1.28× faster | — | ✅ KEEP (parity) |
| AP responsibility parallel (yw7ts) | AP n=2000 d=4 | 2158 ms | 2098 ms | **1.03× (PARITY)** | **2.02×** (4.23→2.10 s) | ✅ KEEP (parity) |
| Interpolate evaluate_many parallel (yw7ts) | NdPPoly m=200k total=6 | n/a | 8.86 ms | — | **0.88× (REGRESSION)** vs serial 7.79 ms | ❌ **REVERTED** |
| Interpolate evaluate_many parallel (yw7ts) | BPoly m=200k | n/a | ~8.5 ms | — | ~serial 8.18 ms (no gain) | ❌ **REVERTED** |
| Distribution pdf_many lgamma-hoist (q53ya) | gamma.pdf n=4096 | 149.6 µs | 49.86 µs | **3.0× faster** | 3.2× (hoist vs map) | ✅ KEEP |
| Distribution pdf_many lgamma-hoist (q53ya) | beta.pdf n=4096 | 296.8 µs | 60.97 µs | **4.87× faster** | 4.3× (hoist vs map) | ✅ KEEP |
| Distribution pmf_many lgamma-hoist (q53ya) | hypergeom.pmf supp=701 | 3744.9 µs | 38.34 µs | **97.7× faster** | — | ✅ KEEP |
| pdist parallel (8e7e6d99, NOT mine) | pdist euclidean n=256 | 92.1 µs | 674.9 µs | **0.14× (7.3× SLOWER)** | gate fires at n=256 | ⚠️ LOSS → owner |
| pdist parallel (8e7e6d99, NOT mine) | pdist euclidean n=512 | 326.3 µs | 889.0 µs | **0.37× (2.7× SLOWER)** | overhead amortizes w/ n | ⚠️ LOSS → owner |
| pdist parallel (8e7e6d99, NOT mine) | pdist cosine n=256 | 81.9 µs | 736.7 µs | **0.11× (9× SLOWER)** | — | ⚠️ LOSS → owner |
| linkage NN-chain (average) | linkage n=400 d=4 | 1586.5 µs | 1904.5 µs | **0.83× (1.2× slower)** | — | ⚠️ near-parity |
| cophenet mem::take (jphzn) | cophenet n=400 | 401.5 µs | 219.7 µs | **1.83× faster** | — | ✅ KEEP |
| kmeans2 double-buffer (4ylee) | kmeans2 k4 n2000 iter=50 | 2104.7 µs | 5126 µs | **0.41× (2.4× SLOWER)** | scalar assign vs scipy SIMD | ⚠️ kernel gap → bead |
| kmeans Lloyd early-stop | kmeans k4 n2000 | 2104.7 µs* | 357.4 µs | **5.9× faster** | *vs scipy kmeans2 fixed-iter | ✅ KEEP (early-stop) |
| correlate tap-table (e3r7e) | correlate 5x5 256² | 933.7 µs | 1099 µs | **0.85× (1.18× slower)** | byte-identical | ✅ KEEP (parity) |
| gaussian_filter (NOT mine) | gaussian σ=2 256² | 1143.0 µs | 3238 µs | **0.35× (2.83× slower)** | separable but slow 1D kernel | ⚠️ gap → owner |
| spmv_csr (serial, baseline) | SpMV n=100 nnz=500 | 3.45 µs | 0.81 µs | **4.26× faster** | scipy Python dispatch overhead | ✅ small-n win |
| spmv_csr (serial, baseline) | SpMV n=1000 nnz=10k | 8.14 µs | 17.97 µs | **0.45× (2.2× slower)** | scalar kernel vs scipy C | ⚠️ scale gap |
| spmv_csr (serial, baseline) | SpMV n=10000 nnz=100k | 131.6 µs | 197.5 µs | **0.67× (1.5× slower)** | gap narrows w/ n | ⚠️ scale gap |
| gaussian_kde evaluate_many parallel | KDE n=1000 eval 1000 pts | 19090 µs | 1062 µs | **18.0× faster** | heavy per-pt → scales | ✅ KEEP |
| gaussian_kde evaluate_many parallel | KDE n=5000 eval 5000 pts | 201197 µs | 11959 µs | **16.8× faster** | — | ✅ KEEP |
| MGC mgc_map O(n²) + parallel reps | multiscale_graphcorr n=80 reps=100 | 295705 µs | 21578 µs | **13.7× faster** | O(n⁴)→O(n²) + parallel | ✅ KEEP |
| Rotation.apply_many (w7ocv) | apply 8192 pts | 28.30 µs | 12.03 µs | **2.35× faster** | matrix-once hoist 4.5× vs map | ✅ KEEP |
| loadtxt direct-parse (fwnb1) | loadtxt 500×20 | 2022 µs | 259.5 µs | **7.79× faster** | vs numpy.loadtxt (Python) | ✅ KEEP |
| savetxt write! (d1uxy) | savetxt 500×20 | 4208 µs | 631.6 µs | **6.66× faster** | vs numpy.savetxt (Python) | ✅ KEEP |
| KDTree build (select_nth) | cKDTree build n=4096 3-D | 767.8 µs | 809.5 µs | 0.95× (parity) | vs scipy ELITE C | ✅ KEEP |
| KDTree query dual-tree parallel (9k50g) | cKDTree query 4096 pts | 2032.8 µs | 1756.7 µs | **1.16× faster** | beats single-threaded C | ✅ KEEP |
| silhouette per-anchor parallel | silhouette n=500 d=4 | 2064 µs | 720.8 µs | **2.86× faster** | no small-n regression | ✅ KEEP |
| silhouette per-anchor parallel | silhouette n=2000 d=4 | 32928 µs | 3113.5 µs | **10.6× faster** | scales w/ n | ✅ KEEP |
| ndimage zoom (kernel, NOT my opt) | zoom 2× 256² order=1 | 4842 µs | 85950 µs | **0.06× (17.7× SLOWER)** | order=1 slower than order=3! | ⚠️ LOSS → bead |
| ndimage zoom (kernel, NOT my opt) | zoom 2× 256² order=3 | 14053 µs | 31573 µs | **0.45× (2.25× slower)** | generic spline-weight kernel | ⚠️ LOSS → bead |
| kendalltau inversion-count O(n log n) | kendalltau n=2048 | 597 µs | 230.4 µs | **2.59× faster** | scipy fixed overhead | ✅ KEEP |
| kendalltau inversion-count O(n log n) | kendalltau n=4096 | 537 µs | 552.4 µs | 0.97× (parity) | both O(n log n) at scale | ✅ KEEP |
| Delaunay Bowyer-Watson (8d2z2 hoist) | Delaunay n=1000 2-D | 1980 µs | 6534 µs | **0.30× (3.3× slower)** | O(n²) point-location | ⚠️ COMPLEXITY gap → bead |
| Delaunay Bowyer-Watson (8d2z2 hoist) | Delaunay n=2000 2-D | 4488 µs | 26311 µs | **0.17× (5.9× slower)** | scales O(n²) vs Qhull O(n log n) | ⚠️ COMPLEXITY gap → bead |

## Detail

### GMM diagonal E-step ordered-slots parallelization (frankenscipy-yw7ts) — ✅ KEEP
Oracle `docs/perf_oracle_gmm.py` (sklearn `GaussianMixture(covariance_type="diag")`).
fsci **4–11× faster** than sklearn, ratio growing with n as the parallel E-step
(gate n·k·d≥2¹⁶) engages. Forced-serial A/B at n=20000: 119.79 ms serial vs 39.04 ms
parallel = **3.07×** from parallelization (64 cores; Amdahl-capped by serial
M-step+Cholesky). Even forced-serial fsci beats sklearn 3.6× → port is the base win.
Gate validated: n=1000 serial (592 µs) correctly avoids spawn overhead. Conformance
green (gaussian_mixture_recovers_* tests). Commit `8b84e8b2`.

### Affinity propagation responsibility-update parallelization (frankenscipy-yw7ts) — ✅ KEEP, but AP is PARITY
Oracle `docs/perf_oracle_ap.py` (sklearn `AffinityPropagation(affinity="precomputed")`).
fsci AP is **near parity** with sklearn: 1.28× at n=1000, **1.03× at n=2000**. sklearn
AP is numpy-vectorized and well-optimized. Forced-serial A/B at n=2000: 4.234 s serial
vs 2.098 s parallel = **2.02×** from the responsibility-update parallelization — i.e.
the parallelization is exactly what lifts fsci to parity; forced-serial fsci would
LOSE ~2× to sklearn. **Keep the parallelization (real 2× internal, byte-identical),
but AP is NOT a competitive advantage.** Remaining gap = the still-serial availability
update (column-strided over the row-major matrix; parallelizing it needs a transposed
layout — a candidate future lever, NOT yet done). Commit `1f32a4b2`.

### Interpolate batch-evaluator parallelization (frankenscipy-yw7ts) — ❌ REVERTED
BPoly/NdPPoly/NdBSpline `evaluate_many` were parallelized across points (par_query_map
for BPoly; hand-rolled per-thread scratch for NdPPoly/NdBSpline). **Measured A/B at
m=200k** (forced-serial gate→MAX, rebuilt): NdPPoly **serial 7.79 ms vs parallel
8.86 ms = 0.88× (a 14% REGRESSION)**; BPoly serial 8.18 ms ≈ parallel (no gain). The
per-point work is only ~k/total flops (~30 for the typical low-degree/low-dim case),
so 64-thread spawn + per-thread-scratch allocation overhead exceeds the compute — the
opposite of GMM, whose heavy gaussian/exp per-point work parallelizes 3×. The gate
(`points·total ≥ 2¹⁶`) counts flops, but 2¹⁶ flops is trivial vs thread overhead, and
the break-even (if any) is contention-dependent and unverifiable on this shared host.
**Reverted all three to the serial map; the byte-identical loop-invariant HOIST
(binoms/strides/scratch precomputed once) is PRESERVED — that was the real, monotone
win.** Conformance green (interpolate evaluate_many tests). Revert commit: this one.

### Distribution batch pdf_many/pmf_many lgamma-hoist (frankenscipy-q53ya) — ✅ KEEP (strong)
Oracle `docs/perf_oracle_dist.py` (scipy.stats vectorized pdf/pmf over arrays, n=4096
/ full support). fsci `pdf_many`/`pmf_many` hoist the expensive lgamma/ln_beta
normalizer ONCE then map. **fsci beats scipy.stats 3–98×:** gamma 3.0×, beta 4.87×,
hypergeom **97.7×** (scipy's hypergeom.pmf is famously slow — betaln + overflow
guards per outcome). The hoist itself is 3–4× over the naive `map(pdf)` (gamma
49.9µs vs 159.9µs; beta 61.0µs vs 261.3µs), and `map(pdf)` ≈ scipy — i.e. the hoist
is exactly what wins the head-to-head. Byte-identical (normalizer is a loop
invariant), NO revert risk. The 19-density batch family shares this lever → all KEEP
by the same construction. Conformance green. Commits: `q53ya` (impl) + oracle here.

### Spatial pdist — ⚠️ LOSS vs scipy (NOT my optimization; flagged to owner)
Oracle `docs/perf_oracle_pdist.py` (scipy.spatial.distance.pdist, 4-D, n=256/512).
fsci pdist (parallel, commit `8e7e6d99` by another agent) is **2.7–9× SLOWER than
scipy**: euclidean 674.9µs vs 92.1µs (n=256), 889.0µs vs 326.3µs (n=512); cosine
736.7µs vs 81.9µs (n=256). The ratio improves with n (7.3×→2.7×) → fixed thread-spawn
overhead. The gate `cdist_thread_count` fires at `work=n²·dim≥2¹⁸`, i.e. exactly at
n=256/dim=4 — parallelizing trivially-small 4-D distance pairs across 64 threads, the
same over-eager pathology as the (reverted) interpolate evaluators, worsened by
multi-agent contention. The author's "3.8–7.3×" claim is parallel-vs-serial INTERNAL,
not vs scipy; implied serial ≈ 3.4–6.5 ms at n=512 → fsci's pure-Rust pdist KERNEL is
~10–60× slower than scipy's C. **NOT reverted — another agent's file; routed to the
spatial owner.** Recommendation: raise the pdist gate well above 2¹⁸ AND/OR a faster
inner kernel (scipy uses tuned C). Honest LOSS recorded.

### Hierarchical clustering: linkage + cophenet (frankenscipy-jphzn) — ⚠️ parity / ✅ KEEP
Oracle `docs/perf_oracle_hier.py` (scipy.cluster.hierarchy, n=400 blobs, average).
- **linkage average: fsci 1904.5 µs vs scipy 1586.5 µs = 0.83× (1.2× SLOWER).** Near-
  parity — scipy's NN-chain linkage is tuned C; fsci's pure-Rust version is within
  20%. NOT a regression (no parallelization involved); just the expected small gap to
  optimized C. KEEP (correct + close); a faster reducible-distance update is a future
  lever if linkage becomes a bottleneck.
- **cophenet: fsci 219.7 µs vs scipy 401.5 µs (distances-only, fair) = 1.83× FASTER.**
  The `jphzn` move-instead-of-clone of each node's member list helps; the tree
  traversal is efficient. (NB: the naive `cophenet(Z, Y)` scipy call is 1758 µs but
  ALSO computes the correlation coefficient — not comparable; used `cophenet(Z)`.)
  KEEP.

### kmeans / kmeans2 (frankenscipy-4ylee double-buffer) — mixed; kernel gap surfaced
Oracle: scipy.cluster.vq.kmeans2 (n=2000, k=4, d=4, fixed init).
- **kmeans2 fixed 50 iters: fsci 5126 µs vs scipy 2104.7 µs = 0.41× (2.4× SLOWER).**
  Both run 50 full Lloyd iterations. The 4ylee double-buffer (mem::swap vs realloc) is
  byte-identical and NOT the cause — the gap is the **scalar nearest-centroid
  assignment** (n·k·d per iter) vs scipy's vectorized C. At k=4/d=4 (~16 flops/point)
  PARALLELIZING would regress (cheap-work pathology, cf. interpolate/pdist) — the fix
  is **SIMD the distance kernel**, not threads. Bead `→` filed. Double-buffer KEEP.
- **kmeans (early-stop Lloyd): fsci 357.4 µs — 5.9× faster than scipy kmeans2's fixed
  50 iters** (scipy.cluster.vq.kmeans2 has no convergence check). fsci's early-stop is
  a real practical advantage on converged data. KEEP. (Not a per-iteration kernel
  claim — it converges in ~5 iters.)

### ndimage correlate + gaussian_filter (correlate = frankenscipy-e3r7e) — parity / gap
Oracle `docs/perf_oracle_ndimage.py` (scipy.ndimage, 256² image).
- **correlate 5×5: fsci 1099 µs vs scipy 933.7 µs = 0.85× (1.18× slower).** Near-
  parity. The `e3r7e` precomputed tap-delta table is byte-identical (not a regression);
  fsci's direct correlation is within 18% of scipy's C. KEEP.
- **gaussian_filter σ=2: fsci 3238 µs vs scipy 1143 µs = 0.35× (2.83× SLOWER).** NOT my
  optimization. fsci IS separable (per-axis `gaussian_filter1d_axis` passes), so the
  gap is a slow 1D convolution kernel vs scipy's tuned C `correlate1d` — a SIMD/inner-
  loop opportunity (same class as kmeans2/pdist), not a parallelization. Noted for the
  ndimage owner; not reverted (not mine, not a regression).

### Sparse SpMV — `spmv_csr` (serial baseline) — small-n WIN, scale LOSS
Oracle: scipy.sparse.random CSR `.dot(x)` (same n/density; SpMV time≈O(nnz)).
`spmv_csr` is a plain serial row-sweep (NOT the parallel internal `csr_matvec`).
- **n=100 nnz=500: fsci 0.81 µs vs scipy 3.45 µs = 4.26× FASTER.** At tiny sizes
  scipy's Python/numpy dispatch overhead (~µs) dwarfs the actual SpMV; fsci is a
  direct Rust call with none. Real advantage for many-small-SpMV workloads.
- **n=1000/10000: fsci 2.2× / 1.5× SLOWER** (17.97 vs 8.14 µs; 197.5 vs 131.6 µs).
  scipy's C SpMV kernel out-throughputs fsci's scalar row-sweep; the gap narrows with
  n (overhead amortizes). Same SIMD-class inner-kernel gap as pdist/kmeans2/gaussian.
- **Nuance for release:** fsci's zero-overhead calls WIN on small/repeated ops where
  scipy pays Python tax; scipy's vectorized C WINS on large single kernels. Not a
  regression, not a parallelization issue. SpMV is a candidate for a SIMD/unrolled
  row-sweep if large sparse problems matter.

### Gaussian KDE evaluate_many (parallel) — ✅ KEEP (marquee win)
Oracle `docs/perf_oracle_kde.py` (scipy.stats.gaussian_kde, scott bw, n=1000/5000
1-D, evaluate at n points). **fsci 17–18× FASTER:** n=1000 1.06 ms vs scipy 19.09 ms;
n=5000 11.96 ms vs scipy 201.2 ms. KDE evaluation is O(m·n_data) with HEAVY per-point
work (a full sum over the dataset per query) — exactly the profile where the ordered-
slots parallelization pays off, the mirror image of the (reverted) interpolate case
(~30 flops/point). scipy's gaussian_kde is a non-vectorized Python/broadcast path,
which fsci's parallel Rust crushes. **This is the cleanest validation of the gauntlet's
central lesson: parallelize HEAVY per-element work (KDE ✅), not cheap (interpolate ❌).**
KEEP. Conformance green.

### Multiscale graph correlation (MGC) — ✅ KEEP (marquee win)
Oracle `docs/perf_oracle_mgc.py` (scipy.stats.multiscale_graphcorr, n=80, reps=100).
**fsci 21.58 ms vs scipy 295.7 ms = 13.7× FASTER.** MGC is one of scipy's slowest
functions — a pure-Python permutation loop (reps × the O(n²) statistic). fsci's
`mgc_map` is the O(n⁴)→O(n²) prefix-sum form AND the `reps` permutation scoring is
parallelized. Double lever (better asymptotics + parallel heavy work) → big win vs
scipy's non-vectorized path. KEEP. Conformance green.

### Spatial Rotation.apply_many (frankenscipy-w7ocv) — ✅ KEEP (win)
Oracle `docs/perf_oracle_xform.py` (scipy.spatial.transform.Rotation.apply, 8192 pts).
**fsci 12.03 µs vs scipy 28.30 µs = 2.35× FASTER.** The `apply_many` batch path builds
the 3×3 rotation matrix ONCE then applies in a tight Rust loop — 4.5× over the naive
per-point `map(apply)` (54 µs). NOTABLE: this is a CHEAP per-point op (3×3 matvec, ~9
flops) yet fsci WINS at n=8192 — because the kernel is REGULAR (dense matrix, linear
access) and scipy's Rotation.apply carries numpy dispatch + intermediate-array overhead.
Refines the boundary: fsci beats scipy on regular low-overhead batch kernels even when
cheap; it loses only on IRREGULAR kernels where scipy's C is tightly tuned (SpMV gather,
pdist). KEEP. Conformance green.

### Text I/O: loadtxt / savetxt (frankenscipy-fwnb1, d1uxy) — ✅ KEEP (win)
Oracle `docs/perf_oracle_io.py` (numpy.loadtxt/savetxt, 500×20 matrix). **fsci 6.7–7.8×
FASTER:** loadtxt 259.5 µs vs numpy 2022 µs (7.79×); savetxt 631.6 µs vs numpy 4208 µs
(6.66×). numpy's text I/O is pure-Python parsing/formatting; fsci's direct-parse (parse
straight into the output buffer) + `write!`-into-buffer crush it. Same family as
KDE/MGC — fsci wins decisively where the original leans on non-vectorized Python. KEEP.
Conformance green.

### KDTree build + query (frankenscipy-9k50g query) — ✅ KEEP (parity + win vs ELITE C)
Oracle `docs/perf_oracle_kdtree.py` (scipy.spatial.cKDTree, 4096 3-D points). cKDTree
is one of scipy's most-optimized C structures — the hardest target in this suite.
- **build: fsci 809.5 µs vs scipy 767.8 µs = 0.95× (PARITY).** fsci's O(n) select_nth
  median build matches elite C within 5%.
- **query: fsci 1756.7 µs vs scipy 2032.8 µs = 1.16× FASTER.** The dual-tree parallel
  query (`9k50g`) edges out scipy's single-threaded cKDTree — and this is UNDER
  multi-agent contention, so single-tenant the margin is larger.
**Significance:** even vs scipy's BEST C (not a Python path), fsci reaches parity and
WINS on the parallelizable half. This narrows the "irregular-kernel loss" further: the
losses are specific to SpMV-gather and pdist's tight C inner loop, NOT tree/spatial
structures generally. KEEP both. Conformance green.

### Silhouette score (per-anchor parallel) — ✅ KEEP (win, regression-hunt NEGATIVE)
Oracle `docs/perf_oracle_silhouette.py` (sklearn.metrics.silhouette_score, blobs).
**fsci 2.86×→10.6× FASTER:** n=500 720.8 µs vs sklearn 2064 µs; n=2000 3113.5 µs vs
32928 µs. Ran this specifically to HUNT for a second over-eager-parallelization
regression (like interpolate) — found NONE: the per-anchor work is O(n·d) (heavy,
unlike interpolate's ~30 flops), so the parallel gate is well-calibrated and even
n=500 wins 2.86×. The ratio grows with n (parallel scales). KEEP. This negative
regression-hunt result is itself evidence: the cluster/spatial parallelizations are
correctly gated; interpolate was the lone over-eager case (already reverted).

### ndimage zoom — ⚠️ LARGEST LOSS (kernel, not my optimization; bead filed)
Oracle `docs/perf_oracle_zoom.py` (scipy.ndimage.zoom 2×, 256² image). **fsci LOSES
2.25–17.7×:** order=1 85.95 ms vs scipy 4.84 ms (17.7×!); order=3 31.57 ms vs 14.05 ms
(2.25×). **Smoking gun: fsci order=1 (86 ms) is SLOWER than order=3 (31.6 ms)** — the
reverse of correct (bilinear should be cheaper than cubic). The output-pixel loop IS
parallelized (fill_pixels_parallel, gate `pixels·kernel_work≥2¹⁸` fires for both), so
this is NOT the parallelization — it's the per-pixel `sample_interpolated` computing
B-spline weights generically with no fast low-order (bilinear) special case, so order=1
pays nearly the full generic-spline cost AND apparently more (likely a per-pixel
prefilter/weight recompute). **NOT my optimization (the geometric-transform
parallelization is correct/byte-identical; the slow kernel is the underlying spline
interpolation). Bead filed for the ndimage owner.** This is the gauntlet's single
biggest loss and the clearest fix-target: special-case order≤1 (direct bilinear) +
hoist any per-pixel weight setup. Honest LOSS recorded.
- **REPRODUCED (2nd run, integrity check):** order=3 STABLE (31.6→33.2 ms) while
  order=1 consistently far slower (86→133 ms). So the anomaly is a REAL kernel
  pathology, NOT contention variance (a contention spike would have hit order=3 too).
  Refined hypothesis: for cheap order=1 pixels, the per-pixel `thread_local`
  INTERP_SCRATCH borrow + generic B-spline weight path dominates — overhead that is
  amortized away by order=3's heavier interpolation. Fix = add an order≤1 fast path
  (direct linear weights, no thread_local borrow per pixel). Bead `wm14d` confirmed.
- **EXACT ROOT CAUSE (round 14 code read):** in `sample_interpolated`, order 2..=5 use
  the fast `cardinal_reflect_nearest` path (`cardinal_bspline` direct weights, gated
  `matches!(order, 2..=5)`) and order==3 has a Wrap/Constant fast path — but **order=1
  is excluded from BOTH** and falls through to the generic `uniform_interpolation_knots`
  + `eval_bspline_basis_all` (full knot-vector B-spline basis eval per pixel per axis),
  which is far slower than the cardinal kernel. order=0 has its own fast branch. So
  order=1 is the LONE interpolating order with no fast path. **Fix: extend the cardinal
  path to order=1 (`matches!(order, 1..=5)`)** — `cardinal_bspline(1, cc-k)` over the
  3-tap span yields the linear weights `[(floor,1-t),(floor+1,t)]`. CAVEAT: must verify
  byte-identity vs `eval_bspline_basis_all` for order=1 (the linear B-spline weights are
  mathematically equal but the FP computation order differs — needs the conformance
  tests, which exceed this phase's build/bench allowance). Reduced to a ~1-line fix +
  a byte-identity check for the ndimage owner. Bead `wm14d` updated.

### kendalltau (inversion-count O(n log n)) — ✅ KEEP (win small-n, parity at scale)
Oracle `docs/perf_oracle_kendall.py` (scipy.stats.kendalltau, same x/y). fsci **2.59×
faster at n=2048** (230.4 vs 597 µs) but **parity at n=4096** (552.4 vs 537 µs, 0.97×).
scipy has a ~500 µs fixed overhead (array conversion + tie-handling setup) that
dominates at smaller n; at n=4096 both O(n log n) merge-sort kernels converge. Honest
read: fsci's algorithmic kernel MATCHES scipy's C (parity at scale) and WINS where
scipy's per-call overhead dominates — the same low-overhead advantage seen in
SpMV-small/Rotation/transform. KEEP. Conformance green.

### Delaunay (Bowyer-Watson) — ⚠️ COMPLEXITY gap vs Qhull (first asymptotic gap found)
Oracle `docs/perf_oracle_delaunay.py` (scipy.spatial.Delaunay = Qhull, 2-D). fsci
**3.3→5.9× SLOWER:** n=1000 6.53 ms vs 1.98 ms; n=2000 26.31 ms vs 4.49 ms. The
SCALING is the finding: **fsci 6.53→26.31 ms = 4.0× per 2× n (≈O(n²))** while Qhull is
2.27× (≈O(n log n)). Unlike every prior loss (constant-factor kernel gaps), this is an
ASYMPTOTIC gap — fsci's Bowyer-Watson does linear point-location (walk all triangles)
instead of a spatial-index-accelerated locate. The `8d2z2` buffer hoist is a real
constant-factor win but sits atop an O(n²) algorithm, so the gap WIDENS with n. NOT a
regression (the hoist helps); the complexity is the underlying triangulation. Fix:
spatial-accelerated point location (grid/quadtree/jump-and-walk) → O(n log n). Bead
filed for the spatial owner. This is the highest-leverage spatial fix (the others are
SIMD constant-factor; this is an algorithm-class change).

## Release-readiness summary (CrimsonForge beads, as of this round)

5 beads measured head-to-head vs scipy/sklearn (release, 64 cores, multi-agent load):
- **3 KEEP (wins):** GMM E-step parallel (4–11×), distribution pdf_many/pmf_many
  (3–98×). These are release-ready marquee wins.
- **1 KEEP (parity):** AP responsibility parallel (1.03–1.28× vs sklearn; the
  parallelization itself is a real 2× internal). Acceptable; not a differentiator.
- **1 REVERT:** interpolate `evaluate_many` parallel (0.88× regression) — reverted,
  byte-identical hoist preserved.
- **1 LOSS (not mine):** spatial pdist 2.7–9× slower than scipy → bead `nm8ex`
  filed for the spatial owner (over-eager gate + slow serial kernel).

**Cross-cutting lesson for release:** parallelize only HEAVY per-element work. The
work-gate threshold must scale with the COST of the work unit, not just its count —
2¹⁶ gaussian/exp evals (GMM) is worth parallelizing; 2¹⁶ flops (interpolate, pdist
low-dim) is not. Gates expressed as raw `count·dim` flop-products fire too early for
cheap kernels. Recommend auditing every `< 1 << 1x` parallel gate in the codebase
against the per-element op cost.

### Updated tally (11 beads, rounds 1–3)
- **WINS (algorithmic / byte-identical lever + heavy work):** GMM 4–11×, distribution
  3–98×, kmeans early-stop 5.9×, cophenet 1.83×, AP parallel 2.02× internal.
- **PARITY (within ~20% of scipy's tuned C):** AP vs sklearn, linkage 0.83×,
  correlate 0.85×.
- **REVERTED:** interpolate evaluate_many parallel (0.88×).
- **KERNEL GAPS (fsci scalar inner loop vs scipy SIMD/C; flagged, not mine to fix):**
  pdist 2.7–9× (`nm8ex`), kmeans2 2.4× (`9g6ku`), gaussian_filter 2.83×.

**Emerging release pattern:** fsci WINS decisively where the lever is algorithmic
(better asymptotics, early-stop, normalizer-hoist) and the work is heavy; it reaches
PARITY-to-LOSS on tight inner numeric kernels (distance, 1D convolution, centroid
assignment) where scipy's C is SIMD-vectorized and fsci's is scalar. **The highest-
leverage release work is SIMD-vectorizing those 3–4 inner kernels** (`nm8ex`/`9g6ku`
+ ndimage 1D filter) — NOT more threads (cheap-per-element parallelism regresses, as
the interpolate revert proves). The byte-identical alloc/precompute/batch wins are all
safe KEEPs by construction.

## Notes / negative evidence
- The ~50 byte-identical allocation/precompute/batch wins (buffer reuse, mem::take,
  loop-invariant hoist, interval binary-search, write!-amplification, retain) carry
  **no correctness-regression risk** and are monotone by construction (removing an
  alloc/recompute cannot be slower), so they are NOT individually re-benched here;
  the gauntlet revert-risk lives in the **parallelizations** (spawn overhead at small
  n) — those are gate-validated above.
- AP availability-update parallelization is the one OPEN lever surfaced by measurement.
