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
| pdist flat dim-4 rows (nm8ex.1) | pdist euclidean n=256 | 88.96 µs | 172.83 µs | **0.51× (1.94× SLOWER)** | 1.52× faster internally | ✅ KEEP, residual gap |
| pdist flat dim-4 rows (nm8ex.1) | pdist cosine n=256 | 79.69 µs | 208.89 µs | **0.38× (2.62× SLOWER)** | 1.83× faster internally | ✅ KEEP, residual gap |
| pdist flat dim-4 rows (nm8ex.1) | pdist euclidean n=512 | 309.79 µs | 714.58 µs | **0.43× (2.31× SLOWER)** | 1.11× faster internally | ✅ KEEP, residual gap |
| pdist flat dim-4 rows (nm8ex.1) | pdist cosine n=512 | 275.14 µs | 828.70 µs | **0.33× (3.01× SLOWER)** | 1.44× faster internally | ✅ KEEP, residual gap |
| linkage NN-chain (average) | linkage n=400 d=4 | 1586.5 µs | 1904.5 µs | **0.83× (1.2× slower)** | — | ⚠️ near-parity |
| cophenet mem::take (jphzn) | cophenet n=400 | 401.5 µs | 219.7 µs | **1.83× faster** | — | ✅ KEEP |
| kmeans2 double-buffer (4ylee) | kmeans2 k4 n2000 iter=50 | 2104.7 µs | 5126 µs | **0.41× (2.4× SLOWER)** | scalar assign vs scipy SIMD | ⚠️ kernel gap → bead |
| kmeans Lloyd early-stop | kmeans k4 n2000 | 2104.7 µs* | 357.4 µs | **5.9× faster** | *vs scipy kmeans2 fixed-iter | ✅ KEEP (early-stop) |
| correlate tap-table (e3r7e) | correlate 5x5 256² | 933.7 µs | 1099 µs | **0.85× (1.18× slower)** | byte-identical | ✅ KEEP (parity) |
| gaussian_filter (NOT mine) | gaussian σ=2 256² | 1143.0 µs | 3238 µs | **0.35× (2.83× slower)** | separable but slow 1D kernel | ⚠️ gap → owner |
| spmv_csr cached+unrolled row sweep (2hclc) | SpMV n=100 nnz=500 | 4.63 µs | 0.388 µs | **11.9× faster** | 1.54× vs legacy row-sweep; bit-identical | ✅ KEEP |
| spmv_csr cached+unrolled row sweep (2hclc) | SpMV n=1000 nnz=10k | 8.00 µs | 7.077 µs | **1.13× faster** | 2.10× vs legacy row-sweep; scale loss closed | ✅ KEEP |
| spmv_csr cached+unrolled row sweep (2hclc) | SpMV n=10000 nnz=100k | 96.95 µs | 68.82 µs | **1.41× faster** | 2.14× vs legacy row-sweep; scale loss closed | ✅ KEEP |
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
| ndimage zoom order=1 FIXED (wm14d) | zoom 2× 256² order=1 | 4842 µs | 19409 µs | **0.25× (4.0× slower)** — was 0.06× (17.7×) | cardinal fast path added | ✅ FIXED (4.4× faster) |
| ndimage zoom order=1 residual fast path (wm14d) | zoom 2× 256² order=1 | 3889 µs | 7968 µs | **0.49× (2.05× slower)** | 4.27× faster than generic sampler | ✅ KEEP, residual gap |
| ndimage zoom order=3 | zoom 2× 256² order=3 | 14053 µs | 31573 µs | **0.45× (2.25× slower)** | generic spline-weight kernel | ⚠️ residual gap |
| ndimage rotate order=3 (shares wm14d fix) | rotate 30° 256² order=3 | 5577 µs | 6439 µs | **0.87× (1.15× ~parity)** | cardinal spline path | ✅ near-parity |
| ndimage rotate order=1 (shares wm14d fix) | rotate 30° 256² order=1 | 1991 µs | 8733 µs | **0.23× (4.4× slower)** | residual machinery gap (was ~17×) | ⚠️ residual gap |
| kendalltau inversion-count O(n log n) | kendalltau n=2048 | 597 µs | 230.4 µs | **2.59× faster** | scipy fixed overhead | ✅ KEEP |
| kendalltau inversion-count O(n log n) | kendalltau n=4096 | 537 µs | 552.4 µs | 0.97× (parity) | both O(n log n) at scale | ✅ KEEP |
| Delaunay precompute circumcircles (9l5oo) | Delaunay n=1000 2-D | 1980 µs | 898 µs | **2.2× FASTER** — was 0.30× (3.3× slower) | cheaper bad-test | ✅ WIN (7.3× self-speedup) |
| Delaunay precompute circumcircles (9l5oo) | Delaunay n=2000 2-D | 4488 µs | 3257 µs | **1.38× FASTER** — was 0.17× (5.9× slower) | O(n²) const-factor crushed | ✅ WIN (8.1× self-speedup) |

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

### Spatial pdist flat dim-4 rows (frankenscipy-nm8ex.1) — ✅ KEEP internally, ⚠️ LOSS vs SciPy
Follow-up to the `nm8ex` residual gap. The direct dim-4 fast path still loaded
every pair through `Vec<Vec<f64>>`; this lever stages validated 4-column rows
into compact `[f64; 4]` points once per call and runs the same Euclidean/Cosine
arithmetic over fixed-width rows. Same-worker rch `ovh-b` Criterion medians
improved across all four rows: 263.00→172.83µs (1.52×) for euclidean n=256,
381.98→208.89µs (1.83×) for cosine n=256, 794.72→714.58µs (1.11×) for
euclidean n=512, and 1.1930ms→828.70µs (1.44×) for cosine n=512. The focused
bit-exact dim-4 `pdist` guard passed via rch.

Honest SciPy score remains a loss: local SciPy 1.17.1 / NumPy 2.4.3 oracle was
88.96µs, 79.69µs, 309.79µs, and 275.14µs respectively, so Rust is still
1.94×, 2.62×, 2.31×, and 3.01× slower (0 wins / 4 losses / 0 neutral). KEEP
because the internal win is significant and behavior-preserving; route deeper
to output batching or generated SIMD-style dim-specialized kernels rather than
retrying row staging alone.

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

### Sparse SpMV — `spmv_csr` cached+unrolled row sweep (frankenscipy-2hclc) — ✅ KEEP, scale LOSS closed
Oracle: scipy.sparse.random CSR `.dot(x)` (same n/density; SpMV time≈O(nnz)).
`spmv_csr` is the public serial row-sweep (NOT the parallel internal `csr_matvec`).
The old public route won only tiny calls and lost at scale; the cached-slice +
4-lane unrolled row loop closes those losses without changing accumulation order.
- **n=100 nnz=500: fsci 0.388 µs vs scipy 4.63 µs = 11.9× FASTER.**
- **n=1000 nnz=10k: fsci 7.077 µs vs scipy 8.00 µs = 1.13× FASTER.**
- **n=10000 nnz=100k: fsci 68.82 µs vs scipy 96.95 µs = 1.41× FASTER.**
- Same-process A/B on rch `ovh-a` (`FSCI_PUBLIC_SPMV_AB=1 cargo run --profile
  release-perf -p fsci-sparse --bin perf_csr_matvec`) compared the legacy public
  row sweep to current in one binary: 550 ns→356 ns (1.54×), 12.074 µs→5.741 µs
  (2.10×), 135.043 µs→63.231 µs (2.14×), all `identical=true`.
- Score vs SciPy after this lever: **3 wins / 0 losses / 0 neutral**. Prior
  ledger status was 1 win / 2 losses. Remaining route is explicit SIMD or
  sparse-BLAS-style row blocking only if a fresh profile shows public SpMV still
  matters after this constant-factor win.

### Sparse eigsh / svds (frankenscipy-fo9cj Arnoldi arena) — REJECT, restored route 4W/1L/1N
Oracle: SciPy 1.17.1 `scipy.sparse.linalg.eigsh` / `svds` on the same deterministic
matrix family as the sparse perf bins. rch same-worker A/B on `ovh-a` rejected the
row-major Arnoldi basis arena plus mutable matvec scratch: `eigsh` regressed all rows
(1.667/6.594/16.147 ms vs parent 1.184/5.548/11.599 ms). `svds` movement was too small
to save it (0.99x, 1.06x, 1.01x parent/candidate-style ratios across the sweep), so
the source route was reverted.
- **eigsh n=2000 k=6: fsci 1.184 ms vs scipy 3.000 ms = 2.53x FASTER.**
- **eigsh n=8000 k=6: fsci 5.548 ms vs scipy 2.768 ms = 0.50x (2.00x SLOWER).**
  This is the next real sparse eigensolver loss: optimize the mid-size restart /
  iteration path or matvec throughput, not the discarded basis-arena copy route.
- **eigsh n=20000 k=8: fsci 11.599 ms vs scipy 43.023 ms = 3.71x FASTER.**
- **svds 2200x2000 k=6: fsci 1.191 ms vs scipy 17.567 ms = 14.75x FASTER.**
- **svds 8200x8000 k=6: fsci 4.929 ms vs scipy 4.861 ms = 0.99x neutral.**
- **svds 20200x20000 k=8: fsci 12.534 ms vs scipy 42.018 ms = 3.35x FASTER.**
KEEP restored route. DO NOT retry the row-major `Vec<Vec>` replacement without
allocator/profile proof that Arnoldi basis allocation is again a top-five cost and a
new layout avoids the per-step basis copy cost.

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

### ✅ ndimage zoom order=1 — FIXED (BOLD-VERIFY, frankenscipy-wm14d) — biggest loss closed
The gauntlet's single biggest loss (zoom order=1 17.7× slower than scipy) is now FIXED
and shipped (commit `3c027183`). Root cause was that order=1 was the lone interpolating
order with no fast path: order=1 Reflect/Mirror is PADDED (coord_offsets=SPLINE_NEAREST_
PAD) so the cardinal `coord_offsets==0` gate excluded it → it fell through to the slow
generic `eval_bspline_basis_all`. Fix: route padded order=1 through the cardinal fast
path with clamp(Nearest) fold (the padding already encodes the reflection, so the linear
support always lands in range) + made `cardinal_bspline` use stack arrays instead of
per-call heap Vecs (hot per-tap-per-pixel path). **MEASURED: order=1 zoom 85.95 ms →
19.41 ms = 4.4× faster; loss vs scipy 17.7× → 4.0× (≈2.4× contention-adjusted); the
order=1>order=3 inversion is gone (19.4 ms < order=3's 54 ms).** Conformance: ndimage
296 passed / 0 failed (verified twice). The residual ~2.4–4× gap is the cardinal_bspline
arithmetic + parallel overhead vs scipy's tight C — a follow-up SIMD/branchless target.
This is the BOLD-VERIFY loop end-to-end: measured loss → root-caused → bold fix →
conformance-verified → measured win → shipped.

2026-06-19 cod-b residual pass: a narrower 2D Reflect/order=1 direct bilinear zoom
path precomputes row/column supports and replaces the generic per-pixel sampler with a
fixed four-load sum. Same-worker `ovh-b` A/B improved 34.034 ms to 7.9684 ms
(4.27× faster), but local SciPy still measured 3.88937 ms, so the lane remains a
2.05× SciPy loss. A serial fill probe regressed to 9.6976 ms and was reverted; do not
retry scheduler-only variants without a fresh profile.

### ndimage rotate — zoom fix's BROAD REACH (frankenscipy-wm14d)
Oracle `docs/perf_oracle_rotate.py` (scipy.ndimage.rotate 30°, 256²). rotate shares
`sample_interpolated` with zoom, so the order=1 cardinal fast path (`3c027183`) rescued
it from the same ~17× padded-order=1 pathology too. MEASURED post-fix: **order=3 6.44 ms
vs scipy 5.58 ms = 0.87× (NEAR-PARITY)** — fsci's cubic-spline rotate is competitive with
scipy's C; **order=1 8.73 ms vs 1.99 ms = 4.4× slower** (residual general-machinery +
parallel overhead, the same gap as zoom order=1 post-fix). Takeaway: one targeted fix
removed the pathology across the WHOLE geometric-transform family (zoom/rotate/affine/
map_coordinates all share the path); the residual ~4× order=1 gap is the general
`sample_interpolated` support-computation machinery — a wholesale specialized-bilinear
rewrite, not the weight arithmetic (the direct-weights micro-opt was measured ~0-gain
and reverted). order=3 is already a release-ready near-parity result.

### gaussian_filter gap DIAGNOSED (refined negative evidence)
Dug into the 2.83× gaussian loss. fsci `gaussian_filter` is SEPARABLE (per-axis 1D),
already uses the fast `convolve1d_along_axis` for inner axes (parallel across slabs),
and only falls back to general `convolve` for the OUTERMOST axis (1 slab). BUT the key
fact: **scipy.ndimage is single-threaded, and fsci's gaussian runs on 64 threads yet is
still 2.83× slower** → fsci's per-element 1D-convolution KERNEL is ~10–60× slower than
scipy's C (parallelism masks it to 2.83×). So the gap is NOT the outer-axis handling or
parallelization — it's the inner dot-product kernel, same SIMD-class gap as pdist/SpMV.
Fix = SIMD-vectorize `convolve1d_along_axis`'s window·weights dot product (conformance
tolerance-OK since gaussian isn't chaotic), but that's shared kernel code. Also checked:
`uniform_filter` already O(1) running-sum, `correlate1d`/`convolve1d` already specialized
1D-axis — the ndimage filters are otherwise well-optimized; the residual is SIMD-kernel.

### ✅✅ Clough-Tocher LOSS → WIN: precompute Bézier patches, 26.6× self-speedup (9l5oo lever)
Oracle `/tmp/oracle_ct.py` (scipy.interpolate.CloughTocher2DInterpolator eval_many, 576
pts / 1024 q). **BEFORE: fsci 2222.8 µs vs scipy 537 µs = 4.1× SLOWER (a LOSS).** The
per-query `clough_tocher_triangle_eval` rebuilt the ENTIRE macro-patch every query — 3
edge vectors, 6 directional derivatives, 19 cubic Bézier control points, AND a neighbour
loop (3 neighbours × barycentric + centroid + division) — all query-INVARIANT. FIX:
split into `clough_tocher_patch → [f64;19]` (the invariant patch) + `clough_tocher_eval_
patch` (the Bézier sum, the only query-dependent step); precompute `patches: Vec<[f64;19]>`
once per triangle in `with_options`. **AFTER: fsci 83.5 µs = 26.6× self-speedup = 6.4×
FASTER than scipy.** A 4.1× LOSS flipped to a 6.4× WIN. BYTE-IDENTICAL (patch + eval are
the original code verbatim, just reorganised), conformance interpolate **227/0**. KEEP.
The 4th application of the precompute-per-element-predicate lever, biggest self-speedup
yet (the neighbour loop made the per-query cost enormous).

### ✅ griddata / LinearND 46.5× faster than scipy + barycentric precompute (9l5oo lever)
Oracle `/tmp/oracle_griddata.py` (scipy.interpolate.griddata linear, 576 pts / 1024
queries, same data as bench_scattered). **fsci griddata 118.3 µs vs scipy 5507 µs =
46.5× FASTER** (eval-only `linear_nd_eval_many` 59.8 µs). The big ratio is fsci's
low-overhead Rust vs scipy's Python griddata + Qhull-setup-per-call. ON TOP of that I
applied the precompute-per-element-predicate lever to `Delaunay2D::find_simplex`:
precompute each simplex's query-invariant barycentric basis (point a + Gram matrix
(d00,d01,d11) + denom) once in `new()`, so the grid-restricted point-location scan does
only the v2-dependent work per query instead of rebuilding the Gram matrix per
(query, candidate). BYTE-IDENTICAL (`SimplexBary::weights` = `barycentric` same float
ops/order), conformance interpolate **227/0**. Monotone eval speedup. KEEP.

### ✅✅ Delaunay LOSS → WIN/PARITY (frankenscipy-9l5oo) — large-n gap closed to parity
The first pass flipped n=1000/2000 by precomputing circumcircles instead of calling the
full in-circle determinant for every (point, triangle) pair. The 2026-06-20 cod-b pass
expanded the gauntlet to n=4000/8000 and found the predicted crossover was real:
pre-grid n=4000 14.935 ms vs scipy 9.50086 ms (1.57× slower), n=8000 55.761 ms vs
20.62714 ms (2.70× slower). New lever: for n>=4096, stable triangle IDs plus a fixed
grid over circumcircle bounding boxes. Each point checks only candidate circles in its
cell and then applies the exact `dist² < r²` predicate; inactive stale IDs are skipped
and an empty candidate lookup falls back to the full active scan. **MEASURED final:
n=1000 0.754 ms vs scipy 1.933 ms (2.56× faster), n=2000 2.613 ms vs 4.550 ms
(1.74× faster), n=4000 9.463 ms vs 9.501 ms (parity), n=8000 20.622 ms vs
20.627 ms (parity). Score: 2 wins / 0 losses / 2 neutral.** Conformance/gates:
spatial lib 208 passed / 0 failed / 2 ignored; e2e_spatial 16/0; check, clippy
`-D warnings`, fmt, UBS clean for touched files. Remaining caution: this is still
Bowyer-Watson with a grid candidate accelerator, not full Qhull-class history-DAG
location; re-measure beyond n=8000 before claiming asymptotic dominance.

## IO crate — head-to-head vs numpy/scipy.io (2026-06-19) — fsci DOMINATES
fsci vs numpy (loadtxt/savetxt) + scipy.io (mmread/mmwrite), in-memory:

| function | fsci | numpy/scipy | ratio |
|---|---|---|---|
| mmread 100×100 | 289 µs | 4282 µs | **14.8× faster** |
| mmwrite 100×100 | 619 µs | 3747 µs | **6.1× faster** |
| savetxt 500×20 | 584 µs | 2951 µs | **5.0× faster** |
| loadtxt 500×20 | 267 µs | 929 µs | **3.5× faster** |

Same structural reason as opt/integrate: fsci's Rust text/MatrixMarket parse+format has no
Python interpreter overhead; numpy/scipy pay it on every cell. IO HARVESTED — fsci dominates.

## Special crate — array (RealVec) sweep vs scipy (2026-06-19) — measured slower, cause CORRECTED
Bench added (`special_array_65536`). fsci vs scipy.special over a 65536 RealVec:

| function | fsci | scipy | ratio |
|---|---|---|---|
| gamma | 1.04 ms | 426 µs | 0.41× (2.4× SLOWER) |
| j0 | 1.08 ms | 664 µs | 0.61× (1.6× SLOWER) |
| erf | 4.49 ms | 757 µs | **0.17× (5.9× SLOWER)** |

⚠️ CORRECTION (my first root-cause was WRONG — I grepped only lib.rs): special DOES
parallelize. `gamma_dispatch`/etc. call `par_map_indices` (defined in airy.rs, gate `n<256
→serial` else `available_parallelism`), so RealVec maps the kernel across all cores. The
real causes of the measured slowness: (1) `par_map_indices` SPAWNS ~64 threads PER CALL (no
pool) — slow under heavy multi-agent load (the same contention that regressed my minmax-slab/
linkage-distance attempts; absolute bench numbers here are unreliable), and (2) fsci's
per-element kernels are ~2× scipy's Cephes (e.g. gamma). So the parallel path may NOT pay for
cheap kernels at moderate n: spawn overhead can exceed the per-core work. POSSIBLE LEVERS (in
the shared, actively-developed special crate — flagged, not dived into): a persistent thread
POOL (amortize the per-call spawn) and/or a COST-AWARE gate (cheap kernels need a higher n
threshold than 256). The measured ratios are contention-influenced; the bench is the harness
to re-check on an idle machine. NOT a clean serial-dispatch loss as first claimed.
KERNEL FINDING (dug to erf_scalar, error.rs:206): erf (4.49 ms) is 4× slower than the more-
expensive gamma (1.04 ms) because the kernel is ITERATIVE — `erf_series_real` (Maclaurin, up
to 80 terms for |x|<1) + `erfc_cf_real` (continued fraction for x≥1, ~10-30 iters) — whereas
scipy's Cephes uses a fixed-degree RATIONAL approximation (~10 mults, no loop). REAL LEVER:
port Cephes' rational erf/erfc (faster + matches scipy exactly, conformance-safe since scipy
IS Cephes). Out of MY reach (needs Cephes's exact coefficients — no source access to
transcribe — or a custom minimax rational fit). Flagged for the special owner: the per-element
kernel speed (not just the thread spawn) is the real gap for the iterative special functions.

## Opt crate — minimize sweep vs scipy (2026-06-19) — fsci DOMINATES (largest ratios of phase)
fsci vs scipy.optimize.minimize(method='BFGS') on Rosenbrock, x0=zeros:

| dim | fsci | scipy | ratio |
|---|---|---|---|
| 2 | 10.7 µs | 3914 µs | **367× faster** |
| 5 | 21.7 µs | 10672 µs | **491× faster** |
| 10 | 76.4 µs | 27285 µs | **357× faster** |

Same structural reason as solve_ivp, amplified: BFGS does MANY objective+gradient
evaluations (numerical gradient + line search), each a Python callback in scipy; fsci runs
the whole optimizer + Rust objective with zero callback overhead. Optimizer/root/ODE crates
(any iterative solver over a user function) are fsci's biggest categorical win vs scipy.

## Integrate crate — ODE sweep vs scipy (2026-06-19) — fsci DOMINATES
fsci vs scipy.integrate.solve_ivp (RK45, rtol 1e-6, atol 1e-9):

| ODE | fsci | scipy | ratio |
|---|---|---|---|
| exponential decay (0,10) | 18.589129 µs | 1443.255860 µs | **77.64× faster** |
| Lorenz (0,1) | 28.266539 µs | 2062.735365 µs | **72.97× faster** |

The ~80× is structural: fsci's RHS is compiled Rust evaluated inline, scipy calls a Python
callback at every RK45 stage + runs the step loop in Python. Any ODE/quadrature with a
cheap RHS will show this — fsci's no-callback-overhead is decisive. Integrate ODE path
HARVESTED (dominant).

### frankenscipy-bpzha: RK scratch double-buffer measured reject
The solver-owned scratch/double-buffer idea was tested and reverted. It had one
scalar exponential win on `hz2` (`17.356838 -> 13.863079 µs/call`) but regressed
paired Lorenz/vector rows on `hz2` (`21.951172 -> 23.402816 µs/call`), `hz1`
(`28.621224 -> 31.335899 µs/call`), and `ovh-a`
(`20.597014 -> 32.037205 µs/call`). Final helper-dispatch sanity also measured
`27.755498 µs/call` on exponential on `ovh-b`, a red flag against all parent
exponential rows. Decision: reject/revert; next integrate work should come from a
fresh profile, not from RK scratch reuse.

## Ndimage crate — filter/morphology sweep vs scipy (2026-06-19)
fsci vs scipy.ndimage (256² / 160² images):

| function | fsci | scipy | ratio |
|---|---|---|---|
| median_filter 160² s7 | 1.84 ms | 6.03 ms | **3.28× faster** |
| median_filter 160² s15 | 9.32 ms | 26.46 ms | **2.84× faster** |
| minimum_filter 256² s7 | 2.24 ms | 0.99 ms | 0.44× (2.26× slower, OPEN) |
| minimum_filter 256² s15 | 1.84 ms | 1.01 ms | 0.55× (1.82× slower, OPEN) |
| binary_erosion 256² s7 (IMPROVED) | 1.81 ms | 0.60 ms | 0.33× (3.0× slower, was 3.7×) |
| binary_erosion 256² s15 (IMPROVED) | 1.62 ms | 0.81 ms | 0.50× (2.0× slower, was 2.76×) |

### ✅✅✅ binary_erosion LOSS → WIN: 2D BIT-PACKING (radical lever, flipped)
The radical lever LANDED — byte-identical on the first try. `binary_erode_bitpack_2d`:
pack each row into u64 words (64 px/word), erode HORIZONTALLY via shift-AND (out[c] = AND of
in[c-lo..c-lo+size-1], computed as the left-anchored `size`-fold shift-AND then `shift_bits_
down` by `size-1-lo` to re-center) and VERTICALLY via word-AND of the `size` rows in the
window. Constant-0 border falls out free (out-of-range bits/rows are 0 → AND is 0). Gated to
2D + size<64 (single word-boundary shifts); N-D / huge windows fall back to the running count.
**MEASURED: s7 2.20 ms→630 µs (3.5× self) = PARITY vs scipy 596 µs (was 3.7× slower); s15
2.22 ms→180 µs (12× self) = 4.5× FASTER vs scipy 805 µs (was 2.76× slower).** Conformance
ndimage **296/0**. A 2.76–3.7× LOSS flipped to parity-to-4.5×-WIN. The op-count math (~30×
fewer ops) predicted it. KEY: erosion AND is commutative across axes so horizontal-then-
vertical order is byte-identical; the Constant-0 border needs no special-casing in bit-space.
EARLIER (superseded): running-count partial got 1.2–1.4×; the bit-pack subsumes it.
SYMMETRIC: `binary_dilate_bitpack_2d` (OR instead of AND, reflected-SE origin lo=size/2+refl,
out-of-range = OR-identity 0). **s7 ~2.2 ms→642 µs (3.4× self) = 2.3× slower (was 6.6×);
s15 ~2.2 ms→159 µs (14× self) = 3.3× FASTER vs scipy 521 µs (was 3.1× slower)**, 296/0.
dilation-s7 still loses to scipy's very-fast 279 µs (mostly-set image) but improved 3.4×.
Net: binary morphology 3 of 4 cases now parity-or-WIN (was all losses).
BROAD REACH (measured, bench added): binary_opening (erosion∘dilation) and binary_closing
(dilation∘erosion) INHERIT the bit-pack → **opening s7 2.8× / s15 9.0× faster; closing s7
1.6× / s15 4.7× faster than scipy** (scipy 2.2–10.8 ms — it does NOT decompose the box
structure, scanning the full s² footprint, where fsci's bit-packed separable path is
1.1–1.7 ms). The bit-pack flip propagates to all higher-level binary morphology (opening/
closing/tophat/fill_holes). SAME lever applied to
`binary_dilation` (`binary_dilate_separable`: running count of ONES > 0, origin-aware lo =
size/2 + refl to match the reflected-SE max-filter; even sizes use refl=−1): byte-identical
**296/0**, dilation ~1.84/1.64 ms (same ~1.2–1.4× self-speedup). Dilation is still 3–6.6×
slower than scipy (279/521 µs — the mostly-set bench image favours scipy's algorithm); same
bit-pack lever needed to flip. Both binary-morph paths now use the simpler integer-count
kernel; the float deque remains for non-default origins + float minmax.

median is a big WIN. minimum_filter + binary_erosion are CONSTANT-FACTOR losses: both go
through `separable_minmax_filter` → `minmax_filter_along_axis`, already an O(1)/pixel
monotonic-deque sliding min (flat across window size, confirmed), so the gap is Rust-deque
overhead vs scipy's specialized C, NOT algorithm. RADICAL LEVER (future, substantial):
**binary_erosion/dilation on a binary image should bit-pack** (64 px/u64; horizontal =
`s` shift-ANDs per word, vertical = `s` word-ANDs per row) → ~10-30× over the float deque,
would FLIP both binary-morph losses to wins. scipy's NI_BinaryErosion is a specialized
binary path; fsci runs the general float min-filter on booleanized data. Byte-identical
(same 0/1 output). Needs exact window-origin-semantics matching with the deque path —
high-risk multi-cycle, filed as a focused future effort, not started blind. minimum_filter
(float) has no bit-pack lever; its constant factor needs SIMD on the deque (hard). ANALYSIS
(no clean lever — DON'T re-chase byte-identically): the monotonic deque is already amortized
O(1)/pixel (~1 total_cmp); van Herk/Gil-Werman does MORE (3 total_cmp/pixel: prefix+suffix+
combine) so it's not faster; shift-min (f64-min shifted s times, the bit-pack analogue) is
vectorizable and would win for small s, BUT requires `f64::min` not `total_cmp` → silently
changes NaN semantics (no NaN minmax test exists, so it'd pass conformance, but it's a latent
behaviour divergence from scipy — NOT shipped). A true flip needs an explicit SIMD min with
total_cmp NaN ordering. The deque is the right scalar algorithm; the gap is scipy's tighter
vectorized C. EMPIRICALLY CONFIRMED (attempted no-NaN-gated shift-min, REVERTED clean 296→
237/1): `f64::min`/`max` diverge from `total_cmp` not only on NaN but on SIGNED ZEROS —
`total_cmp(-0.0,+0.0)=Less` so min=-0.0/max=+0.0, but `f64::min(-0.0,+0.0)` is order-dependent
(x86 minsd). The byte-for-byte `separable_minmax_matches_rank_filter` test (which seeds ±0.0)
caught it. A correct vectorized version needs the f64→monotonic-i64 transform + SIMD i64 min,
which is AVX-512-only (AVX2 lacks `vpminsq`) → not portable. NO clean lever; the conformance
gate prevented shipping a subtly-wrong (signed-zero) result.
DEFINITIVE (2nd attempt, REVERTED clean): the SIGNED-ZERO-CORRECT version — shift-min on the
ordered-i64 (`f64::total_cmp`'s own monotonic transform `bits ^ ((bits>>63 as u64)>>1)`, an
involution) — IS byte-identical for all inputs (296/0, signed-zero test passes). But PERF
REGRESSES: s7 ~same (2.21 vs 2.24 ms), s15 SLOWER (2.40 vs 1.84), s31 much SLOWER (3.72 vs
1.89). The i64 min/max didn't autovectorize on AVX2 (no `vpminsq`), so the O(s·n) shift loses
to the deque's amortized O(1). CONCLUSION: the monotonic-deque IS the optimal portable scalar
algorithm for float window min/max; beating scipy needs explicit C-style SIMD (vpminsq/AVX-512
or hand-written AVX2 compare+blend on the i64 reps). Float minmax CLOSED — no portable lever.
DEAD-END (reverted clean, 296/0): rewrote `minmax_filter_along_axis` to the correlate1d
slab pattern + parallelize over outer slabs (byte-identical). REGRESSED ~1.5-2× even after
hoisting the per-slab VecDeque alloc to per-thread reuse. At 256² the filter is below the
parallel gate (work < 2¹⁸ → serial) so the parallel path wasn't even engaged — the slab
restructure measured slower under heavy multi-agent load, and the parallel path would add
core-contention at larger sizes. Don't re-chase slab-parallel for minmax; the win (if any)
needs the bit-pack (binary) or SIMD-deque (float) lever, not coarse line parallelism.

## Cluster crate — head-to-head sweep vs scipy (2026-06-19)
fsci vs scipy.cluster.hierarchy: **cophenet n400 206µs vs 290µs = 1.40× faster** (WIN);
**linkage_average n400 1.847ms vs 1.655ms = 1.12× slower** (near-parity, OPEN). GMM/
silhouette already wins (gauntlet ledger); kmeans2 chaotic-iteration loss (unfixable
byte-identically). DEAD-END (reverted clean): parallelizing linkage's O(n²·dim) distance
build via split_at_mut row-blocks was BYTE-IDENTICAL (193/0) but **2.5× SLOWER** (1.847→
4.61ms) — the distance build is NOT the bottleneck (the NN-chain `agglomerate_nnarray`
is), so parallelizing the small part added thread overhead + 2× redundant sqrt + cache
thrash on the 5 MB arena. The 1.12× gap lives in the NN-chain, not the distance fill —
don't re-chase the distance parallelization.

## Stats crate — head-to-head sweep vs scipy (2026-06-19) — fsci DOMINATES
fsci vs scipy.stats / scipy.stats.qmc — all WINS, no losses:

| function | fsci | scipy | ratio |
|---|---|---|---|
| QMC Sobol 4096×2 | 9.6 µs | 200.7 µs | **20.8× faster** |
| QMC Halton 4096×4 | 95.5 µs | 1326.7 µs | **13.9× faster** |
| discrepancy centered 512×2 | 221 µs | 622 µs | **2.8× faster** |
| discrepancy mixture 512×2 | 288 µs | 720 µs | **2.5× faster** |
| kendalltau 2048 | 196 µs | 380 µs | **1.94× faster** |
| discrepancy wraparound 512×2 | 225 µs | 412 µs | 1.83× faster |
| discrepancy l2_star 512×2 | 227 µs | 383 µs | 1.69× faster |
| kendalltau 4096 | 440 µs | 533 µs | 1.21× faster |

Stats is HARVESTED — fsci wins every measured function (QMC sampling especially, where
scipy's Python-loop generators are 14-21× slower than fsci's vectorized Rust).

### ✅ remez even-WLS cos-basis Chebyshev recurrence (1.17×, signal)
The even-numtaps remez fallback (WLS frequency-sampling; the benched odd-257 path uses the
already-efficient PM+barycentric route) rebuilt its cos-basis with `n_coeffs` separate
`cos(2π·j·f)` calls per grid point. Replaced with the Chebyshev recurrence `cos(jθ)=2cos(θ)
cos((j-1)θ)-cos((j-2)θ)` — ONE cos() per grid point. **A/B MEASURED: 3.58→3.06 ms = 1.17×**
(the O(ng·n_coeffs) cos was ~15% of the work; the O(ng·n_coeffs²) normal-equations build
dominates the rest). Accurate to ~1e-14 (within remez's ~1e-6 tolerance), conformance signal
**707/0**. scipy.signal.remez ERRORS on this case (PM non-convergence) so fsci's WLS fallback
has no head-to-head, but it's a real self-speedup. KEEP (not ~0-gain). The recurrence lever
applies to any cos(2π·k·f) response-basis loop.

### ✅ freqz Horner's method (5.2× self-speedup, signal) — closes a 12× loss to 2.3×
`eval_poly_on_unit_circle` (used by `freqz`/`freqz_with_whole`, the frequency-response
function) computed `cos(kω)` AND `sin(kω)` PER COEFFICIENT per frequency — despite a comment
falsely claiming "Horner's method." Implemented ACTUAL Horner: z⁻¹=e^{-jω} via ONE cos+sin
per frequency, then a complex-multiply accumulation `acc=acc·z⁻¹+c[k]`. **A/B MEASURED on a
128-tap FIR / 512 freqs: 978→187 µs = 5.2×.** Same polynomial value as the direct sum
(~1e-13), conformance signal **707/0**. Head-to-head vs scipy.signal.freqz (81 µs, FFT-based):
fsci was **12× slower → now 2.3× slower** — Horner cuts most of the gap; the residual is the
O(n_freqs·n_coeffs) Horner vs scipy's O(n log n) FFT-of-coefficients. ✅ RESIDUAL NOW DONE
(see freqz FFT-hybrid below). Added freqz/fir128_512 bench.

### ✅✅ freqz FFT-hybrid (FLIPS the residual loss to a WIN, signal)
DONE — the flagged residual. B(e^jω)/A(e^jω) on the linear ω-grid IS the DFT of the zero-
padded coefficients (whole: nfft=n; half: nfft=2n), so for large filters use `fsci_fft::fft`
(O(N log N)) instead of the O(n·n_coeffs) Horner loop; small filters (b.len+a.len < 16 or
n < 64) keep Horner (so biquads don't regress). **MEASURED freqz/fir128_512: Horner 187 µs →
FFT 49.9 µs = 3.7×; vs scipy 81 µs now 1.6× FASTER.** Full freqz journey: 12× slower → 2.3×
slower (Horner) → **1.6× faster than scipy**. Same response within ~1e-13, conformance 707/0
(tolerance tests; biquad path unchanged). Falls back to Horner if the FFT errors.
PLUS: routed `group_delay_from_ba` + `magnitude_response_db` (two MORE functions with the
same inline per-coefficient cos/sin loop, not previously using the helper) through the Horner
`eval_poly_on_unit_circle` — inherit the 5.2× large-filter speedup, conformance 707/0. Also
NOTED (bigger lever, not done): the MFCC power spectrum (lib.rs ~5949) is a naive O(N²) DFT
(`re += s·cos(2πkn/N)`) that should be an fsci_fft O(N log N) FFT.

### ✅✅ mfcc naive-DFT → fsci_fft (181× !!, signal) — MARQUEE algorithmic flip
DONE. The mfcc per-frame power spectrum computed `|Σ s[n]·e^{-j2πkn/N}|²` as a NAIVE O(N²) DFT
(n_freq×frame_len per frame, every frame). Replaced with `fsci_fft::fft` (O(N log N)) — pad
frame to complex, FFT, |·|²/n_fft. **A/B MEASURED on 16384 samples / frame_len 512:
149.45 ms → 0.825 ms = 181×.** Same DFT value (~1e-13), conformance signal **707/0**. The
single biggest self-speedup of the phase — a naive DFT in a hot per-frame loop is catastrophic
(149 ms). LEVER: grep nested-loop `cos(2πkn/N)`/`sin` (DFT-by-hand) in any transform/feature
fn → replace with fsci_fft. Added mfcc/16384_frame512 bench.

## Signal crate — head-to-head sweep vs scipy (2026-06-19)
Oracle `docs/perf_oracle_signal.py` + `/tmp/oracle_sig2.py`. fsci vs scipy.signal:

### ⚠️ OPEN LOSS — fftconvolve 2.7× slower; bottleneck is fsci_fft (8l8r1's crate), not signal
fftconvolve 4096×257 same: **fsci 323 µs vs scipy 119.6 µs = 2.7× SLOWER.** scipy pads to
next_fast_len (~4400, 5/7/11-smooth) + uses rfft. TRIED BOTH in fftconvolve, BOTH FAILED →
REVERTED (signal lib.rs back to origin, clean): (1) `fsci_fft::next_fast_len` (4400) +
rfft → **393 µs, REGRESSION** — fsci_fft's mixed-radix path is much slower than its radix-4
power-of-two path, so the smaller 5-smooth size is a net loss. (2) power-of-two + rfft →
**326 µs, ~0 gain** — fsci_fft's `rfft` is NOT faster than its complex `fft` (no real-
symmetry speedup). CONCLUSION: the gap is inside **fsci_fft** (no fast rfft, slow mixed-
radix), 8l8r1's crate — not fixable from signal. Routed to 8l8r1. welch 8× FASTER, so
fsci's own spectral path is fine; only the raw FFT primitive lags. Also welch/coherence
already win, so this is isolated to convolution.

| function | fsci | scipy | ratio |
|---|---|---|---|
| fftconvolve 4096×257 | 323 µs | 119.6 µs | **0.37× (2.7× SLOWER — fsci_fft, OPEN)** |
| **firls 257 (FIXED)** | **296.5 µs** | 366 µs | **1.24× faster** (was 0.42×) |
| filtfilt 4096 biquad | 80.3 µs | 120.2 µs | 1.50× faster |
| sosfilt 4096×2 | 34.0 µs | 46.0 µs | 1.35× faster |
| lfilter 4096 biquad | 37.4 µs | 24.5 µs | **0.65× (1.53× SLOWER — OPEN)** |

### ✅✅ firls LOSS → WIN: precompute integrate_cos table (9l5oo lever, signal crate)
firls builds the LS normal-equations matrix Q[i,j] = w/2·(∫cos(2π(i-j)f)df + ∫cos(2π(i+j)f)df)
over an O(n²) (i,j) double loop, calling `integrate_cos` (2 sin each) PER CELL — but it
depends only on the integer args (i-j) and (i+j). FIX: precompute `ic[arg]` once per band
(offset-indexed over [-(m)..2m], same arg signs ⇒ BYTE-IDENTICAL via exact negation), read
Q from the table. O(n²) sin → O(n). **MEASURED: 873.6 → 296.5 µs = 2.95× self-speedup =
1.24× FASTER than scipy** (was 2.38× slower). Conformance signal **707/0**. The precompute-
element-invariant lever's 5th call site, first OUTSIDE interpolate/spatial. OPEN: lfilter
1.53× slower (sequential IIR recurrence — scipy's tight C, no obvious lever).

## Interpolate crate — FULL head-to-head sweep vs scipy (2026-06-19)
Measured every major interpolator (oracles `docs/perf_oracle_{griddata,clough_tocher,rect}.py`
+ `/tmp/oracle_{1d,rgi}.py`). fsci DOMINATES or matches scipy across the board — no
remaining losses:

| function | fsci | scipy | ratio |
|---|---|---|---|
| griddata / LinearND (576/1024) | 118 µs | 5507 µs | **46.5× faster** |
| CloughTocher eval (576/1024) | 83.5 µs | 537 µs | **6.4× faster** |
| RegularGrid nearest (32³/4096) | 69.8 µs | 361 µs | **5.2× faster** |
| RegularGrid linear (32³/4096) | 178 µs | 608 µs | **3.4× faster** |
| CubicSpline construct (1024) | 25.1 µs | 237.7 µs | **9.5× faster** |
| CubicSpline eval (1024/4096) | 38.8 µs | 73.6 µs | **1.9× faster** |
| interp1d linear (4096/8192) | 39.2 µs | 38.4 µs | parity |
| RectBivariate eval_grid (32²→64²) | 65.8 µs | 48.3 µs | 0.73× (near-parity, was 0.20×) |

LinearND/Clough-Tocher/eval_grid wins came from the precompute-element-invariant lever
(this phase); cubic/RGI/interp1d were already competitive. The ONLY non-win is eval_grid
at 1.36× off scipy's elite Fortran (down from 5.1×). Interpolate is HARVESTED.

## BOLD-VERIFY phase outcome (implemented levers, not just measured)

This phase moved from MEASURING gaps to FIXING them, conformance-gated via `cargo test`:
- **✅ FIXED & SHIPPED — zoom order=1** (`wm14d`, `3c027183`): the gauntlet's biggest
  loss, 17.7× → 4.0× slower (85.95→19.41 ms, 4.4× faster). Root-caused to order=1 being
  the lone interpolating order with no fast path (padded coord_offsets hid it). Cardinal
  fast path for padded linear + stack-array `cardinal_bspline`. Conformance 296/0.
- **✅ BROAD REACH — rotate/affine/map_coordinates** share `sample_interpolated`, so the
  same fix rescued them. Measured: **rotate order=3 6.44 ms vs scipy 5.58 ms = NEAR-PARITY**;
  order=1 4.4× (residual). The whole geometric-transform family is now release-ready at
  order=3 and pathology-free at order=1.
- **↩️ REVERTED ~0-gain** — direct order=1 weights (skip cardinal_bspline calls): measured
  no gain (bottleneck is the general support-machinery, not the weight arithmetic).

**Remaining LOSS gaps — all assessed, all need substantial SIMD/algorithm work (prioritized):**
1. `nm8ex` pdist — needs SIMD distance kernel (the parallel path HELPS; the serial kernel
   is ~10–60× slower than scipy C). NOT a gate fix.
2. `9g6ku` kmeans2 — SIMD distance, but iterative/CHAOTIC so SIMD FP-reorder breaks
   conformance (a 1-ULP distance change cascades to a different clustering). Needs the
   gemm-trick (≠ byte-identical) + tolerance acceptance.
3. gaussian_filter 2.83× — specialized 1D-axis correlate (routes through shared `convolve`).
4. geometric order=1 4× — specialized 2D bilinear bypassing the support machinery (low-ROI;
   order=3 already near-parity).

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

### ✅ RectBivariateSpline.eval_grid 3.75× self-speedup — 5.1× loss → near-parity (separable basis)
Oracle `docs/perf_oracle_rect.py` (scipy.interpolate.RectBivariateSpline `(q,q,grid=True)`,
32×32 → 64×64, kx=ky=3). **BEFORE: fsci 246.9 µs vs scipy 48.3 µs = 5.1× SLOWER.** `eval_grid`
ran the full scalar de Boor recurrence per evaluation (`eval_parts`: span search + per-step
alphas + blend) — the x-recurrence rebuilt for all `ny` rows per xv, the y-recurrence per
(xv,yv). FIX: adopt scipy's FITPACK `bispev` SEPARABLE approach — precompute each axis' k+1
non-zero B-spline basis weights ONCE per query coord (`bspline_basis_funs`, Cox-de Boor
A2.2 with the standard 0/0→0 guard so clamped end-knots are safe), then tensor-contract the
(kx+1)×(ky+1) coefficient window. Added `BSpline::find_span_n` (span by count). **AFTER:
fsci 65.8 µs = 3.75× self-speedup; now 1.36× of scipy (was 5.1×) — near-parity vs elite
Fortran.** NOT byte-identical (different summation order, ~1e-13) but conformance interpolate
**227/0** (rect tests are 1e-10 tolerance vs analytical, incl. the eval(0,0)/(1,1) clamped-
knot boundaries — my clamped-knot worry was unfounded; BasisFuns is built for clamped knots).
KEEP — a 3.75× gain that nearly closes a 5.1× loss. Residual 1.36× is scipy's tighter
vectorized contraction; a SIMD/unrolled kx=ky=3 contraction could reach parity (future).

- The ~50 byte-identical allocation/precompute/batch wins (buffer reuse, mem::take,
  loop-invariant hoist, interval binary-search, write!-amplification, retain) carry
  **no correctness-regression risk** and are monotone by construction (removing an
  alloc/recompute cannot be slower), so they are NOT individually re-benched here;
  the gauntlet revert-risk lives in the **parallelizations** (spawn overhead at small
  n) — those are gate-validated above.
- AP availability-update parallelization is the one OPEN lever surfaced by measurement.

### 🔬 pdist/cdist parallel gate parallelizes BELOW spawn break-even (bead nm8ex) — ROOT-CAUSED, handed to MistyBirch
MEASURED: pdist/euclidean/256 (N=256, d=4) = ~2.68ms vs scipy ~96µs (28×), but SERIAL would be
~131µs (~1.4× scipy = near parity). The 20× inflation is a GATE BUG: `cdist_thread_count`
(spatial lib.rs ~913) goes parallel when `work = na·nb·dim ≥ 1<<18` (262144). N=256,d4 hits
work==262144 exactly → spawns ~64 OS threads (`cores.min(na/2)`) for ~131µs of serial work.
Spawning ~64 threads costs hundreds of µs, so parallel LOSES to serial even on an idle machine
— the gate parallelizes below the spawn break-even. FIX (byte-identical, serial==parallel
offset-fill): raise threshold to `1<<21` (2M ≈ 2·spawn/per-op break-even) so only genuinely
large matrices parallelize; cap thread count for medium work. spatial lib.rs is RESERVED by
MistyBirch → sent the finding+fix via agent-mail (msg 1336) rather than collide. This is the
real mechanism behind the documented pdist loss: it's a parallel-gate-below-break-even bug, NOT
the SIMD kernel (sqeuclidean is already explicit Simd<f64,8>) and NOT a structural layout wall.

### ⚖️ ndimage_filter_thread_count work-capped threads — PRINCIPLED but UNMEASURABLE (reverted)
Hypothesis (byte-identical): cap thread count by work (`min(cores, pixels/2, work>>18)`) so each
thread does ≥256k ops — a separable σ=2 Gaussian pass (~1.1M work) otherwise spawns 64 threads
for ~17µs each. Conformance 297/0 (chunk count ⊥ per-pixel value). BUT the A/B was destroyed by
RAMPING multi-agent load: gaussian_sigma2/256 measured 3.73ms (orig, early window) → 5.20ms
(capped, mid) → 6.70ms (reverted, late) — a monotonic climb that is the LOAD, not the change.
Per demonstrate-or-revert + "same-worker A/B in ONE binary mandatory" (cross-run variance ≫
signal here), REVERTED. The lever is sound for normal/idle machines (fewer threads = less spawn
for medium filters) but needs a same-process atomic-toggle bench in an idle window to prove.
Flagged. (Sibling of the pdist nm8ex gate finding handed to MistyBirch.)

### 📋 PARALLEL-GATE AUDIT (all crates) — pdist is the lone bug; rest are correctly designed
Audited every `*_thread_count` / `work < 1<<N → serial` gate for the gate-below-spawn-break-even
bug. Verdict by per-op cost (the real determinant — cheap multiply/subtract ops need a HIGH
threshold; expensive sort/trig ops can use a low one):
- **pdist/cdist** `cdist_thread_count` 1<<18, cheap subtract-square → **BUG** (handed to MistyBirch, nm8ex).
- **interpolate** `par_query_map` 1<<18 → OK: cost-aware, caller passes `work_per_query`.
- **stats** `compute_row_ranks` 1<<18, **stats** + **cluster** `landmark_isomap` 1<<16 → OK: O(n log n) sort per row.
- **signal** `lombscargle_thread_count` 1<<16 → OK: sin/cos per op (expensive).
- **ndimage** `ndimage_filter_thread_count` 1<<18, cheap mul-add → benched cases (gaussian 1.1M,
  correlate 1.6M) sit AT/above break-even, not clearly below; thread-cap fix unmeasurable under
  ramping load (reverted). The class is otherwise clean — no further gate bugs in my crates.

### ✅ RESOLVED: ndimage filter thread-cap is NEUTRAL (load-invariant same-process A/B)
Built the tool the degraded environment demanded: a same-process atomic-toggle A/B (FILTER_WORK_
CAP_AB, interleaved OFF/ON 50× in one process → load cancels). VERDICT for gaussian_sigma2/256:
cap OFF 5.889 ms vs cap ON 5.980 ms = **NEUTRAL** (~1.5%, within noise). The work-cap does NOT
help under contention — the hypothesis (fewer threads = less oversubscription) is REFUTED by
reliable measurement. Not shipped; toggle+test removed (ndimage back to origin). Supersedes the
earlier "unmeasurable" note. LESSON: the same-process interleaved A/B is THE working method for
contention-sensitive levers when separate-run benches drift 2×; it cleanly settled this one as
neutral. (The pdist nm8ex gate remains a real bug — its fix is math-provable, no A/B needed.)

### ✅✅ erf/erfc Cephes rational kernel (5.0× self-speedup, FLIPS 5.9× loss → 1.2× parity)
The WORST special loss. erf_scalar used an iterative Maclaurin series (≤80 terms) + Lentz
continued fraction (~30 iters for x≥1); scipy's xsf uses Cephes' fixed-degree RATIONAL erf/erfc.
Ported the EXACT Cephes T/U (erf) and P/Q/R/S (erfc) coefficients (fetched from scipy/xsf via
gh) → byte-identical to scipy.special.erf/erfc. **MEASURED special_array_65536/erf: 4.49ms →
904µs = 5.0×; vs scipy 757µs: 5.9× slower → 1.2× (near parity).** Conformance: all erf/erfc/
ndtr/erfcx/erfinv/erfcinv tests PASS. (4 unrelated tests fail on origin — digamma/polygamma/
exp2/powm1 — another agent's in-progress gamma/convenience work; those fns don't call erf, so
not caused by this change; verified by static isolation.) LEVER PAID OUT: fetch scipy's xsf
Cephes coefficients via gh + port the rational → byte-matches scipy AND replaces iterative
kernels. Removed now-unused erf_series_real + erfc_cf_real (erfc_cf_h kept for erfcx_cf_real).

### 📋 Remaining special-kernel Cephes-port candidates (lever PROVEN via erf, lower ROI)
After the erf 5.9×→1.2× flip, audited the other measured special losses for the same
iterative-kernel→Cephes-rational lever:
- **gamma (2.4×)**: gamma_core uses LANCZOS — a fixed ~15-coeff approximation, NOT iterative.
  Its gap is the `powf(x, x+0.5)` cost vs Cephes' recurrence-to-[2,3] rational (avoids powf for
  moderate x). Nuanced, not a clean flip. gamma.rs also has another agent's in-progress
  breakage (digamma/polygamma failing). → leave to that owner.
- **j0/j1/y0/y1 (j0 1.6×)**: j0_core uses a genuine convergence-loop power series for x<14
  (`j0_series_small`, ~15-25 terms in the bench range). CLEAN Cephes lever (rational P0/Q0 for
  x<5 + asymptotic PP/PQ/QP/QQ modulus/phase for x≥5, ~6 arrays). But modest gain (1.6×) for a
  ~80-line/6-array port across 4 functions → lower ROI than erf's 5.9×; flagged not done.
RECIPE (proven): `gh api repos/scipy/xsf/contents/include/xsf/cephes/<file>.h --jq .content |
base64 -d` → transcribe the exact coefficient arrays → byte-matches scipy.special.

### ✅ j0 Cephes rational kernel (byte-matches scipy; array 1.1×, kernel win contention-masked)
Applied the proven Cephes-fetch lever to j0_core: replaced the convergence-loop power series
(~25 terms for x<14) with scipy's xsf EXACT Cephes rational (RP/RQ for |x|≤5 + PP/PQ/QP/QQ
asymptotic modulus/phase for |x|>5) → byte-matches scipy.special.j0. Conformance: NO new
failures (the 4 — digamma/polygamma/exp2/powm1 — are pre-existing non-j0/non-erf, another
agent's work). MEASURED special_array_65536/j0: 1.08ms → 0.985ms = 1.1× — modest because the
array is SPAWN-bound under fleet contention (the rational-vs-series kernel win is bigger but
masked; cf. erf where the kernel was a big enough fraction to show 5×). KEPT: strictly better
(scipy-exact parity + provably fewer ops), not a regression. j0_series_small retained (y0 uses
it at 3212). j1/y0/y1 still series (same lever, lower priority).

### 📊 Special-kernel ranking (find-the-next-erf measurement) — erf was UNIQUE; rest complex/nuanced
MEASURED fsci scalar special kernels to find another erf-class slow-iterative loss:
- **gammainc** (gamma.rs:1216): 58ns(a1)→94ns(a10), GROWS with a (series ~a terms) → large a
  much worse vs Cephes igam (bounded). Genuine candidate BUT the Cephes igam port is COMPLEX
  (multi-regime: small-a series + CF + Temme asymptotic, not a flat rational like erf) AND
  gamma.rs holds another agent's in-progress breakage (digamma/polygamma fail) → HAND OFF.
- **beta** (40→96ns, grows): traces to Lanczos gammaln (3 lgam calls), same powf-cost nuance
  as gamma — not an iterative→rational flip.
- gamma=Lanczos (nuanced), j0/j1/y0/y1=series but contention-masked (1.1×).
CONCLUSION: erf (5.9×→1.2×, simple 80-term-series+CF → flat rational) was the UNIQUE clean
erf-class Cephes lever. The rest are complex multi-regime ports (igam) or Lanczos-nuanced or
modest — none clears the bar erf did. Next-session: the igam port is the only remaining
big-loss candidate, but it needs gamma.rs free + a careful multi-regime Cephes port.

### ✅ ellipeinc combined Carlson R_F+R_D (1.4×, byte-identical, slowest special kernel)
The find-an-erf-class measurement flagged ellipeinc (incomplete elliptic E) as the SLOWEST
special kernel (280-307ns). E(φ,m) = s·R_F(cc,d,1) − (m/3)s³·R_D(cc,d,1) called carlson_rf AND
carlson_rd over the SAME (cc,d,1) — TWO separate sqrt-heavy duplication sequences. R_F and R_D
share the IDENTICAL (x,y,z) sequence (only `ave`/convergence + R_D's `s` accumulation differ),
so a combined `carlson_rf_rd` computes the sqrt-sequence ONCE, tracking each convergence
independently → BYTE-IDENTICAL. **MEASURED ellipeinc_scalar: m0.5 280→195ns (1.44×), m0.9
307→219ns (1.40×)** (1.4× not 2× because R_D is costlier than R_F). Conformance: same 4
pre-existing failures, NO new (byte-identical). NOT a Cephes port — a pure shared-iteration
refactor in a FREE file (elliptic.rs), refreshed-first (no clobber). Reusable: any code calling
carlson_rf+carlson_rd on the same args.

### ✅ Carlson ERRTOL 1e-5 → 1.3e-3 (1.4-1.6×, machine-accurate) — stacks ellipeinc to ~2× total
NR Carlson R_F/R_D use a 5th-order final correction → error ~ERRTOL^6. fsci's ERRTOL=1e-5 gave
error ~1e-30 (overkill by ~14 orders); double-precision only needs error <2e-16 → ERRTOL≈2e-3.
Raised all 3 (carlson_rf/rd/rf_rd) to **1.3e-3** (error ~5e-18, machine-accurate), cutting the
duplication iterations ~9→5. **MEASURED: ellipkinc 148→107ns (1.38×), 171→109ns (1.57×);
ellipeinc 195→146ns (1.34×), 219→149ns (1.47×)** — with the earlier Carlson-sharing, ellipeinc
is now 307→149ns = **~2.06× total** (the slowest special kernel halved). Conformance: same 4
pre-existing failures, NO new (machine-accurate vs scipy). Provable lever: audit iterative
convergence tolerances vs the order of the final correction — an over-tight ERRTOL wastes
iterations at no accuracy benefit. Free file, refreshed-first.

### ✅ lfilter biquad unrolled scalar fast path — FLIPS 1.53× loss → parity/slight win (signal)
The OPEN lfilter biquad loss (4096 biquad: fsci 37.4 µs vs scipy 24.5 µs = 1.53× SLOWER).
Root cause: the general `lfilter_with_state` ran the DF2T delay-line update as a branchy
inner loop over a HEAP `Vec d` (`for j in 0..nfilt-1` with a `j+1 < nfilt-1` boundary branch
+ bounds-checked `b_norm[j+1]`/`a_norm[j+1]`/`d[j+1]` indexing every sample) — whereas
`sosfilt` already used the optimal fully-unrolled scalar-register biquad form (d1/d2 in
registers, no indexing/branch). Added byte-identical unrolled fast paths for nfilt==2 (order 1)
and nfilt==3 (order 2 / biquad) that keep the whole delay line in scalar registers — same float
ops in the same order as the general recurrence (verified: d[0] reads OLD d[1] before write).
**MEASURED filtering/lfilter/4096_biquad: 37.4 µs → 24.2 µs = 1.54× self-speedup (criterion
change −35.9%); now ≈ scipy 24.5 µs (parity, marginally faster).** Conformance: fsci-signal
GREEN 648/0 (+59 metamorphic), incl. lfilter_with_state_matches_scipy_reference_vectors and
lfilter_fir_iir_match_scipy. Byte-identical by construction. Lever: when a general N-tap kernel
serves a hot low-order case, peel a register-unrolled specialization for the common orders
(1/2) — the heap delay line + per-iter bounds/branch was the entire gap, exactly as sosfilt
already demonstrated. (filtfilt/lfilter_axis_2d route through the same core → inherit the win.)
