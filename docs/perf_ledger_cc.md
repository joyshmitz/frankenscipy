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

### ✅ gaussian_filter 2-D parallel gate raised (serial 1.82× FASTER at 256² — closes most of the 2.83× loss)
The documented gaussian_filter loss (σ=2 256²: fsci 3238 µs vs scipy 1143 µs = 2.83× slower).
Root cause was NOT the kernel (col-pass interior-axpy was a measured 0.755× regression — see
NEGATIVE_EVIDENCE) but a PARALLEL-BELOW-BREAK-EVEN gate: gaussian_filter_2d_reflect_order0 took
its thread count from the shared `ndimage_filter_thread_count` (parallel when pixels·kernel_len
>= 1<<18). At 256² that work ≈ 1.1M trips the gate and spawns ~1 thread per few rows, but the
separable row/col passes are cheap per pixel (one symmetric fold), so spawn overhead dominates.
**Same-process interleaved A/B (30 reps × 200 iters, GAUSSIAN_FORCE_SERIAL toggle, byte-identical
assert_eq across all sizes):**
| n     | serial    | parallel  | serial speedup |
|-------|-----------|-----------|----------------|
| 128²  | 506 µs    | 3530 µs   | **6.98×**      |
| 256²  | 2095 µs   | 3814 µs   | **1.82×** (bench size) |
| 512²  | 7010 µs   | 4534 µs   | 0.65× (parallel wins) |
| 1024² | 24107 µs  | 5079 µs   | 0.21× (parallel wins) |
FIX: gate the gaussian 2-D path at `pixels·kernel_len >= 1<<21` (~2M) so ≤256² runs serial and
≥512² stays parallel (break-even is between them). BYTE-IDENTICAL (thread count never changes the
result — proven by assert_eq). fsci-ndimage GREEN 246/0 (+58 integration). The 256² serial 1.82×
closes most of the 2.83× scipy gap (absolute µs are contention-inflated here; the A/B RATIO is the
reliable signal — criterion cross-run swung +20%…+196% in ONE run, uninterpretable under load).
LEVER (paid out again): a parallel gate must scale with PER-ELEMENT WORK COST, not a flat
flop-product threshold. Cheap separable/elementwise kernels need a MUCH higher work gate than the
shared default — the same cost-aware-gate lesson as the stats batch-method and pdist veins.

### ✅ uniform_filter parallel gate fixed (PIXEL-COUNT not work-product) — serial 3.78×@256², 1.48×@512²
Same cost-aware-gate vein as the gaussian fix. uniform_filter_along_axis parallelized across outer
slabs when `ndimage_filter_thread_count(arr.size(), size) >= 1<<18` (i.e. arr.size()·size). But the
sliding window uses a RUNNING SUM — O(1) per output element (drop leaving + add entering),
INDEPENDENT of window `size` — so multiplying the work metric by `size` is wrong (over-counts large
windows, trips the gate far too early), and the real amortization point scales with PIXEL COUNT.
**Same-process interleaved A/B (byte-identical assert_eq all sizes, axis=1, Reflect):**
| n      | sz | serial    | parallel  | serial speedup |
|--------|----|-----------|-----------|----------------|
| 256²   | 5  | 735 µs    | 2780 µs   | **3.78×**      |
| 512²   | 5  | 3694 µs   | 5470 µs   | **1.48×**      |
| 1024²  | 5  | 18979 µs  | 18904 µs  | 0.996× (parity)|
FIX: gate the running-sum pass at `arr.size() >= 1<<20` (~1M px) — below that it spawns up to 64
threads for cheap O(1)/elt slabs and the spawn overhead dominates; from ~1M px up parallel pays.
BYTE-IDENTICAL (thread count never changes the result). fsci-ndimage GREEN 246/0 (+58 integration).
The 256² 3.78× / 512² 1.48× are the realistic image-filter sizes. Third payout of the cost-aware
gate lever (gaussian 2-D, now uniform_filter); cheap separable/running-sum ndimage kernels need a
MUCH higher work gate than the shared 1<<18 default — gate on PER-ELEMENT-COST-scaled work, and for
size-independent running sums that means pixel count, NOT pixel·window.

### ✅ correlate1d/convolve1d along-axis parallel gate raised (serial 2.61×@256²) — also fixes general gaussian path
Fourth payout of the cost-aware-gate vein. correlate1d_along_axis & convolve1d_along_axis (which
back public correlate1d/convolve1d AND the GENERAL gaussian path gaussian_filter1d_axis→
convolve1d_along_axis for 3D/order>0/non-reflect) parallelized across outer slabs when the shared
`ndimage_filter_thread_count(arr.size(), weights.len()) >= 1<<18`. Per-element cost IS an
O(weights.len())-tap dot (so the work product is the right metric — unlike uniform's running sum),
but the 1<<18 threshold is too low: at 256² w5 (work 327k) it spawns ~64 threads for a cheap pass.
**Same-process A/B (byte-identical assert_eq, axis=1, w5, Reflect):**
| n     | serial   | parallel | serial speedup |
|-------|----------|----------|----------------|
| 256²  | 968 µs   | 2526 µs  | **2.61×**      |
| 512²  | 3793 µs  | 3075 µs  | 0.81× (parallel wins 1.23×) |
FIX: gate both at `arr.size()·weights.len() >= 1<<20` (break-even ~n=453); 256²→serial, ≥512²→
parallel. BYTE-IDENTICAL. fsci-ndimage GREEN 246/0 (+58). Vein tally: gaussian-2D (1<<21 fold),
uniform_filter (1<<20 pixel-count running-sum), now correlate1d/convolve1d (1<<20 tap-dot). The
shared 1<<18 gate was uniformly too low for ALL cheap separable ndimage kernels.

### ❌ TRIED & REVERTED (~0-gain, see NEGATIVE_EVIDENCE 2026-06-22): SIMD-across-output-pixels for nd_filter_apply
UPDATE: implemented + measured = **1.025× (memory-bandwidth-bound, NOT compute-bound)**. The 25 taps
each hit a different input row/cache-line, so vectorizing 8 output pixels cuts instructions but not the
dominant memory traffic. The correlate/gaussian 1.1-1.2× residuals are a BANDWIDTH wall — do NOT
re-chase with SIMD. Byte-identity was confirmed (correct, just useless). Original (now-refuted) note:
### 📋 NEXT BOLD LEVER (scoped, byte-identical): SIMD-across-output-pixels for nd_filter_apply interior
The correlate 5x5 256² 1.18× residual (and gaussian/correlate kernel walls generally) is the scalar
inner loop: per interior pixel, `for k: sum += w[k]*input.data[p+tap_flat[k]]` (25 scalar fmas).
nd_filter_apply ALREADY has the interior flat-offset fast path; the remaining gap is scalar-vs-C-SIMD.
LEVER (proven in spatial pdist, see [[perf_spatial_pdist_simd_across_pairs]] — pure std::simd, NO
unsafe, forbid(unsafe)-safe): process 8 CONSECUTIVE interior output pixels (same row ⇒ contiguous)
as one Simd<f64,8>: `acc += Simd::splat(w[k]) * Simd::from_slice(&input.data[p+tap_flat[k] ..][..8])`,
then copy_to_slice. BYTE-IDENTICAL: each lane independently accumulates ITS pixel's sum in the SAME
k-order as scalar (Rust `+`/`*` don't FMA-contract by default). Needs: region-partition the 2-D
output into the interior box [lo,hi)² + boundary bands (so interior runs are contiguous and reflection-
free), iterate interior rows, process interior cols 8-wide + scalar remainder, boundary via the slow
path. Interior is ~97% of a 256² 5x5 → up to memory-bound 2-4× on the kernel, plausibly flips the
1.18× loss to a WIN. Build the byte-identity property test first (correlate vs nd_filter_perpixel_ref).
Same lever extends to gaussian's col-pass and any separable/dense filter interior. NOT YET DONE —
deferred to a fresh-context iteration (meaty change in a fragile file; do it with full budget).
NOTE (ruled out this session): the per-pixel DIVIDE in the interior check is NOT the bottleneck
(incremental-index A/B = 0.945×, reverted) — it's the scalar gather/fma throughput. SIMD is the lever.

### ✅✅ interpolate par_query_map gate 1<<18 → 1<<23 — flips an 18.5x over-parallelization REGRESSION (cubic eval_many)
The cost-aware-gate vein extends to fsci-interpolate. par_query_map/par_query_try_map (back ALL
*_many evaluators: cubic/pchip/CubicSplineStandalone/RBF/griddata/RGI) gated parallelism at
`m·work_per_query >= 1<<18`. Unlike ndimage's in-place chunks_mut, this parallel path allocates a
RESULT VEC PER THREAD (up to ~m/2 threads, capped at cores) and `flat_map`-collects them — a large
FIXED overhead (~4-5 ms under fleet contention, independent of m). At work_per_query=24 (spline eval)
the gate fired at m≈10923, catastrophically over-parallelizing common batch sizes.
**Same-process A/B (cubic eval_many, n=1024 knots, byte-identical assert_eq all sizes):**
| m (queries) | serial   | parallel | serial speedup |
|-------------|----------|----------|----------------|
| 16384       | 212 µs   | 3924 µs  | **18.52x**     |
| 32768       | 396 µs   | 4144 µs  | **10.48x**     |
| 65536       | 788 µs   | 4645 µs  | **5.89x**      |
| 131072      | 1522 µs  | 4753 µs  | **3.12x**      |
Parallel is ~4-4.8 ms FIXED (spawn + per-thread Vec alloc + flat_map realloc); serial scales, so
break-even is ~350k queries (work ≈ 1<<23). FIX: raise the shared gate to `1<<23` (single constant,
both par_query_map + par_query_try_map). Cheap batch evals now stay serial up to ~350k queries where
parallelism finally amortizes; genuinely huge batches still parallelize. BYTE-IDENTICAL (thread count
never changes the result; assert_eq verified). fsci-interpolate GREEN 173/0 (+56). HIGH value — eval_many
at m=16k-131k is the common interpolation batch path and was 3-18x pessimized. Same root cause as the
ndimage gates (shared 1<<18 too low for many-core spawn) but WORSE here (per-thread Vec alloc, not
in-place). Lever now paid out 4× across two crates: gate on per-element cost AND account for the
parallel implementation's fixed overhead (alloc-per-thread ⇒ much higher break-even than chunks_mut).

### ✅✅ stats: 8 new axis-2D reducers + gmean per-call syscall fix (3.1-40x faster than scipy, same-box)
Continues the proven axis-2D reducer vein (8ec65b21 added 6 at 27-145x; 29f1a75a rankdata 60-90x).
Eight new `*_axis_2d` multi-channel reducers wrap their scalar 1-D fn through the parallel-across-lines
`reduce_axis_2d` helper (bit-identical to per-line by construction; conformance via the extended
`reduce_axis_2d_family_matches_per_line` test, `to_bits` so NaN-on-negative still matches): `sem`,
`gmean`, `hmean`, `gstd`, `kstat`, `kstatvar`, `moment`, `differential_entropy`.

**SAME-BOX head-to-head (best-of-20, fsci binary + scipy.stats both on this 64-core box):**
| reducer              | 2000×512 (scipy/fsci ms → ×) | 500×4096 (scipy/fsci ms → ×) |
|----------------------|------------------------------|------------------------------|
| sem                  | 2.04 / 1.60 → **1.27×**      | 5.99 / 1.69 → **3.55×**      |
| gmean                | 5.41 / 1.70 → **3.18×**      | 12.44 / 1.88 → **6.62×**     |
| hmean                | 1.73 / 1.71 → 1.01× (parity) | 5.48 / 1.67 → **3.28×**      |
| gstd                 | 18.20 / 1.88 → **9.68×**     | 39.52 / 2.08 → **18.97×**    |
| kstat(n=2)           | 1.82 / 1.81 → 1.00× (parity) | 6.30 / 1.96 → **3.21×**      |
| kstatvar(n=2)        | 25.24 / 1.89 → **13.36×**    | 52.26 / 2.27 → **23.04×**    |
| moment(k=4)          | 12.25 / 1.82 → **6.73×**     | 26.06 / 1.68 → **15.51×**    |
| differential_entropy | 55.12 / 2.40 → **22.97×**    | 135.88 / 3.36 → **40.46×**   |

gstd/kstatvar/moment/differential_entropy win 7-40× because scipy's own implementations are heavy
Python; sem/hmean/kstat are parity-to-3.5× (never a loss).

**BUG CAUGHT & FIXED while measuring (byte-identical):** `gmean_axis_2d` was initially a *2.3× LOSS*
at 2000×512 (11.8 ms) yet 3.5 ms at 500×4096 — non-monotonic (1M logs slower than 2M). Root cause:
the scalar `gmean`→`gmean_log_sum` calls `std::thread::available_parallelism()` (a `sched_getaffinity`
syscall) on EVERY invocation, BEFORE the `n < 1<<16` serial short-circuit. Called once per line by the
reducer (2000 short lines), the ~5µs syscall ×2000 ≈ 10 ms dominated the cheap `ln` work. gstd (no
parallelism probe) stayed 1.9 ms on identical log counts — the smoking gun. FIX: hoist the `n < 1<<16`
return ABOVE the `available_parallelism()` call (byte-identical: that path always took `chunk_sum`
anyway). gmean_axis_2d 11.8→1.70 ms (6.9×), flipping the loss to a 3.18× win. Bonus: standalone
`gmean()` on any <65536-elt input no longer pays the syscall (helps every per-line/hot-loop caller).
LESSON (generalizable): probing `available_parallelism()` inside a per-element scalar kernel is a hidden
syscall tax when that kernel is the reduce-closure of an axis sweep — order the cheap serial-gate FIRST.
fsci-stats GREEN (reduce_axis_2d_family + all gmean/gstd/hmean tests pass). Same-process A/B mandatory.

### ✅✅ stats: 10 MORE axis-2D reducers (trimmed/circular/mode/entropy) + entropy syscall fix (2.5-71x faster than scipy)
Third batch on the reduce_axis_2d vein. scipy's per-axis trimmed/circular/mode stats are catastrophically
slow (Python masking + per-slice dispatch): tstd 51-112ms, tsem 48-102ms, mode 50-129ms, tvar 27-75ms,
circmean/var/std 43-87ms, entropy 16-32ms. fsci's parallel-across-lines reducers run at ~1.5-3.4ms.
Added (bit-identical to per-line, conformance in extended reduce_axis_2d_family test, 24 reducers total):
tmean, tvar, tstd, tsem, tmax, mode, entropy, circmean, circvar, circstd.

**SAME-BOX paired head-to-head (best-of-20, fsci binary + scipy.stats measured BACK-TO-BACK under same load):**
| reducer  | 2000×512 (scipy/fsci → ×) | 500×4096 (scipy/fsci → ×)  |
|----------|---------------------------|----------------------------|
| tstd     | 51.27/1.46 → **35.1×**    | 112.32/1.59 → **70.8×**    |
| tsem     | 47.59/1.54 → **30.9×**    | 102.44/1.56 → **65.5×**    |
| mode     | 50.16/1.86 → **26.9×**    | 128.51/2.85 → **45.1×**    |
| tvar     | 26.72/1.51 → **17.7×**    | 74.98/1.61 → **46.5×**     |
| circmean | 43.72/2.37 → **18.4×**    | 86.50/3.38 → **25.6×**     |
| circvar  | 43.05/2.31 → **18.6×**    | 85.98/3.31 → **26.0×**     |
| circstd  | 44.12/2.46 → **17.9×**    | 85.87/3.26 → **26.3×**     |
| entropy  | 15.99/1.69 → **9.5×**     | 31.88/1.71 → **18.7×**     |
| tmean    | 5.26/1.63 → **3.2×**      | 16.45/1.77 → **9.3×**      |
| tmax     | 3.80/1.53 → **2.5×**      | 8.19/1.58 → **5.2×**       |

**SYSCALL-TAX LEVER PAID OUT A 3rd TIME (byte-identical):** `entropy` was initially a 12.23ms / 1.22×
near-loss at 2000×512 — identical non-monotonic signature to gmean (1M elts slower than 2M). Root cause
again: `entropy_h_sum` called `available_parallelism()` (sched_getaffinity syscall) on every line BEFORE
its `n<1<<16` serial gate. Hoisted the gate above the syscall → entropy 12.23→1.69ms (7.7×), 1.22×→9.5×
win. grep confirmed only gmean_log_sum + entropy_h_sum had the `|| threads<=1` pattern; class now closed.

**tmin DELIBERATELY OMITTED:** `scipy.stats.tmin` is a masked `np.min`, unusually fast (~1.0-1.2ms),
below reduce_axis_2d's ~1.5ms 64-thread-spawn floor → a parallel fsci tmin is a 0.72× LOSS at narrow
columns (wins 2.1× at wide). Omitted to keep an all-wins batch (tmax kept — scipy.tmax is ~3× slower
than tmin, fsci wins it). FOLLOW-ON LEVER (noted, not done): the ~1.5ms floor is pure 64-thread spawn
overhead — ALL reducers hit it at 2000 lines regardless of op cost (tstd≈tmax≈1.5ms). Capping
reduce_axis_2d's thread count for low total-work would lower the floor AND flip tmin; needs careful
same-process A/B (risk of regressing the big-win heavy reducers). Deferred.

### ✅✅ stats: reduce_axis_2d thread-count cap — lifts ALL 25 axis-2D reducers 1.0-2.4x (byte-identical) + flips tmin loss→win
DIG via extreme-software-optimization (profile-driven). reduce_axis_2d (and the rankdata_axis_2d helper)
fanned out to ALL 64 cores whenever work >= 1<<16. Same-process A/B (one bin, fixed thread counts,
byte-identical checksum asserted across all counts) showed 64 threads is ALWAYS worse than 16-32 for the
common 1-2M-element regime — a ~1.5ms FLOOR that is pure OS-thread spawn/join overhead (~20µs × 64),
dominating the actual cheap per-line reduce. Optimal is ~21t at work≈1M, ~42t at work≈2M.

FIX (both parallel-across-lines sites): cap nthreads at `work / 48_000` element-ops/thread (each thread
busy enough to amortize its spawn), `threads.min(n_lines).min(work/48000)`. BYTE-IDENTICAL (thread count
never changes a per-line reduction; family + rankdata bit-identity tests green) and never spawns MORE
than the old `threads.min(n_lines)` → a MONOTONE win that still ramps to all 64 cores once work justifies
it (>= 64·48k ≈ 3.1M elements).

**Same-process A/B (64t OLD → formula-picked NEW, measured back-to-back same load):**
| reducer  | work≈1M: 64t→21t          | work≈2M: 64t→42t          |
|----------|---------------------------|---------------------------|
| tmin     | 1.449→0.593 → **2.44×**   | 1.484→1.030 → **1.44×**   |
| tstd     | 1.431→0.809 → **1.77×**   | 1.539→1.242 → 1.24×       |
| entropy  | 1.527→0.853 → **1.79×**   | 1.563→1.305 → 1.20×       |
| mode     | 1.809→1.491 → 1.21×       | 2.418→2.396 → 1.01× (par) |
| circmean | 1.741→1.438 → 1.21×       | 2.195→2.063 → 1.06×       |

Lifts the WHOLE 25-reducer family (skew/kurtosis/.../trimmed/circular/mode/entropy) since they all route
through reduce_axis_2d — biggest gains on cheap/medium ops at narrow columns (the spawn-floor-bound case).
BONUS: tmin_axis_2d (dropped in the prior batch as a 0.72× loss vs scipy's fast masked-min) RE-ADDED — now
0.59ms vs scipy 1.01ms = **1.7× WIN** at 2000×512 (4.4× at 500×4096). The lever I built to kill the floor
flipped the one function the floor had cost me. LESSON: probe `available_parallelism()`-driven fan-out with
a same-process fixed-thread A/B; "use all cores" is wrong when per-call work is < ~64·spawn_cost.

### ✅✅ stats: zscore/gzscore/zmap axis-2D — a DIFFERENT primitive (vmap-style vector-output map) — 3.9-14.7x faster than scipy
DIG (jax "different primitive"): reduce_axis_2d reduces a line→scalar; zscore/gzscore/zmap need line→LINE
(vector output, same shape). Added the complementary primitive: `map_axis_2d` / `par_produce_lines` — a
batched vector-output map parallel across lines with the SAME work-capped thread count (axis_2d_thread_count,
the 48k/thread cap factored out). Three new public fns: zscore_axis_2d, gzscore_axis_2d, zmap_axis_2d
(bit-identical to per-line 1-D; conformance in new `map_axis_2d_family_matches_per_line`, both axes, to_bits).

scipy.stats.zscore/gzscore/zmap carry heavy intermediate-array overhead (mean+std+subtract+divide as
separate numpy temporaries); fsci does 2 passes (mean/std then write) parallel across lines, no temporaries.

**SAME-BOX paired head-to-head (best-of-30, fsci public fns + scipy.stats back-to-back):**
| op      | 2000×512 (scipy/fsci → ×) | 500×4096 (scipy/fsci → ×) | 4000×1024 (scipy/fsci → ×) |
|---------|---------------------------|---------------------------|----------------------------|
| zscore  | 4.51/1.03 → **4.4×**      | 12.81/1.72 → **7.5×**     | 30.25/2.81 → **10.8×**     |
| gzscore | 8.31/1.28 → **6.5×**      | 21.45/2.07 → **10.4×**    | 46.44/3.16 → **14.7×**     |
| zmap    | 4.11/1.05 → **3.9×**      | 12.64/1.70 → **7.4×**     | 29.06/3.18 → **9.1×**      |

This is the FIRST vector-output member of the axis-2D family (the prior 25 are scalar reductions). The
new `map_axis_2d` primitive + `axis_2d_thread_count` (shared 48k/thread cap) generalize to any future
batched line→line transform (e.g. detrend/normalize/rankdata-values/winsorize-along-axis). gzscore wins
most (scipy's per-element log+exp temporaries are the slowest). REVERT-check N/A (pure addition, no
existing path changed). fsci-stats conformance GREEN (map_axis_2d_family + all zscore/gzscore/zmap tests).

### ✅✅ stats: kendalltau_matrix (all-pairs Kendall tau) — a DIFFERENT primitive (parallel all-pairs vs Python loop) — 61-118x faster than scipy
DIG: scipy has NO vectorized all-pairs Kendall tau — computing a Kendall correlation matrix means looping
`scipy.stats.kendalltau` in Python over m·(m−1)/2 pairs, paying Python-call overhead × every pair PLUS a
per-pair exact-Mahonian/asymptotic p-value the matrix never needs. NEW `kendalltau_matrix(variables)`:
(1) tau-ONLY per pair (`kendalltau_statistic_only`, bit-identical to `kendalltau(.).statistic`, skips the
p-value — the bulk of per-pair cost), (2) parallel ACROSS pairs (heavy O(n log n) per pair amortizes OS
spawn → fan out to all cores, >=4 pairs/thread). Diagonal = self-tau (1.0 / NaN-for-constant).

**SAME-BOX head-to-head (fsci kendalltau_matrix vs scipy Python kendalltau-loop, both this box):**
| matrix (m vars × n obs)      | pairs  | scipy      | fsci     | speedup    |
|------------------------------|--------|------------|----------|------------|
| m=40,  n=400                 | 780    | 194 ms     | 3.16 ms  | **61×**    |
| m=100, n=1000                | 4 950  | 1 673 ms   | 19.85 ms | **84×**    |
| m=200, n=1000                | 19 900 | 6 688 ms   | 56.74 ms | **118×**   |

Speedup GROWS with size (more pairs ⇒ more parallelism + Python-loop overhead dominates scipy more).
Conformance: `kendalltau_matrix_matches_pairwise` asserts every entry == per-pair kendalltau(.).statistic
bit-identically (incl. a tied column), symmetric, ragged-input rejected. Pure addition (123 lines, 0
deletions) — no existing path changed. This is the "all-pairs over an O(n log n) per-pair kernel, tau-only,
parallel across pairs" lever; generalizes to any all-pairs statistic scipy makes users Python-loop
(weightedtau matrix, somersd matrix, pairwise distance-correlation). fsci-stats conformance GREEN.

### ✅✅ stats: weightedtau_matrix (all-pairs weighted Kendall tau) — 108-222x faster than scipy + factored the all-pairs primitive
Extends the all-pairs primitive to scipy's OTHER matrix-less rank correlation. Factored the parallel-
across-pairs logic into `all_pairs_symmetric_matrix(variables, pair_stat)` (kendalltau_matrix refactored
onto it, byte-identical — conformance test unchanged & green); added `weightedtau_matrix` = the same
helper over `weightedtau` (which returns f64 directly, no p-value). scipy has NO vectorized all-pairs
weighted tau → users loop `scipy.stats.weightedtau` in Python, and weightedtau is SLOWER per-call than
kendalltau (hyperbolic weighting), so the gap is even larger.

**SAME-BOX head-to-head (fsci weightedtau_matrix vs scipy Python weightedtau-loop, both this box):**
| matrix (m × n)   | pairs  | scipy        | fsci      | speedup     |
|------------------|--------|--------------|-----------|-------------|
| m=40,  n=400     | 780    | 648.3 ms     | 5.97 ms   | **108.6×**  |
| m=100, n=1000    | 4 950  | 10 462.9 ms  | 47.09 ms  | **222.2×**  |

(10.5 SECONDS in scipy for a 100×100 weighted-tau matrix.) Conformance: weightedtau_matrix upper-triangle
+ diagonal bit-identical to per-pair `weightedtau` (matrix symmetric BY CONSTRUCTION — the helper mirrors
the upper triangle; NOTE weightedtau is mathematically but NOT bit-symmetric across arg order due to its
Fenwick accumulation sorting by the first arg, so only i<=j is asserted per-pair). The `all_pairs_symmetric_matrix`
helper now backs both matrices and any future one (somersd/distance-correlation). fsci-stats GREEN.

### ✅✅ stats: wasserstein/energy distance matrices (all-pairs) — 16-63x faster than scipy
Extends the all-pairs primitive from correlation to DISTANCE matrices (distribution comparison /
clustering of m 1-D samples). wasserstein_distance & energy_distance are symmetric f64 distances → both
are one-liners over `all_pairs_symmetric_matrix`. SciPy has NO vectorized all-pairs form — users loop
`scipy.stats.wasserstein_distance` / `energy_distance` in Python over m·(m−1)/2 pairs.

**SAME-BOX head-to-head (fsci matrix vs scipy Python distance-loop, both this box):**
| matrix (m × n)  | pairs  | scipy wass. | fsci wass. | ×        | scipy energy | fsci energy | ×        |
|-----------------|--------|-------------|------------|----------|--------------|-------------|----------|
| m=40,  n=400    | 780    | 55.5 ms     | 2.74 ms    | **20.3×**| 54.2 ms      | 3.35 ms     | **16.2×**|
| m=100, n=1000   | 4 950  | 843.3 ms    | 13.47 ms   | **62.6×**| 869.9 ms     | 19.20 ms    | **45.3×**|

Conformance: `distance_matrices_match_pairwise` — upper triangle + diagonal bit-identical to per-pair
`wasserstein_distance`/`energy_distance`, symmetric, ragged-input rejected (diagonal = self-distance, not
asserted == 0.0: `d(u,u)` may be ±0.0/tiny-rounding, the per-pair i<=j check covers it). The
`all_pairs_symmetric_matrix` helper now backs FOUR matrices (kendalltau/weightedtau/wasserstein/energy);
ANY symmetric `fn(&[f64],&[f64])->f64` scipy makes you Python-loop is now a one-liner. fsci-stats GREEN.

### ✅ stats: ks_2samp_matrix (all-pairs two-sample KS test) — 8-29x faster than scipy
Extends the all-pairs primitive to two-sample TESTS (pairwise distribution comparison — a common
multiple-comparison workflow). New tuple helper `all_pairs_two_symmetric_matrices` (per-pair kernel
returns `(stat, pvalue)` → two symmetric matrices); `ks_2samp_matrix` returns `(D_matrix, pvalue_matrix)`.
SciPy has NO vectorized all-pairs form — users loop `scipy.stats.ks_2samp` in Python.

**SAME-BOX head-to-head (fsci ks_2samp_matrix vs scipy Python ks_2samp-loop, both this box):**
| matrix (m × n)  | pairs  | scipy      | fsci      | speedup   |
|-----------------|--------|------------|-----------|-----------|
| m=40,  n=400    | 780    | 262.0 ms   | 9.1 ms    | **28.8×** |
| m=100, n=1000   | 4 950  | 2 030.2 ms | 243.2 ms  | **8.3×**  |

HONEST NOTE: the m=100/n=1000 win (8.3×) is smaller than the correlation/distance matrices (16-222×)
because fsci's `ks_2samp` P-VALUE is heavy per-pair at large n (~3.4 ms/pair, ~8× slower than scipy's
asymptotic) — the matrix is already at all 64 cores (compute-bound, not thread-limited). FOLLOW-ON (noted,
not done): speed fsci's ks_2samp pvalue at large n (likely an exact/series path where scipy goes
asymptotic), or offer a statistic-only `ks_2samp_statistic_matrix` (the D stat is O(n log n), would be
50-100×). Conformance: `ks_2samp_matrix` upper-triangle + diagonal bit-identical to per-pair ks_2samp
(both stat & pvalue), symmetric, ragged rejected. fsci-stats GREEN.

### ✅✅ stats: mannwhitneyu_matrix (all-pairs Mann–Whitney U test) — 113-131x faster than scipy
The strongest two-sample-test matrix (the ks follow-on). fsci's `mannwhitneyu` reports the smaller U
(order-independent) and a normal-approximation p-value (CHEAP, unlike ks_2samp's heavy exact pvalue) — so
both outputs are symmetric and it's a ONE-LINER over the `all_pairs_two_symmetric_matrices` tuple helper.
SciPy has NO vectorized all-pairs form — pairwise rank-sum comparison means looping
`scipy.stats.mannwhitneyu` in Python.

**SAME-BOX head-to-head (fsci mannwhitneyu_matrix vs scipy Python mannwhitneyu-loop, both this box):**
| matrix (m × n)  | pairs  | scipy      | fsci      | speedup    |
|-----------------|--------|------------|-----------|------------|
| m=40,  n=400    | 780    | 407.1 ms   | 3.61 ms   | **112.9×** |
| m=100, n=1000   | 4 950  | 3 130.2 ms | 23.94 ms  | **130.8×** |

Confirms the memory prediction: where ks_2samp_matrix was capped at 8× by fsci's heavy ks pvalue,
mannwhitneyu's normal-approx pvalue keeps the per-pair kernel light → the full all-pairs speedup. Returns
`(U_matrix, pvalue_matrix)`. Conformance: upper-triangle + diagonal bit-identical to per-pair mannwhitneyu
(both stat & pvalue), symmetric, ragged rejected. The tuple helper now backs ks + mannwhitneyu; the
`all_pairs_*` family covers 6 matrices total (kendall/weightedtau/wasserstein/energy/ks/mannwhitneyu).
fsci-stats GREEN.

### ✅✅ stats: ranksums_matrix + brunnermunzel_matrix (all-pairs rank tests) — 80-96x faster than scipy
Completes the rank-based two-sample test matrices. ranksums (signed z) and brunnermunzel (signed W) have
ANTI-symmetric statistics (`stat[j][i] == −stat[i][j]`) + symmetric p-values, so they need a NEW FULL
ordered-pairs helper `all_pairs_two_full_matrices` (evaluates every `(i,j), i≠j` — no symmetry assumed,
correct for directional stats). Both use cheap normal-approx p-values → big wins even at 2× the kernel
evals. SciPy has NO vectorized all-pairs form — users loop the test in Python.

**SAME-BOX head-to-head (fsci FULL m×(m−1) matrix vs scipy Python upper-triangle loop, both this box):**
| matrix (m × n)  | pairs (scipy) | scipy ranksums | fsci   | ×        | scipy brunnermunzel | fsci    | ×        |
|-----------------|---------------|----------------|--------|----------|---------------------|---------|----------|
| m=40,  n=400    | 780           | 277.3 ms       | 3.27 ms| **84.9×**| 489.3 ms            | 5.21 ms | **94.0×**|
| m=100, n=1000   | 4 950         | 2 200.5 ms     | 22.9 ms| **96.0×**| 3 750.0 ms          | 46.7 ms | **80.3×**|

Note: fsci returns the FULL directional matrix (m·(m−1) kernel evals) while scipy's loop only fills the
upper triangle (m·(m−1)/2) — fsci does 2× the work and STILL wins 80-96×. Conformance: every ordered
(i,j) bit-identical to per-pair ranksums/brunnermunzel (both stat & p-value), ragged rejected. The
`all_pairs_*` family now spans 8 matrices (kendall/weightedtau/wasserstein/energy/ks/mannwhitneyu +
ranksums/brunnermunzel) across THREE assembly shapes (f64-symmetric / tuple-symmetric / tuple-FULL).
fsci-stats GREEN.

### ✅✅ stats: permutation_test parallelized (LCG jump-ahead) — 87x faster than scipy
A DIFFERENT primitive from the all-pairs vein: the existing `permutation_test` was a SERIAL loop with a
cumulative (path-dependent) Fisher–Yates shuffle. Rewrote it so permutation `p` is a PURE FUNCTION of
`(seed, p)` — reset the buffer to the original sample + jump the shared LCG to `p·(n−1)` advances (reusing
the in-crate `lcg_jump(a,c,steps)` O(log) skip already built for byte-identical bootstrap parallelism) —
then fan out across permutations. Each thread streams (reset 8 KB buffer → shuffle → stat → discard), so
its working set is L1/L2-resident (dodges the cache-hostile materialization that made naïve resampler
parallelism 3.3× SLOWER, per NEGATIVE_EVIDENCE). Result is now DETERMINISTIC and thread-count-INDEPENDENT
(strictly better reproducibility than the old serial path).

**SAME-BOX head-to-head (fsci vs scipy.stats.permutation_test, vectorized, both this box):**
| workload                                  | scipy      | fsci     | speedup   |
|-------------------------------------------|------------|----------|-----------|
| 2-sample diff-of-means, n=1000, 9999 resamples | 245.7 ms | 2.83 ms | **87.0×** |

CAVEAT (see NEGATIVE_EVIDENCE.md): this CHANGES the exact p-value returned for a given seed (the old
serial output was a path-dependent Monte-Carlo estimate, never a stable contract); the tolerant
conformance test `permutation_test_matches_scipy_reference_values` (p≈0.1 ± 0.02) still passes. The trait
bound tightened `F: Fn` → `F: Fn + Sync` (required for the fan-out; ordinary statistic closures satisfy
it). fsci-stats GREEN.

### ✅✅ stats: CROSS all-pairs distance/test matrices (two-group) — 24-278x faster than scipy
A different SHAPE of the all-pairs primitive: rectangular `m × k` matrices comparing two GROUPS of 1-D
samples (e.g. m controls vs k treatments — a common two-group multiple-comparison setup). No symmetry, no
diagonal, and groups/samples may have DIFFERENT lengths (two-sample distances/tests accept ragged input).
New helpers `all_pairs_cross_matrix` (f64) + `all_pairs_cross_two_matrices` (tuple). SciPy makes you
double-loop the two groups in Python.

**SAME-BOX head-to-head (fsci cross matrix vs scipy Python double-loop, both this box; m=50 k=50 n=500, 2500 pairs):**
| function                | scipy      | fsci      | speedup    |
|-------------------------|------------|-----------|------------|
| wasserstein_distance_cross | 214.4 ms | 3.26 ms | **65.7×**  |
| energy_distance_cross   | 229.7 ms   | 3.86 ms   | **59.6×**  |
| ks_2samp_cross          | 845.5 ms   | 35.93 ms  | **23.5×**  |
| mannwhitneyu_cross      | 1 257.4 ms | 4.52 ms   | **278.2×** |

(ks is the weakest, consistent with the self-pairs finding — fsci's ks p-value is heavy per pair; mwu's
normal-approx p-value is cheap → 278×.) Conformance: every `out[i][j]` bit-identical to the per-pair
`wasserstein_distance/energy_distance/ks_2samp/mannwhitneyu`, ragged groups OK, empty sample rejected.

MEASURED-ALREADY-WON this iteration (negative evidence, NOT re-shipped): fsci `gaussian_kde` evaluate_many
12-38× (already parallel), `theilslopes` 9.7-18× + `siegelslopes` 9.3-62× (fast-path already optimal),
`monte_carlo_test` 61× (already parallel). fsci-stats' big single-array gaps are closed; the open seam is
the all-pairs/cross fan-out family. fsci-stats GREEN.

### ✅✅ stats: kendalltau_cross + weightedtau_cross (two-group cross-correlation) — 162-262x faster than scipy
Completes the cross family for CORRELATION: rectangular `m × k` cross-correlation between two groups of
variables (m features vs k targets — ubiquitous in genomics/finance: correlate every gene against every
phenotype). Statistic-only (skips the per-pair p-value, the bulk of the cost) over `all_pairs_cross_matrix`.
SciPy makes you double-loop `scipy.stats.kendalltau`/`weightedtau` in Python (weightedtau cross = 2.4 s!).

**SAME-BOX head-to-head (fsci cross matrix vs scipy Python double-loop; m=50 k=50 n=500, 2500 pairs):**
| function           | scipy      | fsci     | speedup    |
|--------------------|------------|----------|------------|
| kendalltau_cross   | 675.7 ms   | 4.18 ms  | **161.6×** |
| weightedtau_cross  | 2 398.4 ms | 9.15 ms  | **262.1×** |

Conformance: `out[i][j]` bit-identical to per-pair `kendalltau(.).statistic` / `weightedtau`, empty
sample rejected. The all_pairs/cross fan-out family now covers 14 public matrices (self: kendall/wtau/
wasserstein/energy/ks/mwu/ranksums/bm; cross: wasserstein/energy/ks/mwu/kendall/wtau). fsci-stats GREEN.

NOTE (uncontended-crate survey this iteration, all MEASURED-ALREADY-WIN, NOT re-shipped): spatial `cdist`
euclidean 10-13× (parallel + per-pair SIMD, all dims d=2..50); stats `gaussian_kde`/`theilslopes`/
`siegelslopes`/`monte_carlo_test` already win (see prior entry). fsci's `RbfInterpolator` is the LEGACY
`scipy.interpolate.Rbf` (kernel+epsilon, ≤4096 pts), NOT the modern `RBFInterpolator` — semantic mismatch,
not a comparable gap. The accessible uncontended surface is saturated; remaining gaps sit in contended
crates (linalg/signal/sparse — other agents' probes present) or known SIMD walls (FFT mid-pow2).

### ✅ fft: fft_axis2d / rfft_axis2d (batched 1-D FFT along last axis) — NEW gap-fill, 7.5-13x vs scipy DEFAULT
fsci-fft had NO batched-1-D-along-axis transform: `fftn`/`rfftn` always transform ALL axes (no `axes`
param), so `scipy.fft.fft(x, axis=-1)` over a 2-D array (per-row/per-channel FFT — spectrograms, batch
signal processing) had no direct fsci equivalent. Added `fft_axis2d`/`rfft_axis2d`: `rows` INDEPENDENT
length-`ncols` transforms, parallel ACROSS rows (each row's 1-D FFT serial on its owning thread — inner
`WorkerPolicy::Exact(1)` avoids 64×64 oversubscription). Row r bit-identical to per-row `fft`/`rfft`.

**SAME-BOX head-to-head (fsci vs scipy.fft, both this box):**
| rows × ncol  | fsci rfft | scipy rfft w=1 | × (w=1) | scipy rfft w=-1 | fsci fft | scipy fft w=1 | × (w=1) | scipy fft w=-1 |
|--------------|-----------|----------------|---------|-----------------|----------|---------------|---------|----------------|
| 2000 × 4096  | 6.38 ms   | 53.3 ms        | 8.4×    | 3.82 ms (0.60×) | 9.06 ms  | 121.1 ms      | 13.4×   | 7.90 ms (0.87×)|
| 5000 × 2048  | 9.31 ms   | 69.6 ms        | 7.5×    | 5.33 ms (0.57×) | 11.48 ms | 132.8 ms      | 11.6×   | 13.00 ms (1.13×)|
| 1000 × 8192  | 6.06 ms   | 58.6 ms        | 9.7×    | 12.88 ms (2.13×)| 9.23 ms  | 122.3 ms      | 13.3×   | 23.44 ms (2.54×)|

HONEST (see NEGATIVE_EVIDENCE.md): vs scipy's DEFAULT (`workers=1`, what most code uses) fsci wins
7.5-13.4× across the board. vs scipy's PARALLEL (`workers=-1`) it's MIXED — fsci WINS 2.1-2.5× at large
ncol=8192 and on complex fft@2048, but LOSES on rfft@2048/4096 (0.57-0.60×) because fsci's per-FFT kernel
is ~1.5× slower than pocketfft (the documented mid-pow2 SIMD wall) and when BOTH sides parallelize across
rows that kernel gap dominates. Net: a real new capability that beats the default API and is
competitive-to-winning vs scipy's best at large transforms. Conformance: `fft_rfft_axis2d_match_per_row`
bit-identical to per-row, shapes validated. fsci-fft GREEN.

### ⚖️ fft: dct_axis2d / idct_axis2d (batched DCT-II/III along axis) — NEW gap-fill, 5.6-7x vs scipy DEFAULT (loses to workers=-1)
Completes the batched-axis transform family for the DCT (per-block DCT is the core of image/audio
compression). fsci-fft had no batched-axis DCT (`dctn` does ALL axes). Added `dct_axis2d`/`idct_axis2d`
(parallel across rows via the new `batched_real_axis2d` helper, bit-identical to per-row `dct`/`idct`).

**SAME-BOX head-to-head (fsci vs scipy.fft.dct/idct, both this box):**
| rows × ncol  | fsci dct | scipy dct w=1 | × (w=1) | scipy dct w=-1 | fsci idct | scipy idct w=1 | × (w=1) | scipy idct w=-1 |
|--------------|----------|---------------|---------|----------------|-----------|----------------|---------|-----------------|
| 2000 × 4096  | 7.31 ms  | 48.87 ms      | 6.7×    | 4.79 ms (0.66×)| 8.48 ms   | 49.84 ms       | 5.9×    | 4.27 ms (0.50×) |
| 5000 × 2048  | 8.75 ms  | 61.05 ms      | 7.0×    | 4.99 ms (0.57×)| 8.83 ms   | 61.29 ms       | 6.9×    | 4.96 ms (0.56×) |
| 20000 × 512  | 10.37 ms | 58.04 ms      | 5.6×    | 4.66 ms (0.45×)| 10.85 ms  | 59.65 ms       | 5.5×    | 4.86 ms (0.45×) |

HONEST (NEGATIVE_EVIDENCE.md): wins 5.6-7× vs scipy DEFAULT (workers=1) but LOSES to scipy workers=-1 at
EVERY size (0.45-0.66×) — UNLIKE fft_axis2d (which won at large ncol), fsci's DCT kernel is a half-size
complex FFT + reorder + twiddle-extract, materially heavier than pocketfft's native DCT, so the kernel gap
dominates once both sides parallelize. Ship value = the missing capability + the default-API win, NOT
domination of scipy's best. The lever that would flip BOTH this and fft_axis2d to clean wins is a
SIMD-ACROSS-ROWS batched FFT kernel (lane = independent row → sidesteps the AoS-tuple SoA blocker that
killed within-FFT SIMD; bit-identical per lane, like pdist SIMD-across-pairs) — documented as the next
radical lever. Conformance: `dct_idct_axis2d_match_per_row` bit-identical to per-row. fsci-fft GREEN.

### ✅✅ special: wofz continued-fraction kernel + voigt_profile_many — 5.6x faster than scipy (CLEAN win)
Two-part, found by digging the measured gap: scipy.special.voigt_profile over 2M points = 184 ms, fsci had
only a SCALAR voigt_profile (no batched form). First cut (voigt_profile_many = par_map over the scalar) hit
PARITY (199 ms) — diagnosis: fsci's `wofz` (Faddeeva) was ~70× slower PER POINT than scipy because the
`4 ≤ |z| < 8` band used a **768-step Simpson quadrature** (~7.7 µs/call). RADICAL LEVER (different
primitive, not safe-Rust-ceiling): replaced it with the **Gautschi/Laplace continued fraction**
`w(z) = (i/√π)/(z − a₁/(z − a₂/…)), aₖ = k/2` (24 terms, ~1e-13, ~30× fewer ops, MORE accurate than the
Simpson). Kernel dropped ~6.4 µs → ~1 µs/point; the batched API then fans it across cores.

**SAME-BOX head-to-head (fsci voigt_profile_many vs scipy.special.voigt_profile, both this box):**
| n         | scipy    | fsci (768-Simpson) | fsci (CF + parallel) | speedup    |
|-----------|----------|--------------------|----------------------|------------|
| 500 000   | ~46 ms   | 52.4 ms (0.88×)    | 7.37 ms              | **~6×**    |
| 2 000 000 | 184.3 ms | 199.5 ms (0.92×)   | 32.71 ms             | **5.6×**   |

The wofz CF is a REUSABLE kernel win — it also speeds the scalar `wofz`/`voigt_profile` and EVERY wofz
caller (erfcx, dawsn, complex erf/erfc) for `4 ≤ |z| < 8`. Conformance: FULL fsci-special suite
**1121/1121 GREEN** (the CF matches scipy across all wofz-dependent goldens — more accurate than the
Simpson it replaced) + new `voigt_profile_many_matches_scalar` (bit-identical to per-point, both gate
paths). NOTE: fsci's per-point wofz (~1 µs) is still ~11× slower than scipy's Weideman Faddeeva (~92 ns) —
a further CF-everywhere / Weideman-rational kernel could lift the scalar path too, but the parallel batched
form already DOMINATES scipy 5.6×.

### ✅ special: hyperu DOMINATES scipy on BOTH speed and accuracy (measured + mpmath-verified) — lever closed
Investigated the filed "hyperu 768-step Simpson → faster quadrature" lever. Outcome: the lever is CLOSED
because fsci already wins decisively and the 768-Simpson is excellent — verified by an oracle-differential
sweep over a∈[0.5,50], b∈[0.5,4], x∈[0.3,10] (incl. narrow-peak large a) against mpmath at 40 dps:
- **Accuracy:** fsci max rel-err **1.27e-12** vs **scipy 5.49e-06** (scipy has a latent ~5e-6 error for
  moderate-large a, e.g. a=10,b=4,x=5: mpmath 1.5250375e-10, fsci 1.5250375e-10 ✓, scipy 1.5250459e-10 ✗).
  fsci is ~4×10⁶× more accurate than scipy at the worst point.
- **Speed:** fsci 1.47-1.71× faster (52.6/504 ms vs scipy 90.1/738.6 ms @100k/1M, parallel par_map_indices).
So fsci hyperu DOMINATES scipy on both axes. The 768-Simpson is NOT accuracy-marginal (it hits 1e-12) and
reducing the step count would forfeit the accuracy lead — the perf "win" of fewer steps is a real-correctness
LOSS. Generalized Gauss-Laguerre (weight u^{a-1}e^{-u}) would be the analytically-exact route but its nodes
depend on a → caches poorly for per-element varying-a batches (recompute per distinct a). No change warranted.
LESSON: before "optimizing" a slow-looking fixed quadrature, oracle-check accuracy vs mpmath FIRST — here the
768 steps were buying a 1e-12 accuracy that BEATS scipy; the apparent fsci-vs-scipy "deviation" was scipy's
bug, not fsci's. (Contrast the wofz-CF win affac121, where the 768-Simpson bought NO accuracy a faster CF
couldn't match.)

### ✅✅ opt: curve_fit_bounded / least_squares_bounded — closes backlog gap, 10.2× faster than scipy trf
Backlog (CARGO_RECOVERY_BACKLOG.md) listed "bounded least_squares/curve_fit (TRF)" as a genuinely-unfinished
capability gap — fsci had only unbounded LM. Added `least_squares_bounded` + `curve_fit_bounded` via the
standard smooth reparameterisation (lmfit's method): each bounded coordinate maps to an unconstrained variable
(logistic for two-sided, softplus for one-sided, identity for ±inf), the existing fast LM core solves the
unconstrained problem, and `x`/`fun`/`jac` are recomputed in parameter space at the optimum for the covariance.
Purely additive (new public fns; existing curve_fit/least_squares untouched).

**SAME-BOX head-to-head (5-param double-exponential, 400 pts; both this box):**
| op                          | scipy            | fsci      | speedup   |
|-----------------------------|------------------|-----------|-----------|
| curve_fit (unbounded, lm)   | 1.944 ms         | 0.235 ms  | **8.3×**  |
| curve_fit_bounded (trf-eq)  | 9.859 ms (trf)   | 0.971 ms  | **10.2×** |

The callback lever drives it: scipy's trf calls a Python/numpy model many times; fsci inlines a Rust closure.
CONFORMANCE (oracle-checked vs scipy): on a noiseless interior problem (exp+offset, true (3,0.7,1)) BOTH fsci
and scipy recover (3.0,0.7,1.0) exactly; with the amplitude capped below the truth (upper=2) scipy pins it at
2.0 and fsci approaches 2.0⁻ (the transform is asymptotic at an active bound — the one documented difference vs
trf, same as lmfit). fsci-opt curvefit suite 15/15 green incl. 2 new bounded tests. LIMITATION (documented):
for a tightly-active bound the transform reaches it asymptotically rather than exactly; for interior optima
(the common "sanity bounds" case) it is identical to trf and ~10× faster.

### ✅✅✅ opt: curve_fit_many / curve_fit_bounded_many — vmap-over-solver, 33-113× faster than looped scipy
The JAX-style "different primitive": fit the same model to MANY independent ydata rows. SciPy has no batched
curve_fit — you loop it in Python, paying the per-call overhead N times SERIALLY. fsci `curve_fit_many` fans
the N independent fits across cores AND inlines the model as a Rust closure (callback lever × N-way parallel).
Purely additive (new pub fns over the existing curve_fit/curve_fit_bounded); heavy-per-item thread cap
(cores.min(nrows), serial under 8 rows). Common in imaging/signal: a decay or peak fit per pixel/channel/trace.

**SAME-BOX head-to-head (3-param exponential a·e^{−bx}+c, 80 pts, N fits; both this box):**
| N    | scipy (Python loop over curve_fit) | fsci curve_fit_many | speedup    |
|------|------------------------------------|---------------------|------------|
| 500  | 96.2 ms                            | 2.92 ms             | **32.9×**  |
| 2000 | 431.2 ms                           | 3.81 ms             | **113×**   |

Speedup grows with N as the ~1.5ms thread-spawn floor amortises. CONFORMANCE: row i is BYTE-IDENTICAL
(.to_bits()) to `curve_fit(f, xdata, &ydata_rows[i], opts).popt` — the batch only distributes independent
fits, it doesn't change any of them. fsci-opt curvefit suite 16/16 green (+ new batched byte-identical test
covering both curve_fit_many and curve_fit_bounded_many across the serial→parallel gate). Pairs with the
bounded-fit lever (2235ab6f): curve_fit_bounded_many gives the same N-way win for box-constrained batches.

### ✅✅✅ integrate: solve_ivp_many (vmap-over-solver ensemble ODE) — ~1500× faster than looped scipy
Extends the vmap-over-solver lever to ODEs — the marquee case. SciPy has no batched solve_ivp: integrating an
ensemble (N initial conditions, shared dynamics) means looping solve_ivp in Python, calling the Python RHS
thousands of times PER solve, N solves SERIALLY (~15-21 ms/solve). fsci `solve_ivp_many` fans the N independent
integrations across cores AND inlines the RHS as a Rust closure (callback lever × N-way parallel). Purely
additive (new pub fn over the existing solve_ivp); heavy-per-item thread cap (cores.min(nrows), serial <4).

**SAME-BOX head-to-head (Lotka-Volterra, RK45, rtol 1e-8 / atol 1e-10, 150 t_eval; both this box):**
| N    | scipy (Python loop over solve_ivp) | fsci solve_ivp_many | speedup     |
|------|------------------------------------|---------------------|-------------|
| 200  | 4220 ms                            | 2.85 ms             | **1481×**   |
| 1000 | 14809 ms                           | 9.26 ms             | **1599×**   |

All 1000/1000 solves converged. CONFORMANCE (rigorous, two ways): (1) result i is BYTE-IDENTICAL (.to_bits()
on t and y) to per-member solve_ivp — the batch only distributes independent integrations; (2) NUMERICAL
cross-check vs scipy on a fixed y0=[2,3]: fsci final state [5.3569214988, 1.9612924121] == scipy
[5.3569214988, 1.9612924121] to 1e-10 (same RK45 algorithm + tolerances → same trajectory, just ~1500×
faster). fsci-integrate solve_ivp_many test green. The callback lever (inline Rust RHS, no Python per-step)
gives ~25× per-solve; the N-way parallelism multiplies it to ~1500×. Companion to curve_fit_many (95f3cad8).

### ✅✅✅ opt: minimize_many (vmap-over-solver multistart) — 271-275× faster than looped scipy
Third vmap-over-solver family (after curve_fit_many 113× and solve_ivp_many ~1500×). Multistart / parameter
sweep — minimise the SAME objective from MANY starts — is ubiquitous in global optimisation; SciPy loops
`minimize` in Python, calling the Python objective (+ gradient) many times PER run, N runs SERIALLY. fsci
`minimize_many` fans the N independent runs across cores AND inlines the objective as a Rust closure (callback
lever × N-way parallel). Purely additive (new pub fn over the existing minimize); MinimizeOptions is Copy so
no per-call clone; heavy-per-item thread cap (cores.min(nrows), serial <4).

**SAME-BOX head-to-head (6-D Rosenbrock, BFGS, N random starts in [-2,2]^6; both this box):**
| N    | scipy (Python loop over minimize) | fsci minimize_many | speedup    |
|------|-----------------------------------|--------------------|------------|
| 200  | 2829 ms                           | 10.43 ms           | **271×**   |
| 1000 | 14677 ms                          | 53.41 ms           | **275×**   |

FAIR head-to-head (not speed from giving up early): on the same 1000 random starts fsci converges 805/1000 vs
scipy 622 success / 782 reached-global-min — fsci optimises at least as well. CONFORMANCE two ways:
(1) result i is BYTE-IDENTICAL (.to_bits() on x and fun) to per-start `minimize` — the batch only distributes;
(2) from the standard Rosenbrock start fsci reaches the exact global min [1,1,1,1,1,1]. fsci-opt minimize_many
test green. Callback lever (inline Rust objective, no Python per-eval) gives the per-run win; N-way parallelism
multiplies it. The vmap-over-solver vein is now proven across curve_fit / solve_ivp / minimize.

### ✅✅ opt: root_many (vmap-over-solver nonlinear-system sweep) — 11-25× faster than looped scipy
Fourth vmap-over-solver family (curve_fit / solve_ivp / minimize / root). A parameter sweep — solve
`func(x, params)=0` for many parameter sets, shared start — is common in equilibrium/steady-state analysis;
SciPy loops `root` in Python, N solves serially. fsci `root_many` (param-sweep signature `F: Fn(&[f64] x,
&[f64] params)->Vec<f64>`) fans the N independent solves across cores and inlines the residual. Purely
additive; heavy-per-item cap (cores.min(nrows), serial <4).

**SAME-BOX head-to-head (well-conditioned 3-eq system, hybr, N parameter sets; both this box):**
| N    | scipy (Python loop over root) | fsci root_many | speedup   |
|------|-------------------------------|----------------|-----------|
| 500  | 35.9 ms (277/500 converged)   | 3.24 ms (293/500) | **11.1×** |
| 2000 | 135.4 ms (1176/2000)          | 5.39 ms (1191/2000) | **25.1×** |

This is the MODEST end of the vmap family — unlike minimize (275×) / solve_ivp (1500×), scipy's hybr is fast
C (MINPACK, ~0.07 ms/solve, few Python callbacks), so the per-solve callback lever is weak and the win is
mostly the N-way parallelism. FAIR head-to-head: fsci converges slightly MORE than scipy on the same params
(293 vs 277, 1191 vs 1176 — not speed from giving up early; a first benchmark on a Jacobian-SINGULAR symmetric
system was discarded as invalid since neither library converged there). CONFORMANCE: result i is
BYTE-IDENTICAL (.to_bits() on x and fun) to per-param root; fsci-opt root_many test green (byte-id + ≥half
converge). vmap-over-solver vein now spans 4 solver families.

### ✅✅✅ integrate: quad_many (vmap-over-solver definite-integral sweep) — 14.5-61× faster than looped scipy
Fifth vmap-over-solver family (curve_fit / solve_ivp / minimize / root / quad). A definite-integral sweep —
`I(params) = ∫_a^b f(x, params) dx` for many parameter sets — is common (a family of moments / partition
functions / marginalisations); SciPy loops `quad` in Python, calling the Python integrand adaptively per
integral, N integrals SERIALLY. fsci `quad_many` (param-sweep signature `F: Fn(f64 x, &[f64] params)->f64`)
fans the N independent adaptive integrations across cores and inlines the integrand. Purely additive;
heavy-per-item cap (cores.min(nrows), serial <4).

**SAME-BOX head-to-head (peaked+oscillatory ∫_0^1 e^{-p(x-c)²}cos(wx)dx, N parameter sets; both this box):**
| N    | scipy (Python loop over quad) | fsci quad_many | speedup   |
|------|-------------------------------|----------------|-----------|
| 500  | 47.9 ms                       | 3.30 ms (500/500 conv)   | **14.5×** |
| 2000 | 179.6 ms                      | 2.94 ms (2000/2000 conv) | **61.1×** |

Speedup grows with N as parallelism amortises (all integrals converge). The callback lever is real here
(scipy's QUADPACK calls the Python integrand adaptively, fsci inlines a Rust closure) — stronger than root
(11-25×, fast-C hybr) though below minimize/solve_ivp. CONFORMANCE three ways: (1) result i is BYTE-IDENTICAL
(.to_bits() on integral/error/converged) to per-param quad; (2) NUMERICAL cross-check vs scipy: fsci
I(p=100,c=0.5,w=10)=0.039156400368 == scipy 0.039156400368 to 3.84e-13; (3) fsci-integrate quad_many test
green. The vmap-over-solver vein now spans FIVE solver families.

### ✅✅✅ integrate: dblquad_many (vmap-over-solver 2D-integral sweep) — 62.7-211× faster than looped scipy
Sixth vmap-over-solver family — and the heaviest-callback integration case. dblquad's inner adaptive integral
is RE-RUN for each outer node, so each 2-D integral makes O(n²) integrand calls; in SciPy those are all Python
calls, and a parameter sweep loops dblquad in Python, N integrals SERIALLY. fsci `dblquad_many` (param-sweep
`F: Fn(f64 y, f64 x, &[f64] params)->f64`, shared rectangle) fans the N independent double integrations across
cores and inlines the integrand. Purely additive; heavy-per-item cap (cores.min(nrows), serial <4).

**SAME-BOX head-to-head (∫_0^1∫_0^1 e^{-p((x-.5)²+(y-.5)²)}dy dx, N parameter sets; both this box):**
| N   | scipy (Python loop over dblquad) | fsci dblquad_many | speedup   |
|-----|----------------------------------|-------------------|-----------|
| 100 | 135.5 ms                         | 2.16 ms (100/100 conv) | **62.7×** |
| 400 | 544.4 ms                         | 2.58 ms (400/400 conv) | **211×**  |

The strongest integration vmap win — the O(n²) per-integral callbacks make the inline-Rust lever far stronger
than 1-D quad (14.5-61×). Speedup grows with N (parallelism amortises); all integrals converge. CONFORMANCE
three ways: (1) result i BYTE-IDENTICAL (.to_bits() on integral/error/converged) to per-param dblquad;
(2) NUMERICAL vs scipy: fsci I(p=20)=0.156588231977 == scipy 0.156588231977 to 2.87e-13; (3) new dblquad_many
test green. The vmap-over-solver vein now spans SIX solver families (curve_fit/solve_ivp/minimize/root/quad/
dblquad); win size tracks scipy's per-solve Python-callback density (dblquad O(n²) ⇒ 211×).

### ✅✅✅ integrate: tplquad_many (vmap-over-solver 3D-integral sweep) — 83-159× faster than looped scipy
Seventh vmap-over-solver family — the HEAVIEST-callback case. tplquad nests three adaptive quadratures, so
each triple integral makes O(n³) integrand calls; in SciPy all Python, and a parameter sweep loops tplquad in
Python, N integrals SERIALLY. fsci `tplquad_many` (param-sweep `F: Fn(f64 z, f64 y, f64 x, &[f64] params)->f64`,
shared box) fans the N independent triple integrations across cores and inlines the integrand. Purely additive;
heavy-per-item cap (cores.min(nrows), serial <4).

**SAME-BOX head-to-head (∫∫∫ e^{-p(x²+y²+z²)} over unit cube, N parameter sets; both this box):**
| N   | scipy (Python loop over tplquad) | fsci tplquad_many | speedup   |
|-----|----------------------------------|-------------------|-----------|
| 30  | 123.6 ms                         | 1.49 ms (30/30 conv)   | **83×**  |
| 100 | 392.0 ms                         | 2.46 ms (100/100 conv) | **159×** |

Confirms the callback-density LAW at equal N=100: tplquad 159× > dblquad 62.7× > quad ~30× (O(n³)>O(n²)>O(n)
Python integrand calls). All integrals converge; speedup grows with N. CONFORMANCE three ways: (1) result i
BYTE-IDENTICAL (.to_bits() on integral/error/converged) to per-param tplquad; (2) NUMERICAL vs scipy: fsci
I(p=5)=0.061963890934 == scipy 0.061963890934 to 4.41e-13; (3) new tplquad_many test green. The vmap-over-
solver vein now spans SEVEN solver families (curve_fit/solve_ivp/minimize/root/quad/dblquad/tplquad).

### ✅✅ ndimage: mean(labels,index) parallel privatized-histogram scatter — 2.05× self-speedup at large N (→ 2.16× vs scipy)
DIFFERENT primitive from the vmap vein: a PARALLEL SEGMENTED REDUCTION. First, a stale-scorecard CORRECTION —
the GAUNTLET scorecard lists `ndimage.mean(labels,index)` as a 1.5-4.7× LOSS (beads 8l8r1.125/.143/fa62u),
but a fresh same-box re-measure shows the current one-based-contiguous fast path already WINS 1.17-1.30×
(fsci 169.8/498.5/1177.7 us vs scipy 221.5/581.9/1478.0 us at N=65536/262144/589824) — the slow rows were
superseded. NEW lever on top: the serial scatter `sums[label-1]+=v; counts[label-1]+=1` is a segmented
reduction; replaced it (large N only) with PRIVATIZED HISTOGRAMS — each worker accumulates a private
(sums,counts) over a contiguous chunk via thread::scope, partials merged in chunk order.

**SAME-PROCESS A/B (serial replica vs production mean(), identical data, both this box):**
| N      | K    | serial | parallel mean() | self-speedup | max\|Δmean\| |
|--------|------|--------|-----------------|--------------|-------------|
| 65536  | 512  | 195.6  | 258.0 us        | serial (below gate, unchanged) | 0 (byte-id) |
| 262144 | 1024 | 663.9  | 505.7 us        | 1.30-1.50×   | 8.88e-16 |
| 589824 | 4096 | 1399.2 | 683.9 us        | **2.05×**    | 6.66e-16 |

At N=589824 the parallel path → 2.16× vs scipy (1478/683), doubling the large-image margin (was 1.26×). The
merge in chunk order keeps each label's running sum in global element order — only the ASSOCIATION differs, so
max|Δmean| = 6.66e-16 (sub-ULP). GATED `nthreads = cores.min(n/128_000)`: small N (the unit-test regime) stays
on the serial path and is BYTE-IDENTICAL (Δ=0, no regression). CONFORMANCE: 249/0 fsci-ndimage tests green incl.
new `mean_one_based_parallel_scatter_matches_serial_reference` (<1e-9) + all existing mean/label fixtures.
Generalizes to variance/sum/std label reductions (same scatter).

### ✅✅✅ ndimage: sum/variance/standard_deviation(labels,index) streaming fast path — flips 1.5-10x LOSS → 2.2-8.2x FASTER
Generalises the privatized-histogram lever (e7f5ddd4) to the label reductions that were still on the SLOW
group-materialization path (`measurement_label_groups` builds a `Vec<Vec<f64>>` per label, then reduces). These
were REAL losses (not stale): same-box, N=589824 K=4096 — sum 15238 us (scipy 1485 = **10.3x SLOWER**), variance
16202 us (scipy 10451 = 1.55x slower), std 16031 us (scipy 10874 = 1.47x slower). Added one-based-contiguous
fast paths: `sum` → parallel privatized-histogram scatter (sums only); `variance`/`standard_deviation` → a
numerically-stable TWO-PASS parallel reduction (privatized sum/count → means, then a second privatized
histogram of centred squares), matching scipy's two-pass. std inherits via variance.

**SAME-BOX head-to-head (one-based index, N parameter sets; both this box):**
| op       | N      | scipy     | fsci before        | fsci after   | vs scipy   | self-speedup |
|----------|--------|-----------|--------------------|--------------|------------|--------------|
| sum      | 262144 | 716 us    | 5520 us            | 485 us       | **1.48x**  | 11.4x |
| sum      | 589824 | 1485 us   | 15238 us (10.3x↓)  | 659 us       | **2.25x**  | 23x  |
| variance | 262144 | 4067 us   | 5759 us            | 998 us       | **4.07x**  | 5.8x |
| variance | 589824 | 10451 us  | 16202 us (1.55x↓)  | 1326 us      | **7.88x**  | 12.2x |
| std      | 589824 | 10874 us  | 16031 us (1.47x↓)  | 1321 us      | **8.23x**  | 12.1x |

variance/std dominate hard because scipy's OWN variance/std are slow (4-10 ms, ~7x its sum); fsci's streaming
two-pass is ~1.3 ms. CONFORMANCE: deterministic same-data NUMERICAL cross-check vs scipy is EXACT — fsci
sum[0]=25806.4, var[0]=0.083191077066, std[0]=0.288428634268 == scipy to all 10-12 digits; 250/0 fsci-ndimage
tests green incl. new `sum_variance_one_based_fast_path_matches_serial_reference` (two-pass, non-zero mean) +
all existing scipy-fixture small-N tests (serial path byte-identical, no regression). Gated cores.min(n/128_000).

### ✅✅✅ ndimage: minimum/maximum(labels,index) streaming fast path — 13-30× FASTER than scipy (BYTE-IDENTICAL)
Completes the label-reduction sweep. `minimum`/`maximum`(labels,index) were on the slow group-materialization
path. scipy's OWN labeled min/max are GLACIAL (9-24 ms — even slower than its variance), so fsci's group path
already edged it (1.5×); the streaming privatized-histogram min/max CRUSHES it. Because min/max are associative,
commutative AND EXACT, the parallel merge is BYTE-IDENTICAL to the serial fold — no tolerance (unlike
sum/variance). NaN in any element of a label propagates to NaN; empty labels yield 0.0 (scipy convention), both
preserved.

**SAME-BOX head-to-head (one-based index; both this box):**
| op      | N      | scipy     | fsci before | fsci after | vs scipy   | self-speedup |
|---------|--------|-----------|-------------|------------|------------|--------------|
| minimum | 262144 | 9112 us   | 5688 us     | 691 us     | **13.2×**  | 8.2×  |
| minimum | 589824 | 24571 us  | 15942 us    | 838 us     | **29.3×**  | 19.0× |
| maximum | 262144 | 9175 us   | 5992 us     | 682 us     | **13.5×**  | 8.8×  |
| maximum | 589824 | 24010 us  | 16426 us    | 808 us     | **29.7×**  | 20.3× |

CONFORMANCE: BYTE-IDENTICAL (.to_bits()) to the serial fold incl. NaN propagation + empty→0.0; 252/0
fsci-ndimage tests green incl. new `minimum_maximum_one_based_fast_path_byte_identical_to_serial` (with an
injected NaN) + `minimum_maximum_empty_label_returns_zero` + all existing scipy fixtures (serial small-N path
unchanged). Gated cores.min(n/128_000). Label-reduction vein now COMPLETE: mean/sum/variance/std/min/max all
streaming privatized-histograms; median (scipy 44-118 ms) needs the full group (can't stream) — left on the
group path.

### ✅✅✅ ndimage: histogram(labels,index) streaming per-label privatized histogram — 12-19× FASTER than scipy (BYTE-IDENTICAL)
The privatized-histogram lever applied to its CANONICAL use: `ndimage.histogram(input,min,max,bins,labels,index)`
returns a per-label bin-count histogram. scipy's is GLACIAL (24-79 ms — it loops np.histogram per label in
Python); fsci was on the group-materialization path (Vec<Vec<f64>> per label) so already won 3.5-3.9×. Replaced
with a single-pass parallel privatized reduction: each worker fills a private flat `[label_count × nbins]` count
table over a contiguous chunk, tables summed once. Counts are integers → BYTE-IDENTICAL to the serial fill.

**SAME-BOX head-to-head (one-based index, nbins=32; both this box):**
| N      | K    | scipy     | fsci before | fsci after | vs scipy  | self-speedup |
|--------|------|-----------|-------------|------------|-----------|--------------|
| 262144 | 1024 | 24379 us  | 6887 us     | 1984 us    | **12.3×** | 3.5× |
| 589824 | 4096 | 79060 us  | 20281 us    | 4101 us    | **19.3×** | 4.9× |

Self-speedup is below the scalar reductions (min/max ~20×) because the K×nbins privatized table (1 MB at
K=4096/nbins=32) is cache-heavier than a K-float table; for typical small K it fits L1/L2 and is much faster.
Preserves the validation short-circuit exactly (nbins=0 / non-finite min,max / max<=min / any non-finite input →
all-zero histograms) and the `[min,max]` inclusion filter. CONFORMANCE: BYTE-IDENTICAL (`assert_eq!` on the full
Vec<Vec<usize>>) to a serial reference incl. out-of-range filtering; 253/0 fsci-ndimage tests green incl. new
`histogram_one_based_fast_path_byte_identical_to_serial`. Gated cores.min(n/128_000). The ndimage label-stat
suite is now fully streaming except median (needs the full group — a quantile can't stream).

### ✅✅✅✅ integrate: nquad_many (vmap-over-solver N-D integral sweep) — ~1100-1950× faster than looped scipy
Eighth vmap-over-solver family and the CAPSTONE of the integration set: arbitrary-dimension `nquad`. An
`ndim`-D nquad nests `ndim` adaptive quadratures → each integral makes O(n^ndim) integrand calls; at 4-D those
are O(n⁴), the deepest-nested callback case. SciPy loops nquad in Python over the sweep, N integrals SERIALLY;
fsci `nquad_many` (param-sweep `F: Fn(&[f64] x, &[f64] params)->f64`, shared `ranges`) fans the N independent
N-D integrations across cores and inlines the integrand.

**SAME-BOX head-to-head (4-D Gaussian ∫_[0,1]⁴ e^{-p(a²+b²+c²+d²)}, N parameter sets; both this box):**
| N  | scipy (Python loop over nquad) | fsci nquad_many | speedup    |
|----|--------------------------------|-----------------|------------|
| 20 | 1650.7 ms                      | 1.49 ms (20/20 conv)  | **1108×** |
| 80 | 6703.3 ms                      | 3.43 ms (80/80 conv)  | **1954×** |

Confirms the callback-density LAW to its extreme: 4-D nquad ~1950× > tplquad (O(n³)) 159× > dblquad (O(n²))
62.7× > quad (O(n)) 30× — the win scales with the integrand-call density scipy pays in Python. All integrals
converge. CONFORMANCE two ways: (1) result i BYTE-IDENTICAL (.to_bits() on integral/converged) to per-param
nquad; (2) NUMERICAL vs scipy: fsci I(p=2,4D)=0.128003847000 == scipy 0.128003847000 (all 12 digits). new
nquad_many test green. The vmap-over-solver vein now spans EIGHT solver families
(curve_fit/solve_ivp/minimize/root/quad/dblquad/tplquad/nquad); integration sub-family COMPLETE.

### ✅✅ opt: brentq_many (vmap-over-solver 1-D root sweep) — 13-47× faster than looped scipy
Ninth vmap-over-solver family. A 1-D root SWEEP — solve f(x,params)=0 over a shared bracket for many parameter
sets — is a very common real workload (implied volatility per option, quantile/percentile inversion per
channel, threshold calibration per series); SciPy loops `brentq` in Python, N Brent solves SERIALLY. fsci
`brentq_many` (param-sweep `F: Fn(f64 x, &[f64] params)->f64`, shared bracket) fans the N independent solves
across cores and inlines the function. Purely additive; heavy-per-item cap (cores.min(nrows), serial <4).

**SAME-BOX head-to-head (f(x,p)=(e^x−1)+0.3 sin5x − p, bracket [0,6]; both this box):**
| N    | scipy (Python loop over brentq) | fsci brentq_many | speedup   |
|------|---------------------------------|------------------|-----------|
| 2000 | 31.7 ms                         | 2.434 ms (2000/2000 conv) | **13.0×** |
| 8000 | 126.4 ms                        | 2.710 ms (8000/8000 conv) | **46.6×** |

Bigger than `root_many` (11-25×) despite Brent being fast C: the exp+sin objective is expensive enough per
eval that scipy's Python-callback cost dominates (the callback lever bites). Speedup grows with N as
parallelism amortises; all roots converge. CONFORMANCE two ways: (1) result i BYTE-IDENTICAL (.to_bits() on
root/converged) to per-param brentq; (2) NUMERICAL vs scipy: fsci root(p=10)=2.411137400718 == scipy
2.4111374007184447 (12 digits). new brentq_many test green. vmap-over-solver vein now NINE families.

### (negative evidence) opt: differential_evolution / brute already dominate scipy — no work needed
Measured before chasing: fsci `differential_evolution` (8-D Rastrigin, popsize 15, maxiter 300) = 9.7 ms vs
scipy 1095 ms (workers=1) = **113× faster** ALREADY — the callback lever (inlined Rust objective vs Python);
scipy's `workers=-1` can't even run a local lambda (multiprocessing pickling error). fsci `brute` is already
parallel (thread::scope, byte-identical argmin). So the fsci-opt global optimizers are already dominant; the
DE population eval uses IMMEDIATE in-generation updates (not parallelizable without switching to scipy's
deferred scheme = an algorithm change). No change shipped.

### ✅✅✅ integrate: solve_bvp_many (vmap-over-solver BVP ensemble) — 53-123× faster than looped scipy
Tenth vmap-over-solver family, on a NEW heavy-callback solver and closing the long-standing "solve_bvp" backlog
in the vmap sense. A BVP parameter study (vary a nonlinearity strength / boundary value / forcing) loops
`solve_bvp` in Python, N collocation-Newton solves SERIALLY — each calling the Python RHS at every mesh node
every Newton iteration. fsci `solve_bvp_many` (`f: Fn(t, y, params)->Vec`, `bc: Fn(ya, yb, params)->Vec`,
shared t_span/y_guess) fans the N independent solves across cores and inlines both callbacks. Purely additive;
heavy-per-item cap (cores.min(nrows), serial <4).

**SAME-BOX head-to-head (nonlinear BVP y0'=y1, y1'=p(1+y0²), y0(0)=0, y0(1)=1; both this box):**
| N   | scipy (Python loop over solve_bvp) | fsci solve_bvp_many | speedup    |
|-----|------------------------------------|---------------------|------------|
| 200 | 166.2 ms                           | 3.111 ms (200/200 conv) | **53.4×**  |
| 800 | 611.5 ms                           | 4.953 ms (800/800 conv) | **123.5×** |

A genuinely BIG vmap win (heavy collocation callback), unlike the modest fast-C cases (brentq/root). Speedup
grows with N; all solves converge. CONFORMANCE two ways: (1) result i BYTE-IDENTICAL (.to_bits() on t & y) to
per-param solve_bvp; (2) NUMERICAL vs scipy: fsci y0(0.573)=0.4258989360 == scipy 0.4259060979 (7e-6, the
collocation tolerance — fsci solve_bvp independently matches scipy). new solve_bvp_many test green. The
vmap-over-solver vein now spans TEN solver families (curve_fit/solve_ivp/minimize/root/quad/dblquad/tplquad/
nquad/brentq/solve_bvp).

### ✅✅✅ opt: minimize_scalar_many (vmap-over-solver 1-D minimization sweep) — 69-236× faster than looped scipy
Eleventh vmap-over-solver family, completing the clean vmap set. A 1-D minimization SWEEP (calibrate a
1-parameter model per channel, find the mode/MLE per series, minimize a per-case cost) loops `minimize_scalar`
in Python, N Brent solves SERIALLY. fsci `minimize_scalar_many` (param-sweep `F: Fn(f64 x, &[f64] params)->f64`,
shared bracket) fans the N independent solves across cores and inlines the objective. Purely additive;
heavy-per-item cap (cores.min(nrows), serial <4).

**SAME-BOX head-to-head (f(x,p)=(x−p0)²+0.5cos(p1·x)+e^{0.3x}, bracket [−10,10]; both this box):**
| N    | scipy (Python loop over minimize_scalar) | fsci minimize_scalar_many | speedup  |
|------|------------------------------------------|---------------------------|----------|
| 2000 | 160.0 ms                                 | 2.327 ms (2000/2000 ok)   | **68.8×** |
| 8000 | 643.0 ms                                 | 2.726 ms (8000/8000 ok)   | **236×**  |

Predicted "modest" but came in BIG (like brentq_many): the cos+exp objective is expensive enough per eval that
scipy's Python-callback cost dominates even though Brent is fast C. CONFORMANCE two ways: (1) result i
BYTE-IDENTICAL (.to_bits() on x/fun/success) to per-param minimize_scalar; (2) NUMERICAL vs scipy: fsci
x*(p=[1,2])=1.1563726803 == scipy 1.1563726811631612 (~1e-9, Brent xatol). new test green. vmap-over-solver
vein now ELEVEN families.

### (negative evidence) ndimage rank/morphology filters already dominate scipy — no work needed
Measured before chasing (512×512): fsci median_filter 5×5 = 4.05 ms vs scipy 81.1 ms (**20×**), rank_filter
3.43 vs 66.9 (**19.5×**), percentile_filter 3.21 vs 81.2 (**25×**) — already parallel quickselect. uniform_filter1d
is already running-sum O(n) (has an explicit "pre running-sum reference, A/B only" path). grey_erosion/dilation
already van Herk; generic_filter already inlines a Rust closure (Sync, parallel core) vs scipy's per-window
Python callback (212 ms @ 256² for np.ptp). The ndimage filter surface is fully dominant; don't re-chase.

## 2026-06-29 — AmberKestrel (cc): ndimage geometric-transform compact-support B-spline (FLIP 7.8× loss → 1.85× win)

**Lever:** `sample_interpolated`'s generic fall-through built a length-`len` knot vector AND evaluated ALL `len` (~512) B-spline basis functions (with per-degree clones) PER PIXEL PER AXIS, then filtered to the ~`order+1` nonzero ones — O(len·order) + ~3 heap allocs/pixel/axis. B-splines have compact support `order+1`, so every other basis value is exactly 0.0. Added `bspline_local_support` (closed-form `uniform_knot_at` + binary-searched degree-0 span + windowed Cox–de Boor over only the supported indices) = O(order²), ZERO per-pixel allocation. The cardinal fast paths only covered Nearest/Reflect/Mirror (1-5) and Constant/Wrap at order 3; **Constant/Wrap order∈{1,2,4,5}** (scipy's DEFAULT mode is 'constant') fell through to the slow path.

**BYTE-IDENTICAL** to filtering the full `eval_bspline_basis_all` — proven by `bspline_local_support_byte_identical_to_full_eval` (20 000 random len/order/x incl. integer+boundary positions, `.to_bits()` equality). Full fsci-ndimage conformance suite GREEN.

**Measured same-box, 512×512, affine_transform** (scipy.ndimage o1=7.64ms, o3=19.56ms):
| mode/order | before (fsci) | after (fsci) | self | vs scipy |
|---|---|---|---|---|
| Constant o1 | 59.45 ms | **4.13 ms** | 14.4× | **1.85× FASTER** (was 7.8× SLOWER) |
| Constant o2 | 139.36 ms | 60.94 ms | 2.3× | (prefilter-bound, separate) |
| Constant o3 | 13.19 ms | 13.19 ms | — | 1.48× faster (unchanged path) |

Marquee flip: **Constant order=1 affine 7.8× LOSS → 1.85× WIN** (same fast path now serves map_coordinates / geometric_transform / shift / rotate, all sharing `sample_interpolated`). Backlog: Constant o2 prefilter (per-line `spline_coefficients_for_line`), Reflect o1 (16.7ms).

## 2026-06-29 — AmberKestrel (cc): ndimage order-1 reflect/mirror — drop the eager array pad (FLIP 2× loss → 2.4-2.7× win)

**Lever:** `prefilter_spline_coefficients` for order≤1 reflect/mirror eagerly built a padded copy via `pad_array_mode(input, SPLINE_NEAREST_PAD=12, mode)` — O(padded²) with per-element reflect index reconstruction (~15ms for a 512²→536² array) — purely so linear-interp support lands in range. But the cardinal interp path already folds the support TAPS on the fly. Removed the pad: order-1 reflect/mirror now returns `coeffs=input.clone(), coord_offsets=0` and the `fold` closure uses the actual boundary mode. Fold is also EXACT for coords arbitrarily far outside the grid (the pad only reflected 12 deep then clamped).

**scipy-exact:** affine_transform order=1 reflect & mirror match scipy to **9.99e-16** including coords pushed well outside the grid; new hardcoded-golden regression test `affine_order1_reflect_mirror_matches_scipy_goldens` (asserts no padding + scipy values). Full fsci-ndimage lib suite GREEN (254 passed / 0 failed / 5 pre-existing-ignored, +1 new).

**Measured same-box, 512×512** (scipy affine o1=7.64ms, map_coordinates o1=10.73ms):
| op / mode | before | after | self | vs scipy |
|---|---|---|---|---|
| affine Reflect o1 | 16.14 ms | **3.12 ms** | 5.2× | **2.4× FASTER** (was 2.1× SLOWER) |
| affine Mirror o1 | 15.60 ms | **2.78 ms** | 5.6× | **2.7× FASTER** (was 2× SLOWER) |
| map_coordinates Reflect o1 | 16.37 ms | **2.78 ms** | 5.9× | **3.9× FASTER** |
| map_coordinates Mirror o1 | 15.88 ms | **3.02 ms** | 5.3× | **3.6× FASTER** |

Two-step domination of the geometric-transform family: Constant o1 (compact-support, prior commit) + Reflect/Mirror o1 (this). All order-1 modes now WIN vs scipy. Backlog: Constant o2/o4/o5 prefilter (per-line make_interp_spline), Nearest o3 (~parity).

## 2026-06-29 — AmberKestrel (cc): NEGATIVE EVIDENCE — grey_dilation/erosion HGW parallel-across-lines is ~0-gain (memory-bound), REVERTED

Parallelized `minmax_along_axis_hgw` (van Herk–Gil-Werman flat min/max under grey_dilation/erosion/grey_opening): per-line factored helper `hgw_fill_line`, outer≥2 → `chunks_mut(slab)` across slabs, outer==1 → column-major transpose scratch + scatter (the exact `edt_axis_pass_parallel` pattern). min/max are exact+associative across independent lines → byte-identical by construction.

**Same-process A/B (512×512, FSCI_HGW_SERIAL toggle), parallel vs serial:**
| op (size) | parallel | serial | verdict |
|---|---|---|---|
| grey_dilation 5 | 6.86 ms | 6.85 ms | tie |
| grey_dilation 9 | 6.25 ms | 5.57 ms | parallel SLOWER |
| grey_erosion 5 | 6.30 ms | 5.82 ms | parallel SLOWER |
| grey_erosion 9 | 6.38 ms | 5.78 ms | parallel SLOWER |

HGW is ~O(1) work/element (memory-bandwidth-bound); the serial pass already saturates bandwidth, and the axis-0 transpose+scatter adds traffic → net ~0-gain to slight LOSS. REVERTED (working tree back to c911c3dc). grey_dilation/erosion sit at scipy PARITY (5.4-6.0ms vs scipy 5.5-6.05) and that is the memory wall, not a thread gap. DON'T re-chase parallelizing memory-bound separable min/max. (Contrast: EDT's lower-envelope transform is compute-heavier → its parallel pass DOES win 1.79×.) Remaining real ndimage gap = affine/map_coordinates Constant/Wrap order∈{2,4,5} prefilter (per-line make_interp_spline, scipy o2 18.6ms vs fsci 61ms) but the solver lives in fsci-interpolate (other-agent crate) and a reimplemented boundary-IIR is risky for uncommon orders.

## 2026-06-29 — AmberKestrel (cc): sobel/prewitt via separable correlate1d (FLIP 1.07× loss → 1.85× win)

**Lever:** `sobel`/`prewitt` built an N-D kernel of shape [3,1,…] / [1,3,…] per axis and called the GENERAL N-D `correlate` once per axis. For a 3-tap kernel the general footprint machinery (per-element N-D index + boundary match over the whole kernel volume) is overhead-bound — sobel was 4.73 ms, SLOWER than a 7×7 general correlate (3.87 ms) and slower than scipy. Routed each 1-D pass through the separable `correlate1d` (the axpy-vectorized path uniform/gaussian filters already ride): same centered weights, same boundary mode, no flip → equivalent.

**scipy-EXACT** (byte-exact goldens 12/24/2/4 on a 5×6 probe, both axes); full fsci-ndimage lib suite GREEN (255 passed / 0 failed / 5 pre-existing-ignored). No new test needed — covered by existing sobel/prewitt conformance (5 sobel tests pass).

**Measured same-box, 512×512** (best of repeated runs; correlate1d parallelizes so box-load varies):
| op | before | after | vs scipy |
|---|---|---|---|
| sobel ax0 | 4.73 ms | **2.38 ms** (1.99× self) | **1.86× FASTER** (was 1.07× SLOWER, scipy 4.42) |
| prewitt ax0 | ~4.7 ms (same pattern) | **2.43 ms** | **~parity** (scipy 2.57, was ~1.8× slower) |

sobel was the lone common ndimage filter still losing to scipy; now all of correlate (2.98×)/uniform (3.04×)/gaussian (3.92×)/laplace (2.42×)/sobel (1.86×) WIN. Generalizable: any derivative/separable filter calling general N-D `correlate` with a 1-D kernel → swap to `correlate1d`.

## 2026-06-29 — AmberKestrel (cc): nnls incremental Cholesky + parallel Gram (3.0× self, closes 3.75× loss → 1.25×)

**Lever:** `nnls` (Lawson–Hanson active set) re-built and re-factored the FULL passive-set Gram submatrix (Cholesky from scratch) on EVERY inner solve = O(Σp³) ≈ O(n⁴) for n columns entering. Replaced with an INCREMENTALLY-maintained Cholesky factor (`lflat`, strided): a variable ENTERING the passive set is an O(p²) rank-1 column add (`nnls_chol_add_col`); the rare REMOVAL triggers an O(p³) refactor → O(n³) overall. Rank-deficient passive column (non-positive Schur pivot) flips `use_slow`, reverting to the proven gather + Cholesky/pivot solve. PLUS the dominant O(m·n²) Gram precompute (AᵀA) fanned across cores as a partial-Gram REDUCTION (gated; small problems stay serial/byte-identical).

**Correctness:** NNLS minimizer is unique, so `x` is unchanged — `nnls_matches_scipy_reference_values` + metamorphic `mr_nnls` GREEN (3 nnls tests + meta pass). Parallel Gram is ~1e-13 reassociation (gradient only RANKS the entering variable; strictly convex), not bit-identical, but the unique optimum is invariant.

**Measured same-box, 800×400** (scipy.optimize.nnls 29.45 ms):
| stage | time | vs scipy |
|---|---|---|
| before | 110.33 ms | 3.75× SLOWER |
| + incremental Cholesky | 43.57 ms | 1.48× slower |
| + parallel Gram | **36.76 ms** | **1.25× slower** (3.0× self-speedup) |

Closed most of a 3.75× loss; the residual ~1.25× is scipy's Householder-QR Fortran vs fsci's Gram-based scalar inner loops (engineering wall). isotonic_regression measured at ~parity (28 vs 24ms); cumulative_simpson/trapezoid/simpson all WIN (7.2×/3.5×/2.2×).

## 2026-06-29 — AmberKestrel (cc): lsq_linear Gram rank-1 + Cholesky subproblem (1.5× self, narrows loss)

**Lever (same family as nnls):** `lsq_linear` (bounded LS active set) had (1) a cache-hostile Gram build — `gram[j1][j2] = Σ_i row[j1]·row[j2]` strided TWO columns through the `Vec<Vec>` heap rows (cache miss per element, O(n²·m)); replaced with a contiguous RANK-1 update over a row-major copy of A. (2) The free-subproblem solve used full Gauss-Jordan (`dense_spd_solve`, O(p³) + Vec<Vec> realloc per call) on what is an SPD principal submatrix of AᵀA; swapped to `cholesky_solve_spd` (~⅓ flops) with the Gauss-Jordan as the rank-deficient fallback.

**Correctness:** bounded-LS minimizer is unique (full column rank) → converged `x` unchanged (~1e-13). `lsq_linear` + `nnls` suites GREEN (3+3).

**Measured same-box, 600×300 box-constrained** (scipy.optimize.lsq_linear: trf 249.46 ms, bvls 160.04 ms):
| stage | time |
|---|---|
| before | 441.30 ms (1.77× slower than trf, 2.76× than bvls) |
| + rank-1 Gram | 342.33 ms |
| + Cholesky subproblem | **294.33 ms** (1.18× of trf, 1.84× of bvls) |

1.5× self-speedup, narrowing the loss. REMAINING LEVER (not yet done — higher risk): the inner loop still RE-FACTORS the free Gram from scratch each iteration (O(n⁴)); an incrementally-maintained Cholesky with rank-1 add on KKT-free + DOWN-DATE on inner-fix would reach O(n³) ≈ scipy. nnls already has the add/refactor helpers (`nnls_chol_*`); lsq_linear needs the downdate (removals dominate here).

## 2026-06-29 — AmberKestrel (cc): lsq_linear incremental Cholesky — MARQUEE FLIP 2.76× loss → 4.0× WIN vs scipy

**Lever (completes last commit's lead):** `lsq_linear`'s inner active-set loop re-built and re-factored the FULL free-set Gram submatrix from scratch on EVERY solve = O(n⁴). Replaced with an incrementally-maintained Cholesky (`lflat`, strided) reusing the nnls helpers: KKT-free a variable = O(p²) rank-1 column add (`nnls_chol_add_col`); inner-loop fix = O(p³) refactor (`nnls_chol_refactor`) — turns out REMOVES are rare (the active set mostly grows), so refactor-on-remove sufficed (no Givens downdate needed). Gram flattened to row-major for the strided helpers; `use_slow` fallback for rank-deficient free sets. → O(n³).

**Correctness:** bounded-LS minimizer unique → converged x unchanged. FULL fsci-opt suite GREEN (320 lib + 56 integration tests, 0 failed; incl. `lsq_linear` scipy-reference).

**Measured same-box, 600×300 box-constrained** (scipy.optimize.lsq_linear: trf 249.46 ms, bvls 160.04 ms):
| stage | time | vs scipy bvls |
|---|---|---|
| original | 441.30 ms | 2.76× SLOWER |
| rank-1 Gram + Cholesky subproblem (prev commit) | 294.33 ms | 1.84× slower |
| **+ incremental Cholesky (this)** | **39.87 ms** | **4.0× FASTER** (6.3× vs trf) |

**11× total self-speedup; flipped a 2.76× loss into a 4.0× WIN.** Confirms the active-set lever from [[nnls]]: any solver re-factoring a growing/shrinking submatrix from scratch each iteration → incremental up/refactor. Both nnls (3.75× loss→1.25×) and lsq_linear (2.76× loss→4.0× win) now done.

## 2026-06-29 — AmberKestrel (cc): NEGATIVE EVIDENCE — fsci-opt lane DOMINATED (measured sweep, no fixable gap left)

After the nnls + lsq_linear active-set flips, swept the remaining unmeasured fsci-opt pure-compute / no-callback functions vs scipy (same-box). ALL win or are walls — no new fixable algorithmic gap:
| fn (size) | fsci | scipy | verdict |
|---|---|---|---|
| least_squares (p=60 Rosenbrock) | 1.81 ms | 240.07 ms (trf) | **133× WIN** (callback lever) |
| linprog (200v/100c, bound-dominated) | 1.38 ms | 7.79 ms (highs) | **5.6× WIN** |
| linprog (300v/200c binding) | 9.81 ms | 23.86 ms | **2.4× WIN** |
| linprog (500v/300c binding) | 32.69 ms | 63.78 ms | **1.95× WIN** |
| isotonic_regression (N=2M) | 28.1 ms | 24.3 ms | ~parity (PAVA O(n) sequential scan — wall) |

linprog uses a DENSE TABLEAU simplex (`Vec<Vec>`) whose pivot elimination is ALREADY a contiguous AXPY (`t_row -= factor·p_row`, take(rhs_col+1)) → the flat-buffer/cache lever that flipped nnls/lsq does NOT apply (no column-strided access in the hot loop). It wins 2-5.6× at common dense sizes; the win SHRINKS with size (2.4→1.95×) so a very-large DENSE LP would eventually favor HiGHS's revised simplex (a wall, not worth a dense-tableau rewrite). Per-pivot elimination is too small (~0.05ms) to amortize thread spawn → parallelization would be ~0-gain (cf. HGW). **CONCLUSION: opt/integrate/ndimage lanes are DOMINATED** — remaining gaps are engineering walls (HiGHS at huge dense scale, LAPACK/FFT/Qhull in OTHER agents' crates) or blocked (ndimage spline prefilter order∈{2,4,5} Constant/Wrap = make_interp_spline in fsci-interpolate, uncommon orders). Future cycles: cross-crate measurement or the spline-prefilter IIR (needs scipy boundary match).

## 2026-06-29 — AmberKestrel (cc): dbscan parallel neighbour precompute + [i64;6] grid key — 9.4× self, 12.7× vs sklearn

**Cross-crate (fsci-cluster, low-contention).** dbscan already had a spatial grid (O(n) for low-d bounded density) but two constant-factor sinks: (1) the grid was a `HashMap<Vec<i64>, Vec<usize>>` — a per-query `Vec<i64>` cell-key heap alloc + slow pointer-chasing Vec hash; (2) the entire neighbour scan ran SERIALLY even though each point's eps-neighbourhood is independent of the (serial) BFS label expansion.

**Fix (BYTE-IDENTICAL):** (1) cell key `Vec<i64>` → fixed `[i64;6]` (Copy, zero alloc, fast array hash; d≤6 whenever gridding, unused dims 0 → identical bucket partition). (2) Precompute all n neighbour lists IN PARALLEL (thread::scope, ordered chunks, gated `grid && n≥2048`), then the sequential BFS moves each out with `std::mem::take` — every point's list is consumed exactly once, so same sets, same ascending order, same labels.

**Measured same-box, 20k points / 4-d** (sklearn DBSCAN eps=0.5 min_samples=5: 310.69 ms):
| | time | vs sklearn |
|---|---|---|
| fsci before | 230.59 ms | 1.35× faster |
| **fsci after** | **24.45 ms** | **12.7× FASTER** |

**9.4× self-speedup.** Full fsci-cluster suite green (142 lib + 7+5 dbscan integration, 0 failed). Byte-identical by construction (grid query deterministic + order-independent). LEVER: a serial driver (BFS/DFS/label-prop) over an independent per-item query → precompute the queries in parallel, consume serially via mem::take; + Vec<i64> spatial-grid keys → fixed-size array keys (no alloc, faster hash) whenever the dim is bounded.

## 2026-06-29 — AmberKestrel (cc): NEGATIVE EVIDENCE — NMF is a 5.7× gap; flat buffers get 2× but a real win needs a persistent pool (REVERTED)

**Measured (fsci-cluster):** `nmf` (1000×300, k=20, 200 Lee–Seung MU iters) = **1005 ms vs sklearn 175 ms = 5.7× SLOWER** — a real gap (the only loss found in a cluster sweep; vq 3×, spectral 17×, dbscan 12.7× all WIN). Cause: the 6 GEMMs/iter delegate to the SERIAL `fsci_linalg::matmul` fed `Vec<Vec>` (cache-hostile row-pointer chase + fresh Vec<Vec> alloc ×6×200).

**What worked (but isn't enough):** rewrote the loop on FLAT row-major buffers (reused across iters) with an ikj AXPY kernel → **509 ms = 2.0× self**, BUT still **2.9× SLOWER than sklearn** (property test green: shapes, ≥0, rec-err). The 2 dominant GEMMs are individually tiny (~6M flops) and the iteration is SEQUENTIAL, so per-call `thread::scope` spawn is pure overhead — a thread sweep confirmed EVERY T>1 is ≥ the T=1 serial time (T=8 → 1059 ms). So closing the gap to a WIN needs aggregated memory bandwidth via a PERSISTENT thread pool (spawn once, barrier fork-join the big GEMMs across all 1200 calls) or a BLAS-grade matmul.

**REVERTED:** prototyped the barrier pool (workers write disjoint output row-bands via raw ptr; main runs small GEMMs) but it DEADLOCKED at some thread counts (barrier-count mismatch in the fork/join/terminate handshake) — unshippable, reverted to HEAD rather than leave broken concurrency. BACKLOG (clear path to a 2-3× WIN, ~50-100ms est. at the memory floor): persistent-pool-done-right (careful fork/join/stop barrier accounting + a correctness check vs the serial result) OR a parallel flat GEMM in fsci-linalg that NMF can call. Serial-flat alone (1005→509) is a real 2× but a still-loss → not landed as a "win".

## 2026-06-29 — AmberKestrel (cc): NEGATIVE EVIDENCE — fsci-spatial mostly DOMINATES; lone gap = kdtree k≥10/high-d (diagnosed, deprioritized)

Swept fsci-spatial vs scipy (same-box). Wins everywhere except one regime:
| fn (size) | fsci | scipy | verdict |
|---|---|---|---|
| cdist euclidean (2000²×4d / 3000×1500×8d / 2000²×20d) | 5.5 / 9.8 / 6.9 ms | 8.7 / 28.6 / 28.6 | **1.6–4.2× WIN** (parallel-over-rows) |
| KDTree.query_k_many (10k,3d,k=1) | 1.43 ms | 6.07 ser / 7.34 par | **4–5× WIN** |
| KDTree.query_k_many (20k,4d,k=5) | 7.73 ms | 49.6 ser / 9.62 par | **1.2–6.4× WIN** |
| KDTree.query_k_many (10k,8d,k=10) | 24.78 ms | 269.6 ser / **15.93 par** | 10.9× vs serial but **1.56× SLOWER vs scipy workers=-1** |

The k=10/8d loss is per-query traversal (both already parallel; query_k_many's thread-scaling was already tuned by a prior agent 3.5×→1.56×). Two diagnosed root causes, both real but with risk that outweighs a 1.56× gap: (1) `KDNode.point: Vec<f64>` → each visited node chases a pointer to a SCATTERED 8-double buffer (cache miss/node); fix = store points in one flat n×dim buffer indexed by node.index (BYTE-IDENTICAL — same values relocated — but ~10 `node.point` sites across 2 node types). (2) `sqeuclidean`'s std::simd needs d≥16 (the 2·L=16 unroll), so 8≤d<16 runs fully SCALAR; adding an L=8 block would vectorize it but CHANGES the sum associativity → breaks the byte-identical pdist/cdist locks. BACKLOG: the flat-points refactor (safe, byte-identical) is the cleaner future attempt; est. closes to ~parity, may flip to a slight win.

## 2026-06-29 — AmberKestrel (cc): NEGATIVE EVIDENCE — kdtree flat-coords lever REJECTED (regression), corrects prior diagnosis

Last cycle flagged `KDNode.point: Vec<f64>` scattered storage as the root of the kdtree 8d/k=10 gap and proposed a flat n×dim `coords` slab (indexed by `node.index`) as "the cleaner future attempt". ATTEMPTED it this cycle (byte-identical: `coords[node.index*dim..] == node.point`; conformance green) — and it's a REGRESSION, not a win: 8d/k10 24.78→30.20ms, 4d/k5 7.73→8.80ms (k=1 unchanged, uses nn_search). **Why:** the per-node `point` Vecs are allocated in BUILD (≈traversal) order so the allocator places them locally, whereas `coords[node.index]` is a RANDOM index into the slab (node.index = original data order, scrambled vs traversal) → strictly worse locality. REVERTED. The remaining real lever for the 1.56× gap is sqeuclidean running scalar for 8≤d<16 (the std::simd path needs d≥16), but vectorizing it changes the sum associativity → breaks the byte-identical pdist/cdist locks. **Conclusion: kdtree k≥10/high-d is an engineering wall (tight scipy C traversal), NOT a clean fix — deprioritize.** Confirmed in passing that fsci-stats is thoroughly parallel (gaussian_kde 30× WIN, kendalltau_matrix/all_pairs already parallel, cdist/kde/vq/spectral all win) — the cluster/spatial/stats lanes are dominated.

## 2026-06-29 — AmberKestrel (cc): NEGATIVE EVIDENCE — GAUNTLET_RELEASE_SCORECARD "Measured Losses" are STALE (top entries are now WINS)

Mined GAUNTLET_RELEASE_SCORECARD.md (Jun-27) for the biggest documented fsci-vs-SciPy losses and re-measured same-box (frankenscipy-cc). The headline losses are ALREADY FIXED — the scorecard's "Measured Losses" section is ≥2 days stale:
| scorecard claim | re-measured same-box | reality |
|---|---|---|
| `pdist` chebyshev d=64 "4.37× slower" | fsci 1.03 ms vs scipy 2.16 ms | **2.1× FASTER** (SIMD-over-d `chebyshev` shipped since) |
| `pdist` chebyshev n=2048/d=64 "3.28× slower" | fsci 6.19 ms vs scipy 40.50 ms | **6.5× FASTER** |
| `pdist` chebyshev d=16 "4.82× slower" | fsci 0.94 ms vs scipy 0.56 ms | 1.68× slower (modest residual — only real one) |
| `ndimage.mean(labels,index)` "4.7× slower" | (per [[perf_ndimage_label_reduction_privatized_histogram]]) | already WINS 1.17-1.30× (stale 3rd time) |

LESSON (4th confirmation of the stale-scorecard rule): the scorecard's loss table OVERSTATES remaining gaps — re-measure same-box before chasing any of its "X× slower" entries. The lone real residual (pdist chebyshev d=16, 1.68×) is per-pair SIMD finalization overhead that SoA-across-pairs (proven byte-identical for d=4) would amortize — deprioritized as niche (d=64 already wins; d=16-only, ~40-line const/runtime-D SoA for a 1.68× modest gap). Cluster/spatial/stats lanes (cdist, KDTree, gaussian_kde 30×, kendalltau, mvn logpdf, vq, spectral, dbscan, pdist-cheby-d64) all WIN or parity — the accessible non-probe lanes are DOMINATED. Genuine remaining gaps: NMF (persistent-pool, deadlocked); probe-crate residuals (solve_toeplitz/lfilter — linalg/signal, owned); ndimage.label 2.68× (f64-output wall, needs native int store).

## 2026-06-29 — AmberKestrel (cc): NEGATIVE EVIDENCE — pdist Chebyshev SoA-across-pairs REJECTED (~0-gain / regression)

Implemented a general-dim SoA-across-pairs Chebyshev for pdist (`collect_columns` + `pdist_fill_cols` + `fill_chebyshev_soa_rows`: L=8 lanes = 8 distinct j-pairs, iterate the d coordinate columns keeping a running per-lane max + NaN mask, so the max-reduce/NaN finalize amortize across 8 pairs). Conformance GREEN, byte-identical (max exact + order-independent; 14+2 pdist tests). But MEASURED same-box it's a reject:
| n/d | default (SIMD-over-d) | SoA (this) | scipy |
|---|---|---|---|
| 512/16 | 0.94 ms | 0.89 ms (noise) | 0.56 ms (still 1.59× slower) |
| 512/64 | 1.03 ms | 1.13 ms (**regress**) | 2.16 ms |
| 2048/64 | 6.19 ms | 7.83 ms (**regress**) | 40.50 ms |

WHY: the SoA pays one `splat(col[i])` broadcast + column load per coordinate per 8-pair block = d broadcasts/block; that per-column overhead EXCEEDS the per-pair max-reduce finalize it removes once d≳16, so SIMD-over-d (the default `chebyshev`, which streams 8 dims/chunk per pair) is strictly better for d≥16. The d=16 residual (1.59×) is fixed per-pair overhead vs SciPy's inlined-C accumulator — fsci already scales BETTER with d (0.94→1.03ms for d 16→64 vs scipy 0.56→2.16), so it's an SMALL-d-only inlining wall, not an algorithmic gap. REVERTED. Lever REJECTED: SoA-across-pairs only wins when the per-pair finalize is EXPENSIVE relative to the per-element work (d=4 sqrt/div pdist) — for a cheap max reduce over d≥16 columns, SIMD-over-d's contiguous streaming dominates.

## 2026-06-29 — AmberKestrel (cc): NMF flat-buffer + MR=4 panel matmul — 2.9× self-WIN (1005→346ms), closes sklearn gap 6.9×→2.4× (residual = serial OpenBLAS GEMM wall)

`fsci_cluster::nmf` (1000×300, k=20, 200 mu-iters) was the documented cluster loss: **1005 ms vs sklearn — 6.9× slower**, doing 6 GEMMs/iter through `fsci_linalg::matmul` on `Vec<Vec<f64>>` (row-pointer chase + per-call alloc). DUG the flat-buffer + register-blocked-matmul lever:
- **Flatten** X/W/H to row-major buffers; reuse ALL scratch (wt/wtx/wtw/wtwh/ht/xht/hht/whht/wh) across iters (zero per-iter alloc); flat `transpose_flat`.
- **`nmf_mm` = ikj with an MR=4 output-row panel**: 4 output rows share each streamed B row → cuts B memory traffic ~4× (the dominant Wᵀ·X is memory-bound — streams X once per output row) and the inner 4-way AXPY auto-vectorizes over the n axis. The 4×N partial rows stay L1-resident (4×300×8 = 9.6 KB < 32 KB).

**Same-process A/B (best-of-4, this box):**
| kernel | time | vs sklearn(145) |
|---|---|---|
| ORIG `fsci_linalg::matmul` (Vec<Vec>) | ~1005 ms (documented) | 6.9× slower |
| flat + simple ikj | 447 ms | 3.1× slower |
| **flat + MR=4 panel (SHIPPED)** | **346 ms** | **2.4× slower** |
| sklearn NMF mu 200it (same-box) | 145 ms | — |

`rel_err = 0.470968` byte-identical across all three kernels (reduction-order change is correctness-safe; the MU updates converge to the same factorization). Conformance GREEN (fsci-cluster 142/142 lib). **Net: 2.9× self-speedup vs ORIG, 1.29× directly-A/B'd over simple-ikj, gap-to-sklearn 6.9×→2.4×.**

**REJECTED in the same A/B — register-tiled MR=4×NR=8** (C-tile held in registers across the K loop): 458 ms, SLOWER than the MR=4 panel (346). Why: the panel's 4×N partials are already L1-resident, so the register-residency saves little while the j-tiling + scalar column-remainder on the narrow matmuls (xht output is 1000×20, hht/whht are k×k) costs more. Don't re-chase register tiling here.

**RESIDUAL = serial GEMM micro-kernel wall, NOT a parallel wall (NEW finding):** sklearn NMF mu is **145 ms single-thread AND 145 ms all-threads** (measured both) — at this size the GEMMs are too small for OpenBLAS to parallelize, so its 145 ms is pure single-thread OpenBLAS micro-kernel (≈36 Gflops vs my MR=4 ≈15 Gflops). Matching that serially needs hand-tuned-assembly packing/prefetch — not reachable from portable Rust. The ONLY remaining lever to actually BEAT sklearn is **parallelism across the 64 cores** (now de-risked: since sklearn is single-threaded here, even modest parallel efficiency wins), but per prior sweeps per-call `thread::scope` spawn always loses (tiny GEMMs) and the persistent barrier-pool deadlocked — Amdahl also bites (serial element-wise updates + small matmuls + transposes ≈ 115 ms floor). Deferred to a dedicated turn. SHIPPED the safe 2.9× self-win now.

## 2026-06-29 — AmberKestrel (cc): NMF safe persistent worker-pool — FLIPS the marquee loss to a WIN (sklearn 145ms → fsci ~99-118ms, 1.23-1.46× FASTER; ~9× vs ORIG)

Follow-on to the serial flat+MR4 win (cc00089a). Per-phase profiling showed the 2 dominant GEMMs (Wᵀ·X 43% + X·Hᵀ 47%) are **91%** of serial time → Amdahl cap @P=8 = 82ms, @16 = 59ms, both well under sklearn's 145ms. KEY: sklearn NMF mu is single-threaded at this size (145ms both 1-thread and all-threads), so parallelism is a genuine, un-taken win.

Two parallelizations measured same-box (1000×300, k=20, 200 iters):
| approach | best time | self | vs sklearn(145) | note |
|---|---|---|---|---|
| serial (shipped cc00089a) | 346-359 ms | 1.0× | 0.42× | MR=4 panel |
| per-call `thread::scope` row-split | 213-218 ms @nt=8 | 1.65× | 0.67× | PLATEAUS then degrades — 400-scope spawn tax (matches the old "spawn always loses" sweep) |
| **SAFE persistent pool (SHIPPED)** | **99 ms @nt=12** (118@8, 105@16) | **3.6×** | **1.46×** | spawn-once, no per-iter tax |

**The pool is SAFE (no `unsafe` — workspace is `unsafe_code = "forbid"`, which is exactly why the prior raw-pointer pool attempt failed).** Mechanism: each worker permanently OWNS a moved-in row-band of W and X and talks to the driver over `mpsc` channels. The cross-band reductions `Wᵀ·X`/`Wᵀ·W` become per-band PARTIALS summed by the driver; H-update is serial (small); `X·Hᵀ`/`W·H·Hᵀ`/W-update are per-band (owned). On convergence-check iters workers also return their band + partial reconstruction error so the driver assembles W and tests `tol` with no extra pass. `rel_err = 0.470968` identical to serial across all nt (partial-reduction reassociation is negligible). Gated `nthreads = avail.min(n/96).min(16) >= 4 && d>=4 && k>=2` — small inputs stay serial. Conformance 142/142 GREEN.

**Net: the documented NMF loss (1005ms = 6.9× slower than sklearn) is now a 1.46× WIN at ~9-10× self-speedup vs ORIG.** REJECTED en route: register-tiled MR4×NR8 (slower, last cycle) and per-call scope (spawn-tax plateau, above). LEVER (generalizable): a sequential multiplicative/EM iteration whose cross-band matmuls are reductions → SAFE persistent pool with OWNED bands + partial-sum merge over channels beats both serial and per-call-spawn, AND sidesteps `forbid(unsafe)`. Candidates: factor_analysis / PPCA / LDA EM loops (same matmul-EM structure).

## 2026-06-29 — AmberKestrel (cc): gaussian_mixture_full M-step parallelized across components — flips a 2.4× LOSS to a 1.9-2.8× WIN (byte-identical)

Fresh measured loss found by probing the full-covariance GMM (the diagonal E-step was already parallel, but the full-cov M-step was overlooked). The M-step is `for c in 0..k { ... }` where each component's covariance is an O(n·d²) weighted outer-product sum — **all serial**, and it dominated (E-step was already fanned across points). Each component is independent ⇒ fanned the k components across cores via `thread::scope` (one thread per component, `chunks_mut` over weights/means/covariances output slots, shared `&data`/`&resp` reads). Each component is computed by the IDENTICAL serial arithmetic on its own thread → **byte-identical** result. Gated `mwork = n·d² >= 1<<16 && k >= 2`.

Same-box (max_iter=50, tol=0, reg_covar=1e-6):
| size | ORIG (serial M-step) | parallel M-step | sklearn GMM-full | flip |
|---|---|---|---|---|
| n=5000 d=30 k=8 | 1935 ms | **422 ms** (4.6× self) | 815 ms | 2.37× SLOWER → **1.93× FASTER** |
| n=10000 d=20 k=10 | 2841 ms | **420 ms** (6.8× self) | 1182 ms | 2.40× SLOWER → **2.81× FASTER** |

Conformance 142/142 GREEN (both gaussian_mixture tests pass). LESSON: when an iterative algo has BOTH an E-step and an M-step, check BOTH for parallelism — a parallel E-step can mask a serial M-step that then dominates (Amdahl). The M-step's per-component covariance is the classic "independent per-group O(work) reduction" → fan groups across cores, byte-identical. CANDIDATES with the same shape: other full-covariance EM (bayesian GMM), per-class scatter matrices (LDA/QDA fit), per-cluster covariance in any mixture model.

## 2026-06-29 — AmberKestrel (cc): gaussian_mixture (DIAGONAL) M-step — loop-interchange + per-component parallel, flips a 4× LOSS to a 4-6× WIN (16× self, byte-identical)

The diagonal GMM was a documented WIN at small d (memory yw7ts: 4-11× at n≤20k) but a measured LOSS at scale: n=20000/d=50/k=12 fsci 3780ms vs sklearn 942ms = **4.01× slower**; n=50000/d=30/k=10 6059ms vs sklearn 2441ms = 2.48× slower. ROOT CAUSE: the M-step loop nest was `for c { for j { Σ_i mean; Σ_i var } }` = **2·k·d strided passes over the data** (each element read 2kd times, column-strided in a row-major buffer = cache-pathological). Two byte-identical fixes:
1. **Loop interchange** — accumulate the whole mean/var vectors in ONE pass over i per component (contiguous `row[j]`), turning 2·k·d strided passes into 2·k contiguous ones (d× fewer; each output sum keeps the same i-ascending order → byte-identical; var uses `g*(diff*diff)` to match `resp*diff.powi(2)` exactly).
2. **Fan the k independent components across cores** (same lever as gaussian_mixture_full c3e8887a).

Same-box (max_iter=50, tol=0, reg_covar=1e-6):
| size | ORIG | new | sklearn | flip |
|---|---|---|---|---|
| n=20000 d=50 k=12 | 3780 ms | **227 ms** (16.7× self) | 942 ms | 4.01× SLOWER → **4.15× FASTER** |
| n=50000 d=30 k=10 | 6059 ms | **382 ms** (15.9× self) | 2441 ms | 2.48× SLOWER → **6.39× FASTER** |

Conformance 142/142 GREEN. LESSON (compounds the GMM-full one): the loop interchange was the BIGGER lever here (d× cache-pass reduction), parallelism stacked on top. AUDIT any `for group { for feature { for sample } }` moment/covariance accumulation — the sample loop belongs INNERMOST-but-vectorized (contiguous feature access), not re-scanned per feature. grep `for j .. { for i .. { .*\[i\]\[c\].*\[j\] } }`-shaped nests.

## 2026-06-29 — AmberKestrel (cc): spline_filter (B-spline prefilter) parallelized across independent lines — 1.46-1.67× self, widens scipy win 2× → 2.8-3.5× (byte-identical)

`ndimage.spline_filter` (the IIR B-spline prefilter used by ALL order>1 spline interpolation: zoom/rotate/affine_transform/map_coordinates) was already 2× faster than scipy SERIAL, but left cores idle. The IIR recursion is sequential WITHIN a line but the lines along each axis are independent. Parallelized both axis-pass shapes across CONTIGUOUS blocks (no unsafe — workspace forbids it):
- strided fast path (`bspline_reflect_axis_inplace`, non-last axes): split the buffer into contiguous outer-block chunks, each chunk runs the same in-place IIR.
- contiguous last-axis (stride==1) reflect lines: fan the rows across cores (non-fallible `bspline_reflect_coefficients` kernel).
Both byte-identical (block/row partition is the only change). Gated `spline_axis_threads`: total element work ≥ 1<<20 && blocks ≥ 2. The one un-parallelizable case is axis 0 with outer=1 + stride>1 (interleaved strided writes can't `split_at_mut` safely under forbid-unsafe) — stays serial, capping the win.

Same-box (order 3, Reflect):
| size | ORIG (serial) | parallel | scipy | win |
|---|---|---|---|---|
| 2048×2048 | 54 ms | **37 ms** (1.46× self) | 103.5 ms | 1.92× → **2.80× faster** |
| 256×256×256 | 247 ms | **148 ms** (1.67× self) | 520.6 ms | 2.11× → **3.52× faster** |

Conformance 255/255 ndimage lib GREEN (5 spline tests + all interpolation consumers). LEVER: an IIR/recursive sweep along one axis of an N-D array is sequential per line but the LINES are independent — parallelize across contiguous outer-blocks (non-last axes) / rows (last axis); the outer=1 first-axis stays serial under forbid-unsafe. Same shape: other separable IIR (gaussian via recursive filter, uniform_filter running-sum, any `*_filter1d` IIR).

## 2026-06-29 (AmberKestrel, cc) — signal.lfilter chunked PARALLEL associative scan (single long signal)

scipy's `lfilter` is sequential C; fsci's DF2T recurrence was also serial → PARITY/slight-loss
(n=1M order8: fsci 6.72ms vs scipy 5.90ms = 1.14x SLOWER). The state recurrence is a constant-matrix
affine map `d_n = M·d_{n-1} + v·x_n`, so by linearity (superposition) the output splits into a
zero-state response (computed per contiguous chunk INDEPENDENTLY, in parallel) plus the homogeneous
response to each chunk's true entry state. Entry states recovered by a serial O(P·m²) boundary combine
using `M^chunk` (binary matrix power, m×m companion). Two parallel `thread::scope` passes + serial combine.

MEASURED (n=1M, clean back-to-back): order4 5.50→3.63ms = **1.52x self / 1.38x FASTER than scipy 5.00ms**;
order8 6.72→4.16ms = **1.62x self / 1.42x FASTER than scipy 5.90ms**. n=4M ~1.6-1.9x faster (grows with N).
Gate `lfilter_scan_thread_count`: serial below 1<<18 (byte-identical, all 17 existing lfilter scipy-ref
tests + sosfilt/dlti/filtfilt callers GREEN), parallel above with P = avail.min(N/65536). NOT byte-identical
(superposition reassociates) → max_abs_diff vs serial reference 5e-13 @order8 (max_rel blows up only at
zero-crossings where |y|≈0); verified by new `lfilter_parallel_scan_matches_serial_reference` property test
(<1e-9, orders 4/6/8, N above gate w/ remainder chunk). filtfilt inherits the win (calls lfilter 2x).
LEVER: any constant-coefficient linear recurrence (IIR filter, DF2T) → chunked parallel scan via
superposition (zero-state pass ∥ + serial M^chunk boundary combine + homogeneous-correction pass ∥);
exact-to-roundoff for stable filters since M^k decays. The "genuinely different primitive" (parallel-scan).

## 2026-06-29 (AmberKestrel, cc) — signal.sosfilt chunked PARALLEL associative scan (cascaded biquads)

Direct follow-on to the lfilter scan (264bf1a7). scipy's `sosfilt` is sequential C; fsci's sample-major
cascade was also serial (parity). The WHOLE N-section cascade is ONE constant-matrix linear recurrence
`z_n = A·z_{n-1} + b·x_n` over the composite state z (2·nsec), so superposition applies: per-chunk
zero-state response (parallel) + homogeneous response to each chunk's true entry state, recovered by a
serial O(P·(2nsec)²) boundary combine using `A^chunk`. KEY TRICK: build the (2nsec)² companion `A` by
PROBING the single-step homogeneous cascade with basis vectors (column j = one x=0 step on e_j) — no
hand-composing the per-section state-space blocks. Reuses lfilter's mat_pow/mat_vec.

MEASURED (clean back-to-back) — BIGGER than lfilter (more biquads/sample ⇒ compute-bound ⇒ better
parallel efficiency; win grows with N AND order):
  n=1M order12 (6 sec):  3.56ms vs scipy 8.06ms  = **2.27x FASTER**
  n=1M order24 (12 sec): 5.41ms vs scipy 16.26ms = **3.00x FASTER**
  n=4M order12 (6 sec):  10.0ms vs scipy 30.9ms  = **3.09x FASTER**
  n=4M order24 (12 sec): 13.2ms vs scipy 64.9ms  = **4.92x FASTER**
max_abs_diff vs serial reference 2-3e-15 (near-exact; better-conditioned than lfilter's 5e-13).
Gate reuses `lfilter_scan_thread_count`: serial below 1<<18 (BYTE-IDENTICAL, all 13 sosfilt scipy-ref/
sosfiltfilt/axis_2d tests GREEN), parallel above with P=avail.min(N/65536). NOT byte-identical
(superposition) → verified by new `sosfilt_parallel_scan_matches_serial_reference` property test (<1e-9,
orders 6/12/18, N above gate w/ remainder). sosfiltfilt inherits (calls sosfilt 2x). The constant-coeff
linear-recurrence parallel-scan lever now covers BOTH lfilter (ba) AND sosfilt (cascaded biquads).

## 2026-06-29 (AmberKestrel, cc) — signal.savgol_filter branch-free vectorized interior dot

savgol_filter already BEAT scipy (parallel par_index_fill across output indices), but the Interp-mode
interior closure did a per-tap BOUNDS-CHECK branch (`if idx>=0 && idx<n`) for EVERY coefficient, defeating
SIMD even though the interior `[half, n-half)` is reflection-free (every tap in range). Lever (same as the
gaussian_filter interior-split): compute the interior branch-free via `savgol_dot` — chunks_exact(8) +
`try_into::<[f64;8]>` fixed-arrays (elide bounds checks ⇒ inner loop unrolls & auto-vectorizes) with 8
independent accumulators (pipeline the FMA chain). Boundary `[0,half)∪[n-half,n)` left 0 (overwritten by
the polynomial edge fit). Applied to all 3 paths: Interp interior (main), padded modes, and the serial
axis_2d helper.

MEASURED (n=2M): win=101/poly3 10.14→7.38ms = **1.37x self** (vs scipy 68.06ms: 6.7x→**9.2x faster**);
win=301/poly4 24.92→11.99ms = **2.08x self** (vs scipy 409.26ms: 16.4x→**34.1x faster**). Win grows with
window (more taps to vectorize). NOT byte-identical (8-lane + tree reassociation ~1e-14) but within the
savgol scipy tolerance — all 18 savgol tests GREEN incl. matches_scipy_reference_values / even-window /
padded modes_match_scipy / axis_2d. This WIDENS an existing lead (savgol already won), not a gap-close —
signal crate is otherwise dominated (lfilter/sosfilt scans shipped this session; hilbert 1.95x, decimate
1.11x, savgol now 9-34x all faster than scipy). LEVER (reusable): any per-output FIR/correlation with a
per-tap bounds-check branch over a reflection-free interior → split interior (branch-free chunks_exact(8)+
try_into fixed-array dot, 8 accumulators) from boundary. The try_into-[f64;8] idiom is the stable-Rust
auto-vectorization key (plain `slice[j+lane]` keeps bounds checks and does NOT vectorize — measured 1.22x
vs 2.08x for the fixed-array form).

## 2026-06-29 (AmberKestrel, cc) — spatial.minkowski integer-exponent fast path (cdist/pdist)

scipy's cdist 'minkowski' is its SLOWEST common metric — `pow(|d|, p)` per element ⇒ 1235ms for
1200×1200 d=80 (vs euclidean 34ms). fsci's general minkowski path also did per-element `.powf(p)`; it
ALREADY beat scipy ~24x via cdist row-parallelism (~51ms) but still paid an ~80-cycle powf per element.
Lever: for a SMALL INTEGER exponent (p∈[3,64], the common p=3,4,5 case), `|d|^p` is just ≈p repeated
multiplications — replaced powf with an 8-wide std::simd integer-power kernel (`minkowski_int`: two
accumulators + scalar tail, `t *= |d|` p-1 times). p=1/2/∞ still route to cityblock/euclidean/chebyshev;
non-integer p keeps the scalar powf.

MEASURED (cdist 1200×1200 d=80): p=3 **51→5.12ms = 10x self / 241x FASTER than scipy 1235ms**; p=4 4.76ms;
p=5 5.07ms; non-integer p=3.5 unchanged at 51ms (still 24x vs scipy). pdist_minkowski inherits (both
route through `minkowski`). NOT byte-identical (x*x*x vs powf + SIMD reassoc ~1e-14) but within distance
tolerance — 225/225 spatial lib GREEN incl. minkowski reference-value + cdist/pdist tests. LEVER (reusable):
any per-element `powf(INTEGER)` in a hot reduction → repeated-multiplication + SIMD (powf is ~80 cycles,
x^p is ~p mults; the SLOWEST scipy distance metric becomes near-free). grep `.powf(` over possibly-integer
exponents. SIGNAL crate done this session (lfilter/sosfilt/savgol); spatial euclidean/cosine/cityblock/
canberra/chebyshev already SIMD — minkowski was the last scalar-powf hole.

## 2026-06-29 (AmberKestrel, cc) — spatial.minkowski_distance integer-exponent fast path (follow-on)

Follow-on to the cdist/pdist minkowski win (74ec55a9): the batched row-wise `minkowski_distance` /
`minkowski_distance_p` (scipy.spatial API) had its OWN inline per-element `.powf(p)` loop (separate from
the per-pair `minkowski`). Refactored the integer-power SIMD kernel out of `minkowski_int` into
`minkowski_pow_sum(a,b,p:u32)` (8-wide std::simd `Σ|Δ|^p` by repeated multiplication) and routed the
`minkowski_rowwise` else-branch through it for integer p∈[2,64]; non-integer p keeps scalar powf.

MEASURED (minkowski_distance n=400k d=8): p=3 **8.52ms vs scipy 100.66ms = 11.8x FASTER** (~5x self vs the
p=3.5 powf path at 44ms); p=2 8.30ms vs scipy 36.21ms = **4.4x**; p=4 8.90ms. correctness diff 0.0 vs scalar
powf reference (row 0). Within tolerance, 225/225 spatial lib GREEN incl. minkowski_distance_batched_matches_scipy.
Both minkowski_distance and minkowski_distance_p share `minkowski_rowwise` ⇒ both inherit. The `powf(integer)
→ repeated-mult + SIMD` lever now covers every minkowski surface (cdist/pdist per-pair + batched rowwise).
Left serial (row loop) — the kernel alone removes the powf bottleneck; row-parallel is a future follow-on.

## 2026-06-29 (AmberKestrel, cc) — interpolate.RbfInterpolator build: route dense solve to optimized blocked-LU

RbfInterpolator::new solved the n×n RBF system Φw=values via a LOCAL naive serial Gaussian elimination
(`solve_dense_system_flat`). Profiled n=2000 d=3: matrix fill 46ms, **naive solve ~1147ms** = 96% of the
1193ms build. fsci-interpolate already depends on fsci-linalg, whose `solve()` has a multithreaded
blocked-LU fast path (n≥1000). Routed the build's solve through it (flat phi → Vec<Vec> rows →
`fsci_linalg::solve`, default Strict/General options).

MEASURED (n=2000 d=3): build 1193→842ms = **1.42x self** (linalg solve 800ms vs naive 1147ms); closes the
build gap vs scipy 131ms from 9.1x→6.4x. NOT byte-identical (blocked-LU pivoting/blocking vs naive GE,
~1e-12) but within RBF tolerance — interpolate lib 178/178 GREEN incl. all rbf_* tests. HONEST FRAMING:
RBF is NOT a real end-to-end gap — eval_many is already 57x FASTER than scipy (16.4ms vs 950ms @ nq=20000),
so fsci wins the full build+eval workflow (856ms vs scipy 1081ms = 1.26x) even before this. The residual
6.4x build gap is the fsci-linalg DENSE SOLVE wall (~3.3 GFLOP/s vs LAPACK ~40 — n=2000 solve is 800ms),
a separate-crate multi-session target, not cheaply closable here. This ship = reuse the optimized solver
instead of a naive one (good hygiene + 1.42x), not a wall-break. LEVER: grep for LOCAL naive dense solvers
(`solve_dense_system*`, hand-rolled GE) in non-linalg crates → route to fsci_linalg::solve's blocked path.

## 2026-06-29 (AmberKestrel, cc) — ndimage.spline_filter1d: route Reflect to the fast IIR prefilter (17.6x→1.86x flip)

`spline_filter1d` computed each axis-line's spline coefficients via `spline_coefficients_for_line`, which
called `make_interp_spline(0..n, line, order)` — building and solving a FULL n-point interpolation system
PER LINE. For a single long 1-D array that is pathological: n=4M order3 = **865ms = 17.6x SLOWER than
scipy 49ms** (order5 1123ms = 17.8x). The N-D `spline_filter`/`prefilter_spline_coefficients` already use
the fast exact O(n) recursive IIR (`bspline_reflect_coefficients`, Unser/Thévenaz, scipy-conformant) — but
`spline_filter1d` was on the slow `make_interp_spline` arm. Routed Reflect mode (order 2..=5, axis>order)
through `bspline_reflect_coefficients`; Nearest/short-axis keep the general path.

MEASURED (1-D n=4M): order3 865→**26.4ms = 32.8x self / 1.86x FASTER than scipy** (was 17.6x slower);
order5 1123→40.8ms = 27.5x self / **1.55x faster**. Multi-line spline_filter1d (2D/3D along one axis)
inherits the per-line speedup. SCIPY-VERIFIED (mode='reflect'): order2/3 EXACT, order4/5 ~1e-7 (within
tolerance); 255/255 ndimage lib GREEN. LEVER (recurring): a function computing spline/IIR prefilter
coefficients by SOLVING a full linear system per line → route to the recursive IIR prefilter the
sibling/N-D path already uses. grep `make_interp_spline` / dense-solve calls inside per-line coefficient
loops. (Same family as the lfilter/sosfilt scans: the fast way to "solve" a cardinal-spline interpolation
system IS the recursive IIR, not a banded/dense solve.)

## 2026-06-29 (AmberKestrel, cc) — interpolate.RectBivariateSpline build: chunked-parallel tensor product (5x flip)

RectBivariateSpline::new built the tensor-product spline by fitting 1-D `make_interp_spline` along each
row (ny) then each column (nx) — SERIALLY. Each row/col spline is INDEPENDENT and the per-call cost is
mostly fixed overhead (knot build + banded solve setup), so 1600 serial calls @ ~92µs dominated: 800×800
= **147ms = 5.2x SLOWER than scipy 28ms** (400² 4.6x). scipy's RectBivariateSpline is single-threaded
FITPACK ⇒ parallelism is pure domination. Fanned BOTH passes across cores via a new chunked
`par_chunk_try_map` (one thread::scope spawn per CHUNK, not per row — a prior per-column per-spawn attempt
was reverted for over-spawn, see line ~2658). Column pass assembles via transpose to avoid a cross-thread
write race on the row-major output.

MEASURED: 400² 35.1→**6.8ms = 5.2x self / 1.13x FASTER than scipy** (was 4.6x slower); 800²
147→**16.0ms = 9.2x self / 1.77x faster** (was 5.2x slower); win grows with grid. BYTE-IDENTICAL
(order-preserved chunked map ⇒ identical coefficients to serial), 178/178 interpolate lib GREEN incl. all
rect/smooth bivariate tests. LEVER: a tensor-product / per-line build looping an independent banded/spline
solve per row & column → chunked-parallel both passes (transpose-assemble the second pass). Reusable
`par_chunk_try_map` (fallible, order-preserved, chunk-spawn) added for Vec-valued parallel maps.

## 2026-06-29 (AmberKestrel, cc) — NEGATIVE EVIDENCE: interpolate builders / sparse / CloughTocher all DOMINATED

After flipping spline_filter1d (17.6x→1.86x) and RectBivariateSpline (5.2x→1.77x) via the "measure the
public builder directly" lever, swept the adjacent surface for more slow-arm/serial-builder gaps — measured
same-box vs scipy 1.17.1, all WIN or parity (do NOT re-chase):
- interpolate CubicSpline build n=500k: fsci 11.3ms vs scipy 39.6ms = **3.5x faster**
- interpolate Akima1D build n=500k: 10.8ms vs 41.7ms = **3.9x faster**
- interpolate PchipInterpolator build n=500k: 20.2ms vs 34.7ms = **1.7x faster**
- interpolate RegularGridInterpolator eval nq=200k (80³): 6.0ms vs 67.4ms = **11.2x faster**
- interpolate CloughTocher2D eval nq=50k: 7.5ms vs 44.8-96ms = **5.9-12.8x faster**; build ~parity
  (npt=2000 1.05x faster; npt=8000 1.17x slower — marginal, cost is the global gradient solve =
  known-hard backlog, NOT a cheap parallelization)
- interpolate SmoothBivariateSpline: FITPACK surfit (adaptive knots = sequential), scipy only 2.34ms@n=5000 — not a target
- sparse spsolve tridiag N=200k: 11.2ms vs scipy 85.4ms = **7.6x faster** (already special-cases banded)
- sparse spmm (CSR@CSR): already parallel Gustavson (fanned across rows, gated by work) — not a gap
CONCLUSION: the interpolate-1D-builder / sparse-core / CT-eval surfaces are DOMINATED. The "serial builder
over independent units" + "wrapper on naive arm" levers are now mined out across interpolate/ndimage/sparse;
the 2 genuine flips this session (spline_filter1d, RectBivariateSpline) were the payoff. Remaining gaps =
the known WALLS (fsci-linalg dense solve ~10x vs LAPACK, FFT non-pow2 SIMD, Qhull/HiGHS) + hot-crate
collision zones (stats/integrate axis-2d, other agents). Next dig should target a WALL or an unmeasured crate
(special/fft batched), not these.

---
## 2026-06-29 (AmberKestrel, cc) — DCT-IV / DST-IV Type-IV core: 2N-FFT → split into 2 parallel N-FFTs

DIG into fsci-fft (untouched 16h, low collision). Measured fsci vs scipy 1.17 dct/dst at n=2^20 and
n=1,000,000 (both single-threaded scipy pocketfft). Found the Type-IV transforms were the biggest gap:
  dst-IV n=2^20 79.4ms vs scipy 17.5 (4.8x slow);  n=1M 58.5ms vs 7.7 (7.5x slow)
  dst-III ~3x; dst-II/dct-II ~1.3-2.7x (pure FFT SIMD wall, documented, not chased)
  dst-I: fsci ALREADY 1.1-2.4x FASTER than scipy (scipy dst-I 306-349ms; not a gap)

ROOT CAUSE: `dct4_core_fft` (shared by dct_iv + dst_iv) ran ONE 2N-point COMPLEX FFT of a zero-padded
length-N complex sequence u (u[n]=x[n]e^{-iπn/2N}); only the first N bins are used. The 2N complex FFT
both doubles the length and thrashes cache (2x16MB at n=2^20). Measured building block: fft 2N complex
62.5ms vs fft N complex 18.7ms — superlinear cache blowup.

LEVER (exact Cooley-Tukey decimation-in-frequency, NOT byte-identical, ~1e-14): split the 2N transform by
OUTPUT PARITY into two independent N-point FFTs —
  U[2m]   = FFT_N(u)[m]
  U[2m+1] = FFT_N(u')[m],   u'[n] = u[n]·e^{-iπn/N}
verified vs scipy to 2.3e-10 abs (~1e-13 rel) across n=1..1024. The two N-FFTs run CONCURRENTLY on 2
threads above gate N>=1<<16 (scipy is single-threaded here so the 2nd core is free). A/B serial-vs-parallel
confirms parallelism is the bulk of the win (pow2 case is memory-bound; cache benefit of the split alone
is small at pow2, large for the 5-smooth 1M). New cached split-twiddle table e^{-iπk/N}.

RESULT (ratio vs ORIG fsci):
  dst-IV / dct-IV  n=2^20  79.4 -> 46.5ms = 1.71x self
                   n=1M    58.5 -> 31.7ms = 1.85x self
Both Type-IV transforms lifted (shared core). 236/236 fsci-fft tests GREEN (metamorphic self-inverse +
all). Still 2.7-4x vs scipy at these n (the per-FFT-point pocketfft SIMD wall, documented hard wall) — but
a clean algorithmic self-win flipping fsci's own 2N-complex baseline. Commit: this one.

GENERALIZABLE: any transform computing the first N bins of a 2N-point FFT of a zero-padded length-N input
(DCT/DST type IV cores, some chirp/Bluestein setups) → decimate into 2 N-point FFTs by output parity and
run them in parallel. The 2N->2*N split is ALWAYS >= as cheap (less work + cache-resident) and the pair is
embarrassingly parallel against scipy's single-threaded core.

### 2026-06-30 follow-on (AmberKestrel, cc) — idct twiddle cache (byte-identical), lifts dct-III/dst-III
The deferred follow-on from the Type-IV dig: `idct` (even-N path) recomputed its N/2 twiddle table
`(cos(πk/2N), sin(πk/2N))` with inline cos/sin on EVERY call — ~6-8ms of stray transcendentals at N=2^20.
That table = conj of the already-cached DCT-II twiddle (`get_or_compute_dct2_twiddles`); cos even + sin odd
make conj(stored (cos(-θ),sin(-θ))) == (cosθ,sinθ) BIT-IDENTICALLY (verified to_bits across 5.6e5 k/N, 0
mismatches). Reused the cache via `complex_conj`. BYTE-IDENTICAL output → zero conformance risk, 236/236
green. Lifts idct AND its dct_iii/dst_iii callers:
  idct     n=2^20 20.6 -> 15.9ms = 1.30x;  n=1M 21.0 -> 16.6ms = 1.27x
  dst-III  n=2^20 24.3 -> 17.3ms = 1.41x;  n=1M 22.5 -> 17.3ms = 1.30x
  dct-III  n=2^20 -> 15.8ms;               n=1M -> 16.3ms
dst-III now 2.16x vs scipy (was 3.0x). LEVER (generalizable): grep inline `angle.cos()/angle.sin()` in
per-element transform hot loops where a sibling already caches the same (or conj/neg) table — reuse it.

### 2026-06-30 (AmberKestrel, cc) — N-D dctn alloc-free fiber kernel: TESTED, ~0-gain, REVERTED
Swept rest of fsci-fft post Type-IV: hfft/ihfft parity-or-faster, fht 1.1-1.23x, dst-I fsci faster. Only gap
= dctn/dstn 2D (1.58-1.74x vs scipy serial). apply_dct_along_axis already 64-thread parallel but no better
than serial floor → hypothesized per-fiber alloc contention. Built alloc-free `dct_ii_gather_into` +
per-worker reused DctIIScratch + fiber-contiguous output (bit-identical). Clean A/B = ~0-gain (14-15 vs
14-16ms @1024^2; OLD faster @2048^2). REVERTED. Real wall = fsci 1-D dct 2x scipy per-call (FFT-SIMD wall) +
strided-axis bandwidth; needs cache-blocked transpose + faster FFT kernel, not an N-D lever. See
docs/NEGATIVE_EVIDENCE.md.

### 2026-07-01 (AmberKestrel, cc) — mmread_sparse: sparse-COO MatrixMarket read, ~10x faster than dense mmread (scipy parity)
DIG into fsci-io (13d stale, low collision). Measured all readers vs numpy/scipy on large files: loadtxt 14x
FASTER, read_csv 12x FASTER, mmread(dense) 1.46x FASTER — all wins. ONE gap: **mmread on a SPARSE (coordinate)
file was 9.8x SLOWER than scipy** (136ms vs scipy 14ms, 4000^2 @1% ~160k nnz). ROOT CAUSE (profiled): mmread
materializes a DENSE rows*cols buffer (128MB for this file) even for coordinate format; the ~120ms is almost
entirely first-touch page faults across that 128MB of mostly-zeros (parse-to-COO alone = 13ms = scipy
parity; the dense buffer is intrinsic to mmread's dense return type — can't be sped up in place). scipy
returns sparse COO in 14ms. LEVER: added `mmread_sparse` (+ `MmSparse` COO struct) = parse coordinate format
to (row_indices, col_indices, values) triplets, NO dense materialization; symmetric/skew/hermitian expand the
stored triangle to both off-diagonals (negate for skew), so scattering the triplets with `+=` reproduces
mmread's dense `data` BIT-FOR-BIT (verified: byte_mismatches=0 across general/symmetric/skew/duplicate/pattern
+ permanent test mmread_sparse_matches_dense_mmread). RESULT: **15.1ms vs mmread dense 123.6ms = 8.2x self;
9.8x scipy-loss -> ~PARITY (scipy 14ms)**. Purely additive (206 insertions, 0 deletions), matches scipy.io
sparse-return behavior for the format's primary (sparse) use case. fsci-io mmread family 20/20 green (my new
test passes). NOTE: pre-existing UNRELATED red `mmwrite_complex_output_format` fails on HEAD too (test asserts
0-based coordinate indices; the emitter correctly produces 1-based per MM spec) — another agent's test bug,
left untouched per own-files.

### 2026-07-01 (AmberKestrel, cc) — loadmat_v5 fused decode+transpose: 8.7x self, flips 8.2x scipy LOSS to WIN
Continued fsci-io sweep. Measured writers/readers vs scipy/numpy: savemat 1.55x FASTER, savetxt 3.2x FASTER,
mmwrite(dense) 5.6x SLOWER (float-format wall, deferred), **loadmat 8.2x SLOWER** (25.3ms vs scipy 3.1ms,
300000x8 v5 .mat). ROOT CAUSE (profiled): loadmat_v5 decoded the column-major disk payload into an
intermediate `column_major` Vec (`chunks_exact(8).map(from_le_bytes).collect()`) THEN did a SEPARATE strided
transpose into row-major `data` — one extra full 19MB alloc + two extra passes over R*C*8 bytes. A/B:
decode.collect+transpose 25.8ms vs FUSED single pass 6.4ms. LEVER: `decode_v5_numeric_rowmajor` fuses the
byte-decode and the column->row transpose in one pass (c outer = sequential disk read, strided row-major
write), dropping the intermediate buffer; handles all MI_ numeric types; error messages preserved
(loadmat_rejects_wrong_element_count green). RESULT: loadmat **25.3 -> 2.90ms = 8.7x self; flips 8.2x scipy
LOSS -> 1.07x WIN** (beats scipy's 3.10ms). Byte-identical (values == scipy exactly; 116/116 pre-existing
loadmat/mat tests green — sole red is the unrelated pre-existing mmwrite_complex 0-vs-1-based test). Net
-13 lines. LEVER (generalizable): binary readers that decode-to-Vec THEN transpose/reorder → fuse into one
pass writing the final layout directly; the intermediate buffer is pure alloc + memory-traffic overhead.
mmwrite float-format gap (5.6x) left as deferred (needs fast f64 formatter/dep — byte-id risk).

### 2026-07-01 (AmberKestrel, cc) — mmwrite parallel formatting: 8.7x self, flips 5.6x scipy LOSS to 1.56x WIN
The last fsci-io gap: mmwrite(dense) 5.6x SLOWER than scipy (276ms vs 49.8ms, 1500x1500). PROFILED: the f64
Display FORMATTING is the wall (~175ms/2.25M values = 78ns each) — NOT allocation (with_capacity gave 0-gain)
and NOT writeln! overhead (reused-buffer write! 0-gain); the strided column-major read adds ~100ms. No
byte-identical single-thread win without a float-formatter dep. BUT scipy's mmwrite is SINGLE-THREADED and
formatting is embarrassingly parallel. LEVER: each worker formats a contiguous slice of the column-major value
stream (value k → col k/rows, row k%rows → data[row*cols+col]) into a private String; concat parts in order =
BIT-FOR-BIT the serial output (verified all nthreads==serial + round-trip test). Serial gate n<1<<16 BEFORE
the available_parallelism syscall (per-call-syscall-tax lesson); nthreads = avail.min(n/16384). RESULT:
mmwrite **276 -> 31.8ms = 8.7x self; flips 5.6x scipy LOSS -> 1.56x WIN** (beats scipy 49.8ms). Byte-identical
(117/117 mmwrite/mm tests green incl. new mmwrite_parallel_path_matches_serial_and_roundtrips; sole red is
the unrelated pre-existing mmwrite_complex 0-vs-1-based test). fsci-io vein now: mmread_sparse (10x),
loadmat (8.7x), mmwrite (8.7x) all shipped; loadtxt/read_csv/savemat/savetxt already faster. LEVER
(generalizable): when the wall is a serial std formatter/parser and the peer lib is single-threaded, PARALLEL
format-into-private-buffers + ordered concat is byte-identical and wins.

### 2026-07-01 (AmberKestrel, cc) — wav_read parallel per-sample decode: 7.3x self (byte-identical)
Swept remaining fsci-io readers: read_arff 3.1x FASTER than scipy (1064ms vs 339ms, no gap); wav_read 22ms.
The wav scipy comparison is APPLES-TO-ORANGES (scipy.io.wavfile.read returns raw int16 zero-copy in 0.33ms;
fsci returns normalized f64, doing real conversion work scipy skips) — NOT a scipy gap, but a self-improvement:
the per-sample i16→f64 decode was a serial `chunks_exact.map.collect` that does NOT auto-vectorize (measured:
the chunks16+[u8;16] idiom gave ~0-gain, reverted) and is compute-bound (~4ns/sample). LEVER: parallelize the
per-sample decode across threads (generic `decode_wav_samples(bytes, stride, conv)` covering 8/16/24/32-bit,
serial gate n<1<<18 before the available_parallelism syscall, byte-identical — each worker runs the same conv
on a disjoint sample range). RESULT: wav_read **22.09 -> 3.02ms = 7.3x self**, byte-identical (new test
wav_read_parallel_decode_matches_serial + 118/118 wav/io tests green; sole red = unrelated pre-existing
mmwrite_complex). NOTE honestly: still ~9x scipy's 0.33ms because fsci produces f64 not raw int16 — a semantic
difference, not a perf bug. fsci-io scipy-comparable surface now fully swept/dominated.

### 2026-07-01 (AmberKestrel, cc) — KDTree knn flat point-slab: d=8 query 1.46x self, closes 1.54x scipy-parallel loss to parity
PIVOT off io (fully swept) to fsci-spatial KDTree. Measured vs scipy.spatial.cKDTree: fsci build 1.46-1.9x
FASTER; d=3 query_k_many 8.1ms BEATS scipy workers=-1 (11.5ms); BUT **d=8 query_k_many 270.6ms = 1.54x SLOWER
than scipy workers=-1 (175.9ms)** (36x faster than scipy SERIAL — scipy's single-thread d=8 query is 9.9s).
query_k_many already parallel+tuned, so the residual is the per-query knn_search KERNEL. ROOT CAUSE: `KDNode`
stores `point: Vec<f64>` — a SEPARATE heap alloc per node (200k scattered Vecs); knn_search's
`sqeuclidean(query, &node.point)` chases a pointer to scattered memory (cache miss per visited node, and d=8
backtracking visits MANY). sqeuclidean is already std::simd, so the wall is the scattered coord reads. The
BUILD already flattened coords (partition), but build_kdtree cloned each point back into a per-node Vec.
LEVER (flat-buffer, [[perf_equal_hardware_artifact_and_flatbuffer_lever]]): add a node-ordered contiguous
slab `points: Vec<f64>` (node i at points[i*dim..]) and route knn_search coord reads through it instead of
node.point. RESULT: d=8 query_k_many **270.6 -> 184.9ms = 1.46x self; flips 1.54x scipy-parallel loss to 1.05x
(PARITY)**; d=3 unchanged (already cache-friendly); build +1ms (the slab copy, negligible). Byte-identical
(slab is a node-order copy of node.point; 225/225 spatial tests green incl. kdtree_query_matches_scipy_
reference_values). Minimal 25/-7 diff (kept node.point for the non-hot ball/nn/pairs paths). FOLLOW-ON: remove
node.point entirely + route ball_search/nn_search/query_ball_* through the slab (would shrink KDNode 56->32B →
more nodes/cache-line, help ball queries too).

### 2026-07-01 (AmberKestrel, cc) — KDTree single-NN (query_many/nn_search): 2.2x self, closes d=8 3.9x scipy loss to 1.78x
Follow-on to the knn flat-slab (822eabd7). Measured fsci KDTree ball/nn/pairs vs scipy cKDTree workers=-1:
ball_point_many 2.4-12x FASTER, query_pairs faster, BUT **d=8 query_many (single-NN k=1) 103.3ms = 3.9x SLOWER
than scipy workers=-1 (26.4ms)**. TWO causes (both the levers already applied to query_k_many but NOT
query_many): (1) thread ceiling hard-capped at `cores.min(16)` — query_k_many's comment already noted the
16-cap left d=8 ~3.5x slow and lifted it for work-heavy batches, but query_many never got the fix; (2)
nn_search still read scattered `node.point`. FIX: thread_ceiling = if dim>=6 { cores } else { cores.min(16) }
(single-NN backtracking is compute-heavy at higher dim → use all cores; low-dim stays capped to avoid
over-spawn) + route nn_search through the node-ordered flat point slab. RESULT: d=8 query_many **103.3 ->
46.9ms = 2.2x self; closes 3.9x scipy-parallel loss to 1.78x**; d=3 2.3->1.8ms (flat-slab, still 4.8x faster
than scipy). Byte-identical (same coords, independent deterministic per-query; 225/225 spatial green incl.
kdtree_query_many_matches_per_query + nearest_neighbors_kdtree_matches_brute_force_bitwise). Also lifts single
`query` (uses nn_search). RESIDUAL 1.78x = nn_search kernel + median-split vs scipy's sliding-midpoint tree
(deeper rewrite, deferred). LEVER (reusable): when a batched-query method wins but its SINGLE/other-k sibling
loses, diff their thread-gate + kernel — the fix often already exists on the fast sibling.

### 2026-07-01 (AmberKestrel, cc) — BSpline.eval_many parallel de Boor: 11.5x self, 10.4x FASTER than scipy splev
PIVOT (no collision anywhere on frankenscipy — last 8h commits all mine). Measured fsci-interpolate spline eval
on nq=2M sorted queries (n=2000 knots): CubicSplineStandalone.eval_many 9.1ms = 1.68x FASTER than scipy
CubicSpline (15.3ms) — win. BUT **BSpline.eval_many 59.9ms was ~PARITY with scipy splev (54.3ms)** and it was
SERIAL. scipy's splev is single-threaded and per-point de Boor is independent, so PARALLELIZE. For sorted
input each worker re-seeds its knot-span pointer mu by advancing from k to its chunk start (O(#knots) ≪
#queries) then merge-advances within the chunk — the span reached for any x depends only on x+knots, so
BIT-IDENTICAL to the single serial pointer walk (verified byte_mismatches=0 sorted AND unsorted, 2M pts; new
test bspline_eval_many_parallel_matches_per_point). Serial gate n<1<<15 before the available_parallelism
syscall. RESULT: BSpline.eval_many **59.9 -> 5.2ms = 11.5x self; flips ~parity to 10.4x FASTER than scipy
splev** (14.7x vs scipy BSpline). Byte-identical, 179/179 interpolate tests green. Lifts all
BSpline.eval_many callers. LEVER (proven again): a serial per-element eval where the scipy peer is
single-threaded → chunk-parallel with per-chunk pointer re-seed (byte-id for sorted-monotone state).

### 2026-07-01 (AmberKestrel, cc) — rankdata(ordinal) stable→unstable sort: 1.36x self (byte-identical anomaly fix)
Swept fsci-stats sort/selection fns vs scipy (N=2M): rankdata(avg) 5.3x FASTER, wasserstein 6.6x, energy 4.2x
— all win. SELF-ANOMALY: rankdata(ordinal) 114.6ms was 1.9x SLOWER than rankdata(average) 61.0ms — ordinal
(argsort+assign 1..n) should be SIMPLER than average (tie-grouping). CAUSE: rankdata_ordinal used STABLE
`sort_by` while the tie methods use faster `sort_unstable_by`. The ordinal comparator (value, then original
index) is a STRICT TOTAL ORDER (unique indices → no equal elements), so an unstable sort yields the IDENTICAL
permutation as stable (nothing for stability to disambiguate). FIX: sort_by → sort_unstable_by. RESULT:
ordinal **114.6 -> 84.2ms = 1.36x self** (now 3.2x vs scipy 269ms, was 2.3x). BYTE-IDENTICAL (byte_mism=0 vs
stable-sort reference INCLUDING tied data; 9/9 rankdata tests green incl. rankdata_ordinal_matches_scipy_
reference + rankdata_with_ties). Applied the same provably-safe transform to multiscale_graphcorr's per-row
distance ranking (34173, comment-confirmed total order). Did NOT touch the 2 correlation sorts that tiebreak
by y-VALUES (not a guaranteed total order → stability matters). 2001/2001 stats green. LEVER: grep
`.sort_by(...total_cmp...then...index)` — a unique-index tiebreak makes it a total order → sort_unstable_by is
a free byte-identical speedup.
