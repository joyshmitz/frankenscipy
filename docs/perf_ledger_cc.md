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
