# Negative Evidence Ledger

Canonical detailed ledger: `docs/progress/perf-negative-results.md`.

This file exists as the BOLD-VERIFY entry point requested for measured
win/loss/neutral summaries. Keep detailed attempt records in the canonical
ledger above so the project has one source of truth.

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
