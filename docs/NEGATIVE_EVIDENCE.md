# Negative Evidence Ledger

Canonical detailed ledger: `docs/progress/perf-negative-results.md`.

This file exists as the BOLD-VERIFY entry point requested for measured
win/loss/neutral summaries. Keep detailed attempt records in the canonical
ledger above so the project has one source of truth.

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
