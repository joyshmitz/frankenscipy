# Negative Evidence Ledger

Canonical detailed ledger: `docs/progress/perf-negative-results.md`.

This file exists as the BOLD-VERIFY entry point requested for measured
win/loss/neutral summaries. Keep detailed attempt records in the canonical
ledger above so the project has one source of truth.

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
