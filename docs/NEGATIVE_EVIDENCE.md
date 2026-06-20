# Negative Evidence Ledger

Canonical detailed ledger: `docs/progress/perf-negative-results.md`.

This file exists as the BOLD-VERIFY entry point requested for measured
win/loss/neutral summaries. Keep detailed attempt records in the canonical
ledger above so the project has one source of truth.

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
