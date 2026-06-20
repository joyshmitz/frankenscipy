# Negative Evidence Ledger

Canonical detailed ledger: `docs/progress/perf-negative-results.md`.

This file exists as the BOLD-VERIFY entry point requested for measured
win/loss/neutral summaries. Keep detailed attempt records in the canonical
ledger above so the project has one source of truth.

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
