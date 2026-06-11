# Rejected: duplicate row-owned GEMM output materialization

Bead: `frankenscipy-8l8r1.84`

## Decision

No source lever was applied. The proposed row-native output materialization is a
duplicate of the prior `frankenscipy-8l8r1.21` row-owned C-output trial in:

```text
tests/artifacts/perf/2026-06-03-linalg-matmul-row-owned-c/
```

That prior trial selected the same primitive: allocate the public `Vec<Vec<f64>>`
rows up front and write completed GEMM tiles directly into owned output rows
instead of filling `c_flat` and copying chunks with `row.to_vec()`.

## Current baseline

Fresh RCH baseline for this bead:

```text
1358109b5d2cc23731862ae9f7419ffe617203b3e8074eee3e6e21deded8da0e  tests/artifacts/perf/2026-06-11-linalg-8l8r1-84/baseline_matmul_criterion_rch.txt
```

Completed remote worker: `vmi1227854`.

| shape | mean |
| --- | ---: |
| `matmul/256x256` | `5.0186 ms` |
| `matmul/512x512` | `18.232 ms` |
| `matmul/768x768` | `104.81 ms` |
| `matmul/1024x1024` | `169.27 ms` |

## Prior duplicate evidence

Prior row-owned output trial:

```text
tests/artifacts/perf/2026-06-03-linalg-matmul-row-owned-c/CONCLUSION.md
tests/artifacts/perf/2026-06-03-linalg-matmul-row-owned-c/primitive_selection.md
```

The prior trial preserved behavior proofs but failed the score gate:

| shape | prior baseline median | prior after median |
| --- | ---: | ---: |
| `matmul/256x256` | `5.3319 ms` | `12.023 ms` |
| `matmul/512x512` | `41.678 ms` | `116.66 ms` |
| `matmul/768x768` | `130.94 ms` | `819.74 ms` |
| `matmul/1024x1024` | `348.89 ms` | `889.14 ms` |

The prior source edit was restored, and `git diff -- crates/fsci-linalg/src/lib.rs`
is empty for this `.84` closeout.

## Isomorphism status

No production source changed in this bead.

- Ordering/tie behavior: unchanged.
- Floating point: unchanged; no arithmetic route was edited.
- RNG: unchanged; GEMM has no RNG.
- Shape/error behavior: unchanged.
- Golden output: not rerun because the duplicate source lever was not applied.

## Route

Close `.84` as rejected without repeating the duplicate output-materialization
family. The next primitive must be profile-backed and materially different from:
B staging/direct-pack, panel-load spelling, scalar-splat spelling, MR/NR widening,
worker-count row scheduling, 8-row row-panel accumulators, K-major A row-slab
packing, RB geometry, KC-striped C writeback, row-owned output materialization,
and the kept 4x24 tile-width-only family.
