# frankenscipy-8l8r1.84 Baseline Contract

Target: `fsci-linalg::matmul` flat-workspace GEMM row-native output materialization.

Bead: `frankenscipy-8l8r1.84`

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo bench -j 1 -p fsci-linalg --bench linalg_bench --locked -- matmul
```

Artifact:

```text
1358109b5d2cc23731862ae9f7419ffe617203b3e8074eee3e6e21deded8da0e  tests/artifacts/perf/2026-06-11-linalg-8l8r1-84/baseline_matmul_criterion_rch.txt
```

Note: the raw artifact starts with an earlier remote-required refusal from the same attempted baseline capture, then contains the completed remote run. The accepted baseline is the completed `vmi1227854` Criterion run ending with `Remote command finished: exit=0`.

Criterion means on `vmi1227854`:

| shape | mean |
| --- | ---: |
| 256x256 | 5.0186 ms |
| 512x512 | 18.232 ms |
| 768x768 | 104.81 ms |
| 1024x1024 | 169.27 ms |

Contract:

- Preserve public API, shape/error behavior, rectangularity gates, fallback paths, output ordering, and deterministic row ownership.
- Preserve monotonic `k = 0..ka` accumulation for every output element.
- Do not repeat B staging/direct-pack, panel-load abstraction, scalar-splat spelling, MR/NR widening, worker-count row scheduling, 8-row row-panel accumulators, K-major A row-slab packing, RB geometry, KC-striped C writeback, or the kept 4x24 tile-width-only family.
- Keep only with same-worker RCH rebench and Score >= 2.0; otherwise restore source and route deeper.
