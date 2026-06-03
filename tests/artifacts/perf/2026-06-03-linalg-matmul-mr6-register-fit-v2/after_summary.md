# MR6 Register-Fit GEMM V2 After Benchmark

- Timestamp: 2026-06-03T15:36:22-04:00
- Bead: `frankenscipy-8l8r1.24`
- Worker: `vmi1153651`
- Command: `RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked -- matmul --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot`
- Exit: `0`
- Golden SHA: `b756bde7f00b52f08f37f77e67b4f03abcb06b2c551fe04315be57004b40551e`

## Median Delta

| benchmark | baseline median | after median | delta |
| --- | ---: | ---: | ---: |
| `matmul/256x256` | `13.666 ms` | `11.963 ms` | `1.14x faster` |
| `matmul/512x512` | `136.85 ms` | `130.03 ms` | `1.05x faster` |
| `matmul/768x768` | `988.04 ms` | `1.0289 s` | `0.96x slower` |
| `matmul/1024x1024` | `817.83 ms` | `882.22 ms` | `0.93x slower` |

## Decision

Rejected. The proof stayed isomorphic, but the keep-gate row regressed from `817.83 ms` to `882.22 ms`, so the lever scores below `2.0`.
