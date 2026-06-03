# MR6 Register-Fit GEMM V2 Baseline

- Timestamp: 2026-06-03T15:22:24-04:00
- Bead: `frankenscipy-8l8r1.24`
- Worker: `vmi1153651`
- Command: `RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked -- matmul --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot`
- Exit: `0`
- Profile-backed target: `matmul/1024x1024`

## Criterion Medians

| benchmark | lower | median | upper |
| --- | ---: | ---: | ---: |
| `matmul/256x256` | `12.506 ms` | `13.666 ms` | `15.200 ms` |
| `matmul/512x512` | `118.84 ms` | `136.85 ms` | `156.94 ms` |
| `matmul/768x768` | `946.54 ms` | `988.04 ms` | `1.0277 s` |
| `matmul/1024x1024` | `769.70 ms` | `817.83 ms` | `869.75 ms` |

## Gate

The keep gate is a real `matmul/1024x1024` median win versus `817.83 ms`, with unchanged golden-output SHA-256 and Score `>= 2.0`.
