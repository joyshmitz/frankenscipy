# Direct Lstsq SVD Solve Baseline

- Timestamp: 2026-06-03T17:32:23-04:00
- Bead: `frankenscipy-8l8r1.28`
- Worker: `vmi1153651`
- Command: `RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked -- 'baseline_lstsq/1000x500' --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot`
- Exit: `0`
- Profile-backed target: `baseline_lstsq/1000x500`

## Criterion Median

| benchmark | lower | median | upper |
| --- | ---: | ---: | ---: |
| `baseline_lstsq/1000x500` | `818.76 ms` | `832.97 ms` | `847.68 ms` |

## Gate

The keep gate is a real `baseline_lstsq/1000x500` median win versus `832.97 ms`, with unchanged stable golden-output SHA-256 and Score `>= 2.0`.
