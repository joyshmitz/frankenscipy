# Direct Pinv Sigma Scaling Baseline

- Timestamp: 2026-06-03T17:09:10-04:00
- Bead: `frankenscipy-8l8r1.27`
- Worker: `vmi1149989`
- Command: `RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked -- 'baseline_pinv/1000x500' --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot`
- Exit: `0`
- Profile-backed target: `baseline_pinv/1000x500`

## Criterion Median

| benchmark | lower | median | upper |
| --- | ---: | ---: | ---: |
| `baseline_pinv/1000x500` | `312.44 ms` | `316.20 ms` | `320.04 ms` |

## Gate

The keep gate is a real `baseline_pinv/1000x500` median win versus `316.20 ms`, with unchanged stable golden-output SHA-256 and Score `>= 2.0`.
