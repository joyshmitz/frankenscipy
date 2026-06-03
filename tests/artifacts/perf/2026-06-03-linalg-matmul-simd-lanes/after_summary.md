# frankenscipy-8l8r1.22 after benchmark

Pass: 5 - RCH Gate

RCH worker: `vmi1293453`

Command:

```bash
RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked -- matmul --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

Criterion rows:

| row | lower | median | upper |
| --- | ---: | ---: | ---: |
| `matmul/256x256` | `5.1433 ms` | `5.2061 ms` | `5.2955 ms` |
| `matmul/512x512` | `38.334 ms` | `39.680 ms` | `40.852 ms` |
| `matmul/768x768` | `130.31 ms` | `132.01 ms` | `133.61 ms` |
| `matmul/1024x1024` | `212.46 ms` | `219.74 ms` | `229.03 ms` |

Keep comparison:
The saved pass baseline is `817.34 ms` for `matmul/1024x1024`; the after
median is `219.74 ms` (`3.72x`).
