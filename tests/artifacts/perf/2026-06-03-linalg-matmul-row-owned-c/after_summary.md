# frankenscipy-8l8r1.21 after benchmark

Pass: 5 - RCH Gate

RCH worker: `vmi1153651`

Command:

```bash
RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked -- matmul --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

Criterion rows:

| row | lower | median | upper |
| --- | ---: | ---: | ---: |
| `matmul/256x256` | `11.426 ms` | `12.023 ms` | `12.750 ms` |
| `matmul/512x512` | `99.291 ms` | `116.66 ms` | `145.15 ms` |
| `matmul/768x768` | `770.97 ms` | `819.74 ms` | `865.48 ms` |
| `matmul/1024x1024` | `866.79 ms` | `889.14 ms` | `911.89 ms` |

Decision:
Rejected. The saved after run does not show a keepable win against the pass
baseline `348.89 ms` median for `matmul/1024x1024`.
