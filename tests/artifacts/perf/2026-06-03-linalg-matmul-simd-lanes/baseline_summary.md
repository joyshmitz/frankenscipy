# frankenscipy-8l8r1.22 baseline

Pass: 1 - Fresh Baseline And Profile

HEAD: `a633e89d7b1da8755ae580e5f1f64e6e80069116`

RCH worker: `vmi1156319`

Command:

```bash
RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked -- matmul --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

Criterion rows:

| row | lower | median | upper |
| --- | ---: | ---: | ---: |
| `matmul/256x256` | `10.902 ms` | `11.363 ms` | `11.855 ms` |
| `matmul/512x512` | `91.692 ms` | `94.399 ms` | `97.486 ms` |
| `matmul/768x768` | `487.22 ms` | `543.23 ms` | `619.85 ms` |
| `matmul/1024x1024` | `796.89 ms` | `817.34 ms` | `843.93 ms` |

Profile-backed target:
The latest linalg reprofile and focused baselines still identify large GEMM as
the optimization target. This pass uses `matmul/1024x1024` median `817.34 ms`
as the keep-gate baseline.
