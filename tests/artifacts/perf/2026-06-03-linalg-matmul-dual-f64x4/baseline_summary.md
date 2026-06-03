# frankenscipy-lt8kr baseline

Pass: 1 - Fresh Baseline And Profile

HEAD: `c8efcf0a8b59563b47f23489a7daf64516e4916a`

RCH worker: `vmi1149989`

Command:

```bash
RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked -- matmul --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

Criterion rows:

| row | lower | median | upper |
| --- | ---: | ---: | ---: |
| `matmul/256x256` | `4.3860 ms` | `4.6382 ms` | `5.0017 ms` |
| `matmul/512x512` | `35.498 ms` | `37.058 ms` | `38.863 ms` |
| `matmul/768x768` | `112.27 ms` | `119.30 ms` | `123.77 ms` |
| `matmul/1024x1024` | `179.34 ms` | `188.06 ms` | `197.04 ms` |

Profile-backed target:
The latest committed linalg reprofile still ranks `matmul/1024x1024` first.
This pass uses the fresh `188.06 ms` focused median as the keep-gate baseline.
