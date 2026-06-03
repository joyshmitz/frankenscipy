# frankenscipy-lt8kr after benchmark

Command:

```text
RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked -- matmul --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

Remote worker: `vmi1149989`

## Baseline medians

- `matmul/256x256`: `4.6382 ms`
- `matmul/512x512`: `37.058 ms`
- `matmul/768x768`: `119.30 ms`
- `matmul/1024x1024`: `188.06 ms`

## Trial medians

- `matmul/256x256`: `5.0911 ms`
- `matmul/512x512`: `38.759 ms`
- `matmul/768x768`: `141.65 ms`
- `matmul/1024x1024`: `225.97 ms`

## Decision

Rejected. The production gate row regressed from `188.06 ms` to `225.97 ms`
(`0.83x` of baseline throughput), below the Score `2.0` keep gate.

The source edit was manually restored. `git diff -- crates/fsci-linalg/src/lib.rs`
was empty after restore, and `cargo fmt -p fsci-linalg --check` passed.
