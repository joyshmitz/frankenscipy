# fsci-stats argsort MSD radix rejection

Bead: `frankenscipy-rimc9`

## Target

Fresh RCH profile after the stats rand-index and silhouette drops identified
`ordering_and_bins/argsort/65536` as the largest remaining `fsci-stats` row.

Baseline command:

```bash
rch exec -- cargo bench -p fsci-stats --bench stats_bench -- --warm-up-time 1 --measurement-time 2 --sample-size 10
```

Baseline worker: `ts2`

| benchmark | before median |
| --- | ---: |
| `ordering_and_bins/argsort/4096` | 85.863 us |
| `ordering_and_bins/argsort/65536` | 2.4309 ms |

## Lever

Tried a bounded MSD radix partition over the existing monotonic float-order keys
for large NaN-free inputs. This was meant to avoid the prior rejected LSD-radix
failure mode by stopping after high-order bytes split the data instead of
scattering all eight key bytes over the full array.

## Behavior Proof

Golden SHA-256 before and after:

```text
00e186be74d4dbc7099dd8cfb5b1802bc61c410ac9138726158922c76e8ffd6a
```

Golden cases covered the continuous benchmark-shaped input, heavy ties with
signed zero and infinities, and NaN fallback ordering.

Focused validation passed while the lever was present:

```bash
rch exec -- cargo test -p fsci-stats argsort --lib -- --nocapture
rch exec -- cargo check -p fsci-stats --all-targets
rch exec -- cargo clippy -p fsci-stats --all-targets -- -D warnings
cargo fmt -p fsci-stats --check
ubs crates/fsci-stats/src/lib.rs crates/fsci-stats/src/bin/perf_stats.rs
```

## After

After command:

```bash
rch exec -- cargo bench -p fsci-stats --bench stats_bench -- ordering_and_bins/argsort --warm-up-time 1 --measurement-time 2 --sample-size 10
```

After worker: `ts2`

| benchmark | after median | verdict |
| --- | ---: | --- |
| `ordering_and_bins/argsort/4096` | 83.347 us | small win |
| `ordering_and_bins/argsort/65536` | 3.0611 ms | regression |

The target row regressed 26.0%, so Score is 0.0 and the source was restored.

## Next Primitive

Do not iterate this radix-scatter direction. Pivot to a different profiled
primitive from the same stats profile: `psd_welch` (~571.40 us) or the rank
correlation path (~504.32 us), subject to live bead ownership.
