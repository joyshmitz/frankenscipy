# fsci-ndimage Rank Filter Nth-Select

Bead: `frankenscipy-a17zm`

## Target

Fresh crate-scoped RCH Criterion profiling showed `rank_filter` dominated `fsci-ndimage` after the prior separable min/max and morphology wins.

Initial baseline command:

```bash
RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-ndimage --bench ndimage_bench --locked -- --warm-up-time 1 --measurement-time 3 --sample-size 20 --noplot
```

The first baseline run completed remotely on `vmi1153651` (`[RCH] remote vmi1153651 (262.8s)`). Because the first command used stdout-only redirection before the RCH stream behavior was understood, `baseline_ndimage_rch.txt` is empty; these rows are copied from that RCH run's live stream:

| Row | Baseline Median |
| --- | ---: |
| `rank_filter/median_160x160/7` | `137.61 ms` |
| `rank_filter/rank_q25_160x160/7` | `133.17 ms` |
| `rank_filter/median_160x160/15` | `658.06 ms` |
| `rank_filter/rank_q25_160x160/15` | `648.67 ms` |

## Lever

Replace per-neighborhood full `sort_by(total_cmp)` with `select_nth_unstable_by(total_cmp)` at the requested rank in the three selected-footprint paths:

- `median_filter_with_origins`
- `rank_filter_index_with_origins`
- `rank_filter_index_usize_axes_with_origins`

This changes order-statistic extraction from full ordering to nth selection while preserving neighborhood enumeration, boundary/origin handling, rank normalization, and the exact `f64::total_cmp` order.

## Isomorphism

- Ordering: same `total_cmp` comparator.
- Ties: `total_cmp` equality means the selected `f64` bit pattern is equal, so unstable partition order is not output-observable.
- Floating point: no arithmetic expression changes; only the selected value movement changes.
- RNG: none.
- Boundary and origin behavior: unchanged because all neighborhood construction code is unchanged.

Golden command:

```bash
RCH_GENERAL_FORCE_REMOTE=true rch exec -- cargo run -p fsci-ndimage --bin perf_rank_filter --release --locked
```

The payload was captured from the RCH stream. Before and after payloads compare byte-identical:

```text
cmp_exit=0
sha256=f8e053d76d1ccb9c3740b4651faf94ca6d96c585be41888dbba88e92812ff009
```

Both before and after golden runs completed remotely on `ts2`.

## Performance

After benchmark command:

```bash
RCH_GENERAL_FORCE_REMOTE=true rch exec -- cargo bench -p fsci-ndimage --bench ndimage_bench --locked -- rank_filter --warm-up-time 1 --measurement-time 3 --sample-size 20 --noplot
```

After run: `vmi1149989`, `[RCH] remote vmi1149989 (136.8s)`.

| Row | Baseline Median | After Median | Ratio |
| --- | ---: | ---: | ---: |
| `rank_filter/median_160x160/7` | `137.61 ms` | `49.675 ms` | `2.77x` |
| `rank_filter/rank_q25_160x160/7` | `133.17 ms` | `46.284 ms` | `2.88x` |
| `rank_filter/median_160x160/15` | `658.06 ms` | `199.33 ms` | `3.30x` |
| `rank_filter/rank_q25_160x160/15` | `648.67 ms` | `245.58 ms` | `2.64x` |

Confirmation ran remotely on `vmi1264463`, but that worker was much slower and is not comparable for ratios (`137.72 ms`, `138.98 ms`, `691.74 ms`, `696.68 ms` medians). It is retained as a noise artifact, not the keep baseline.

Score: `8.0 = impact 4 * confidence 4 / effort 2`, above the `>=2.0` keep threshold.

## Validation

- `cargo fmt -p fsci-ndimage --check`: pass.
- RCH `cargo test -p fsci-ndimage --lib --locked rank_filter -- --nocapture`: `10 passed; 0 failed` on `ts2`.
- RCH `cargo check -p fsci-ndimage --all-targets --locked`: pass on `ts2`.
- RCH `cargo clippy -p fsci-ndimage --all-targets --locked -- -D warnings`: pass on `ts2`.
- `ubs crates/fsci-ndimage/src/bin/perf_rank_filter.rs`: exit `0`.
- `ubs crates/fsci-ndimage/src/lib.rs crates/fsci-ndimage/src/bin/perf_rank_filter.rs`: exit `1` due pre-existing file-wide `lib.rs` findings; no finding points to `select_total_rank` or the changed rank-selection lines.

## Verdict

Kept. The lever is an algorithmic order-statistic replacement with byte-identical golden output and a clear RCH Criterion win on the targeted rank-filter rows.
