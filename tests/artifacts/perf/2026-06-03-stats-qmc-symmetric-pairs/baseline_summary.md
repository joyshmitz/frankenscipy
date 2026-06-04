# QMC Symmetric Pair Baseline

Bead: `frankenscipy-pgm85`
Head: `1d80c75205de8a9ac4161dfff95756958c97e39e`

## Profile Target

Existing post-stats RCH Criterion reprofiles show 2D QMC discrepancy as the fallback hotspot after the PSD wins:

- `qmc_discrepancy/mixture/512x2`: `276.17 us` median in `2026-06-03-stats-sobol-incremental/reprofile_stats_after_sobol_incremental_rch.txt`.
- `qmc_discrepancy/l2_star/512x2`: `232.68 us` median in the same reprofile.
- Full 100-sample reprofile still showed `mixture/512x2` at `608.03 us` median and `l2_star/512x2` at `500.80 us` median.

## Focused RCH Criterion Baseline

Command:

```text
rch exec -- cargo bench -p fsci-stats --bench stats_bench --locked -- qmc_discrepancy --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot
```

Worker: `ts2`

Rows:

- `qmc_discrepancy/centered/512x2`: `[311.30 us, 312.56 us, 313.91 us]`
- `qmc_discrepancy/mixture/512x2`: `[436.50 us, 437.57 us, 438.19 us]`
- `qmc_discrepancy/l2_star/512x2`: `[328.30 us, 329.44 us, 330.71 us]`
- `qmc_discrepancy/wraparound/512x2`: `[317.86 us, 318.55 us, 319.94 us]`

## Golden Before

Normalized payload SHA-256:

```text
1fb5885cc35367f57b0e818e165a28f87cbb0b9a43fdc7ba4728a6778af44daf
```

Payload:

```text
case=qmc_discrepancy_512x2 len=1024
centered=3eef42483d6e0000
mixture=3eef147788540000
l2_star=3f67c5072f2a347c
wraparound=3ee455e9ae6c0000
```
