# PSD Welch Indexed Inner Loop Negative Result

Bead: `frankenscipy-skjpy`

## Verdict

Rejected. Replacing the innermost PSD Welch iterator zip with an indexed loop
preserved output but regressed the focused RCH benchmark, so production source
was restored and no code change is kept.

## Profile-Backed Target

Source profile:

`tests/artifacts/perf/2026-06-03-stats-psd-twiddle-soa/reprofile_stats_after_psd_soa_rch.txt`

Relevant row:

- `time_series/psd_welch/4096_w128_o64`: `[255.46 us, 257.41 us, 259.60 us]`

Fresh focused baseline for this bead:

- `time_series/psd_welch/4096_w128_o64`: `[541.86 us, 549.72 us, 558.78 us]` on `vmi1156319`

## One Lever Tested

Only the innermost PSD Welch accumulation loop changed during the trial:
the `segment.iter().zip(cos_row.iter()).zip(sin_row.iter())` walk became an
indexed `while` loop over the same three slices. The window, twiddle layout,
segment construction, frequency order, and final averaging were unchanged.

## Behavior Surface

The preserved surfaces were:

- Validation and early-return order.
- Hann window generation bits.
- Twiddle generation order: frequency-major, then sample-major.
- Segment order.
- Frequency order.
- Sample order within each segment.
- Output frequency-bin order.
- Per-sample floating-point sequence: `re += s * cos`, then `im -= s * sin`.
- RNG absence, tie-breaking absence, and existing 128-point `OnceLock` cache semantics.

Golden before and after:

`85048a3c06ab045815cbeb238fee9e1e07a05c27ceed3c3782ec0fd5ea97c6b1`

`cmp` returned 0 for `golden_psd_before.txt` versus `golden_psd_after.txt`.

## Benchmark Gate

Focused RCH after:

- `time_series/psd_welch/4096_w128_o64`: `[645.20 us, 662.35 us, 680.67 us]` on `vmi1156319`

Median comparison: `549.72 us -> 662.35 us`, a regression.

Score: `0.0` because performance impact was negative.

## Post-Restoration Reprofile

After restoring production source, a crate-scoped RCH full stats reprofile was
captured in:

`tests/artifacts/perf/2026-06-03-stats-psd-indexed-inner/reprofile_stats_after_psd_indexed_inner_rch.txt`

Visible rows:

- `qmc_discrepancy/centered/512x2`: `[579.12 us, 662.75 us, 755.55 us]`
- `qmc_discrepancy/mixture/512x2`: `[813.37 us, 972.55 us, 1.1774 ms]`
- `qmc_discrepancy/l2_star/512x2`: `[709.98 us, 875.78 us, 1.0701 ms]`
- `qmc_discrepancy/wraparound/512x2`: `[749.98 us, 989.83 us, 1.2644 ms]`
- `time_series/psd_welch/4096_w128_o64`: `[657.60 us, 740.93 us, 852.79 us]`

Next profile-backed target: QMC 2D discrepancy remains the hottest visible
stats family, with `wraparound` and `mixture` at the top of this post-restore
profile.

## Restoration And Validation

Production `crates/fsci-stats/src/lib.rs` is restored to HEAD.

- `source_restored_diff.txt`: empty.
- `cargo fmt -p fsci-stats --check`: pass.
- RCH `cargo check -p fsci-stats --all-targets`: exit 0.
- RCH `cargo clippy -p fsci-stats --all-targets -- -D warnings`: exit 0.
- RCH `cargo test -p fsci-stats psd_welch -- --nocapture`: exit 0.
