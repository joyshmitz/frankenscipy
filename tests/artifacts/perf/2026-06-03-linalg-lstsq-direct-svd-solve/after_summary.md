# Direct Lstsq SVD Solve After

- Timestamp: 2026-06-03T17:46:06-04:00
- Bead: `frankenscipy-8l8r1.28`
- Profile-backed target: `baseline_lstsq/1000x500`
- Baseline command: `RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked -- 'baseline_lstsq/1000x500' --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot`
- After command: same benchmark command after the source lever.

## Criterion Medians

| benchmark | worker | lower | median | upper |
| --- | --- | ---: | ---: | ---: |
| baseline `baseline_lstsq/1000x500` | `vmi1153651` | `818.76 ms` | `832.97 ms` | `847.68 ms` |
| after `baseline_lstsq/1000x500` | `ts2` | `471.29 ms` | `479.61 ms` | `489.09 ms` |
| after confirm `baseline_lstsq/1000x500` | `ts2` | `468.25 ms` | `470.34 ms` | `472.56 ms` |

## Delta

- First after run: `832.97 ms -> 479.61 ms`, `1.74x` faster.
- Confirmation run: `832.97 ms -> 470.34 ms`, `1.77x` faster.
- RCH does not expose a per-command worker pin; confirmation stayed on `ts2` and stabilized the after estimate.

## Behavior Proof

- Before sorted stable SHA-256: `c2a369f0e0798d8f56cf5f323abdb22229710ab2e43cf7acd58afe2c0341f8f7`
- After sorted stable SHA-256: `c2a369f0e0798d8f56cf5f323abdb22229710ab2e43cf7acd58afe2c0341f8f7`
- Sorted stable before/after diff: empty.
- Order-sensitive stable diff contained only RCH timestamp and nondeterministic Rust test-line ordering; all selected `lstsq` tests passed before and after.
- The lever preserves validation/error order, SVD inputs/options, singular-value ordering, threshold/rank semantics, residual conditions, certificate fields, output vector order, RNG absence, tie-breaking absence, and global-state absence.

## Validation

- `cargo fmt -p fsci-linalg --check`: exit `0`
- RCH `cargo test -p fsci-linalg --release --locked lstsq -- --nocapture`: exit `0`
- UBS `crates/fsci-linalg/src/lib.rs`: exit `0`
- RCH `cargo check -p fsci-linalg --all-targets --locked`: exit `0`
- RCH `cargo clippy -p fsci-linalg --all-targets --locked -- -D warnings`: exit `0`

## Gate

Kept with Score `6.0 = impact 4 * confidence 3 / effort 2`.
