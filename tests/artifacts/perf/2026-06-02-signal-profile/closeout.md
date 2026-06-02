# frankenscipy-ifo7z closeout

## Target

- Crate: `fsci-signal`
- Profile-backed hotspot: `design/remez/257_two_band`
- Broad baseline: `632.46 ms 639.01 ms 645.69 ms` on `vmi1153651`
- Focused baseline: `294.95 ms 299.13 ms 303.39 ms` on `vmi1149989`

## One lever

`remez` now computes the cosine basis for each frequency grid point once and reuses
those values while assembling the weighted normal equations.

## Result

- Focused after: `48.992 ms 52.648 ms 58.029 ms` on `vmi1227854`
- Broad reprofile after: `35.062 ms 35.795 ms 36.553 ms` on `vmi1149989`
- Same-worker comparison: `299.13 ms -> 35.795 ms` median, 8.36x faster
- Original broad comparison: `639.01 ms -> 35.795 ms` median, 17.85x faster

## Behavior proof

- Golden before sha256: `55cd534a7c608b0d07be5ada52a24d38ee5512c57718ae32439028f44b4df45a`
- Golden after sha256: `55cd534a7c608b0d07be5ada52a24d38ee5512c57718ae32439028f44b4df45a`
- Byte comparison: `SIGNAL_GOLDEN_CMP_EXIT:0`
- Isomorphism:
  - Frequency-grid traversal order is unchanged.
  - Coefficient `j` and `k` traversal order is unchanged.
  - Weighted normal-equation accumulation order is unchanged.
  - The formula remains `cos(two_pi * coefficient_index * frequency)`.
  - The linear solve path and output ordering are unchanged.
  - No RNG is used by this routine.

## Validation

- `rch exec -- cargo check -p fsci-signal --bench signal_bench --locked`: pass
- `rch exec -- cargo clippy -p fsci-signal --lib --bin perf_signal --bench signal_bench --locked -- -D warnings`: pass on `vmi1153651`
- `rch exec -- cargo test -p fsci-signal --lib --locked`: pass on `vmi1293453`, 497 passed, 0 failed
- `cargo fmt -p fsci-signal --check`: pass
- `ubs crates/fsci-signal/src/lib.rs crates/fsci-signal/src/bin/perf_signal.rs crates/fsci-signal/benches/signal_bench.rs`: nonzero due existing broad `fsci-signal/src/lib.rs` inventory, filed as `frankenscipy-0cbi6`
- Full all-targets clippy is blocked by existing signal test lints, filed as `frankenscipy-z7ot6`

## Score

- Impact: 5.0
- Confidence: 4.5
- Effort: 1.0
- Score: 22.5
