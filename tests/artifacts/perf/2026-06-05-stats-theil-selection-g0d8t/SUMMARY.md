# frankenscipy-g0d8t: Theil slope interval selection

Status: keep

Lever: replace the production `theil_sen` / `theilslopes` clean-input median and CI slope selection path with deterministic sampled bracketing plus exact bounded interval enumeration. The materialized O(n^2) path remains the fallback for ties, tiny x gaps, non-finite inputs, zero-slope ambiguity, oversized candidate intervals, or any count-bracket proof failure.

Baseline:
- Command: `RCH_FORCE_REMOTE=1 rch exec -- cargo run --release -p fsci-stats --bin perf_theilslopes --locked`
- Artifact: `baseline_perf_theilslopes_rch.txt`
- Worker: ts1
- Timings: n=1000 4.011 ms, n=2000 15.570 ms, n=3000 59.197 ms for the previous in-place median path.

After:
- Command: `RCH_FORCE_REMOTE=1 rch exec -- cargo run --release -p fsci-stats --bin perf_theilslopes --locked`
- Artifact: `after_interval_selection_perf_theilslopes_rch_2.txt`
- Worker: ts2
- Timings: n=1000 5.342 ms, n=2000 21.443 ms, n=3000 51.969 ms for the interval-selection path.
- Primary evidence: the same after run's old-vs-new internal reference ratios are 4.5x, 5.2x, and 5.3x for n=1000/2000/3000.

Behavior proof:
- Golden payload: `golden_after_interval_payload.txt`
- Golden sha256: `be295cbde6b5b9e4e57fa9df3705fa3002f09453b1c0abfcc4b65bce752f641a`
- Harness parity: 0 numeric mismatches across 180 cases / 1080 fields; 20 benign +0.0/-0.0 sign-only diffs remain numerically equal.
- Ordering/ties: fast path requires finite x/y and sorted adjacent x gaps strictly greater than `THEIL_SLOPE_MIN_X_GAP`; otherwise it falls back to the materialized pair-order path.
- Floating point: accepted fast path enumerates the exact final slope interval and selects order statistics with `total_cmp`; count brackets are used only to choose a provably containing interval, not to approximate returned values.
- RNG: the only RNG is a deterministic local LCG used to sample bracket candidates; it has no public state, no external RNG interaction, and cannot affect the returned value unless the exact interval proof succeeds.

Validation:
- `cargo fmt -p fsci-stats --check`: pass.
- `cargo test -p fsci-stats --lib rank_selection --locked -- --nocapture`: pass, 2/2.
- `cargo test -p fsci-stats --lib theil --locked -- --nocapture`: pass, 12/12.
- `cargo check -p fsci-stats --all-targets --locked`: pass.
- `cargo clippy -p fsci-stats --all-targets --no-deps --locked -- -D warnings`: pass.
- `cargo clippy -p fsci-stats --all-targets --locked -- -D warnings`: blocked before stats by an unrelated existing `fsci-special/src/beta.rs` `if_same_then_else` lint.
- `cargo test -p fsci-stats --lib --locked -- --nocapture`: blocked by RCH SSH timeout after 1800s in unrelated long-running distribution tests; targeted Theil proof tests passed.

Score:
- Impact: 5
- Confidence: 4
- Effort: 2
- Score: 10.0

Next profile target after this bead: re-run `br ready --json`; current open perf candidates are sparse `spilu` dense-workspace lookup and interpolate `polymul` FFT convolution, with `spilu` higher priority.
