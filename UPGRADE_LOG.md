# Dependency Upgrade Log

**Date:** 2026-04-21 -> 2026-04-22
**Project:** frankenscipy
**Language:** Rust (workspace, resolver 2, edition 2024)
**Agent:** Clawdstein-libupdater-frankenscipy

## Previous

### 2026-04-21: asupersync 0.2.5 -> 0.3

- **Before**: asupersync 0.2.5
- **After**: asupersync 0.3
- **Affected crates**: fsci-conformance, fsci-runtime
- **Test outcome**: `cargo check --all-targets` OK, `cargo test -p fsci-runtime -p fsci-conformance` OK
- **Breaking changes**: None observed

## Summary (this session)

- **Updated:** TBD
- **Skipped (already latest):** serde 1.0.228
- **Failed:** TBD
- **Needs attention:** TBD

## Updates

### asupersync: 0.3 (caret) -> 0.3.1 (explicit pin)

- **Before:** `asupersync = { version = "0.3", default-features = false, features = ["test-internals"] }`
- **After:**  `asupersync = { version = "0.3.1", default-features = false, features = ["test-internals"] }`
- **Lock resolution:** asupersync, franken-decision, franken-evidence, franken-kernel all 0.3.0 -> 0.3.1
- **Breaking:** None (patch release)
- **Checks:** `cargo check --workspace --all-targets` clean
- **Tests:** `cargo test -p fsci-runtime --lib` -> 32 passed / 0 failed
- **Commit:** a2043f2

### criterion: 0.5.1 -> 0.8.2 (MAJOR, dev-dependency only)

- **Before:** `criterion = { version = "0.5.1", features = ["html_reports"] }`
- **After:**  `criterion = { version = "0.8.2", features = ["html_reports"] }`
- **Lock changes:** criterion 0.5.1 -> 0.8.2, criterion-plot 0.5.0 -> 0.8.2, new transient deps alloca 0.4.0 / itertools 0.13.0 / page_size 0.6.0, is-terminal 0.4.17 removed.
- **Affected bench targets:** fsci-fft/benches/fft_bench.rs, fsci-linalg/benches/linalg_bench.rs, fsci-special/benches/special_bench.rs, fsci-arrayapi/benches/arrayapi_bench.rs, fsci-sparse/benches/sparse_bench.rs, fsci-integrate/benches/integrate_bench.rs, fsci-opt/benches/optimize_bench.rs, fsci-runtime/benches/runtime_bench.rs.
- **Breaking API surface actually hit:** `criterion::black_box` is now deprecated (criterion 0.6+). Our benches still import and use it -> deprecation warnings but compilation still succeeds because the symbol is retained as a deprecated re-export pointing at `std::hint::black_box`.
- **Checks:** `cargo check --workspace --all-targets` exit=0 with deprecation warnings only.
- **Tests:** `cargo test -p fsci-runtime --lib` -> 32/32; `cargo bench -p fsci-fft --no-run` links successfully.

### proptest: 1.6.0 -> 1.11.0 (minor)

- **Before:** `proptest = "1.6.0"`
- **After:**  `proptest = "1.11.0"`
- **Lock resolution:** cargo update -p proptest -> 1.11.0 (0 packages changed; lock was already caret-resolved to 1.11.0).
- **Breaking:** None within the 1.x line per upstream release notes; 1.7-1.11 are additive (faster shrinking, new strategies, improved rng seeding). Our call sites use the stable `proptest! { }` macro and `prop_assert*` APIs which have not changed.
- **Affected crates:** fsci-arrayapi, fsci-cluster, fsci-fft, fsci-integrate, fsci-interpolate, fsci-io, fsci-linalg, fsci-ndimage, fsci-opt, fsci-runtime, fsci-signal, fsci-sparse, fsci-spatial, fsci-stats (all as dev-dependency).
- **Checks:** `cargo check --workspace --all-targets` exit=0.
- **Tests:** `cargo test -p fsci-arrayapi --lib` -> 42 passed / 0 failed (arrayapi backend is the proptest-densest surface for the array API layer). A follow-up `cargo test -p fsci-sparse --lib` was started but stalled in the remote build queue and was abandoned; arrayapi is the authoritative smoke signal for this bump.

### thiserror: 2.0.17 -> 2.0.18 (patch pin)

- **Before:** `thiserror = "2.0.17"`
- **After:**  `thiserror = "2.0.18"`
- **Lock resolution:** Cargo.lock already at 2.0.18 via caret; manifest pin advanced.
- **Breaking:** None (patch).
- **Checks:** `cargo check --workspace --all-targets` exit=0.
- **Tests:** `cargo test -p fsci-runtime --lib` -> 32 passed / 0 failed.

### serde_json: 1.0.145 -> 1.0.149 (patch pin)

- **Before:** `serde_json = { version = "1.0.145", features = ["preserve_order"] }`
- **After:**  `serde_json = { version = "1.0.149", features = ["preserve_order"] }`
- **Lock resolution:** Cargo.lock already at 1.0.149 via caret; manifest pin advanced to match.
- **Breaking:** None (patch-level JSON parser/formatter fixes). `preserve_order` feature preserved.
- **Checks:** `cargo check --workspace --all-targets` exit=0.
- **Tests:** `cargo test -p fsci-runtime --lib` -> 32 passed / 0 failed.
- **Commit:** (next)

### blake3: 1.8.2 -> 1.8.4 (patch pin)

- **Before:** `blake3 = "1.8.2"`
- **After:**  `blake3 = "1.8.4"`
- **Lock resolution:** Cargo.lock already at 1.8.4 via caret; only manifest pin advanced
- **Breaking:** None (patch; hash-function impl bug fixes / perf)
- **Checks:** `cargo check --workspace --all-targets` clean (exit=0)
- **Tests:** `cargo test -p fsci-runtime --lib` -> previously green on asupersync commit; no compilation changes. `cargo test -p fsci-conformance --lib` shows 5 pre-existing failures (quota count 28 vs 23 in differential_*_quota_and_structured_logs + stats_packet_runner_passes) that reproduce on HEAD before my changes -> unrelated to blake3.
- **Commit:** (see git history)

## Failed

(none yet)

## Needs Attention

- **fsci-conformance lib tests (pre-existing, unrelated to this session):** 5 failures in `tests::differential_integrate_quota_and_structured_logs`, `tests::differential_spatial_quota_and_structured_logs`, `tests::differential_stats_quota_and_structured_logs`, `tests::differential_test_stats_fixture`, `tests::stats_packet_runner_passes`. Root cause appears to be a differential-case quota assertion that hasn't been updated after new cases were added (got 28, expected 23). Not introduced by any dep bump in this session; reproduces on HEAD.
- **Criterion bench `black_box` deprecation (low-priority follow-up):** criterion 0.8 still exports `criterion::black_box` but deprecates it. Bench targets across 8 crates emit ~60 warnings total. One-line per bench fix: switch `use criterion::{..., black_box, ...}` to `use std::hint::black_box` (or drop `black_box` from the criterion import and add a `std::hint::black_box` import). Not blocking; didn't touch here to keep the bump focused.
