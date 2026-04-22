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
