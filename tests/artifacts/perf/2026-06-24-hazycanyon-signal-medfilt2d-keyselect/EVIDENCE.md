# fsci-signal medfilt2d key-select probe

- Date: 2026-06-24
- Agent: HazyCanyon
- Crate: `fsci-signal`
- Probe: `scipy.signal.medfilt2d` parity/perf, deterministic 256x256 `f64` image, 7x7 kernel.
- Decision: reject the sortable-key selector lever; keep only the head-to-head benchmark and ledger entry.

## Worktree Sweep

Checked non-ancestor `.scratch`/`.worktrees` FrankenSciPy heads before starting a new lever. The measured
heads found were already represented on `main` or stale relative to newer mainline code; no unlanded
measured win was safe to land.

## Lever Tested

`medfilt` already selects finite `f64` samples through sortable `u64` keys. I tested the same idea in
`medfilt2d`: store each window as sortable keys and call integer `select_nth_unstable` instead of
`select_nth_unstable_by(..., f64::total_cmp)`.

The algorithm patch was reverted after measurement.

## Commands

All Cargo commands were crate-scoped and used:

```text
AGENT_NAME=HazyCanyon
RCH_REQUIRE_REMOTE=1
RCH_WORKER=ovh-b
RCH_TEST_SLOTS=1
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-p2
```

Baseline/current Rust bench:

```text
rch exec -- env CARGO_BUILD_JOBS=1 cargo bench -j 1 -p fsci-signal --bench signal_bench -- medfilt2d_gauntlet_scipy --noplot
```

Rejected candidate bench used the same command after applying the key-select patch.

Post-revert conformance:

```text
rch exec -- env CARGO_BUILD_JOBS=1 cargo test -j 1 -p fsci-signal medfilt2d --lib -- --nocapture
```

Same-worker SciPy comparator used an `ovh-b` venv:

```text
/tmp/frankenscipy-scipy-hazycanyon
numpy 2.5.0
scipy 1.18.0
```

and prepended that venv to `PATH` for the final Criterion run.

## Measurements

Worker: `ovh-b`.

| Variant | Median | Ratio |
| --- | ---: | ---: |
| Baseline/current Rust, initial run | 20.919 ms | 1.00x |
| Key-select candidate Rust | 31.537 ms | 0.66x baseline speed |
| Current Rust, same-machine SciPy run | 20.902 ms | 1.19x SciPy speed |
| SciPy `signal.medfilt2d` | 24.782 ms | 1.00x SciPy |

Criterion reported the key-select candidate as a regression:

```text
time:   [31.383 ms 31.537 ms 31.707 ms]
change: [+50.398% +51.821% +53.234%] (p = 0.00 < 0.05)
Performance has regressed.
```

Final head-to-head after reverting the candidate:

```text
medfilt2d_gauntlet_scipy/256x256_k7_rust
time:   [20.737 ms 20.902 ms 21.109 ms]

medfilt2d_gauntlet_scipy/256x256_k7_scipy
time:   [24.309 ms 24.782 ms 25.256 ms]
```

## Landing Verification

GreenFalcon re-ran the focused conformance and head-to-head bench before landing the evidence bundle.

```text
AGENT_NAME=GreenFalcon
RUSTUP_TOOLCHAIN=nightly-2026-06-10
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a
```

RCH conformance:

```text
rch exec -- cargo test -p fsci-signal medfilt2d --lib -- --nocapture

worker: vmi1149989
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 645 filtered out
```

Local SciPy comparator bench:

```text
cargo bench -p fsci-signal --bench signal_bench -- medfilt2d_gauntlet_scipy --noplot

medfilt2d_gauntlet_scipy/256x256_k7_rust
time:   [14.355 ms 14.763 ms 15.217 ms]

medfilt2d_gauntlet_scipy/256x256_k7_scipy
time:   [18.582 ms 19.421 ms 20.450 ms]
```

Fresh landing ratio: Rust is 1.32x faster than SciPy on this comparator row.

## Conformance

Post-revert focused test gate:

```text
running 3 tests
test tests::medfilt2d_rejects_non_finite_samples ... ok
test tests::medfilt2d_matches_scipy ... ok
test tests::medfilt2d_rejects_overflowing_kernel_area ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 645 filtered out
```

## Conclusion

Do not retry the `medfilt2d` sortable-key window lever. On this 7x7 2-D workload, the extra key conversion
dominates and loses badly despite removing `total_cmp` from selection. The current `f64::total_cmp` path is
already faster than SciPy for this probe.
