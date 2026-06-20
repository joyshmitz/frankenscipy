# Label-Stats Dense Mean Lookup Evidence

- Date: 2026-06-20
- Agent: cod-a / MistyBirch
- Bead: `frankenscipy-klb7o`
- Decision: KEEP as an internal win; residual SciPy loss remains routed deeper.

## Lever

`ndimage.mean(input, labels, index)` now avoids the per-element `HashMap`
probe when the requested labels are compact non-negative integers. It builds a
dense `Vec<usize>` label-to-first-position table with a sentinel, streams the
labels through integer array lookup, and preserves the existing `HashMap`
fallback for sparse/huge indexes.

The dense path keeps the observable label semantics pinned by the previous
flat route: duplicate `index` entries keep the first position, `-0.0` matches
`0`, fractional/negative/non-finite labels do not match unsigned indexes, and
empty groups still return `NaN`.

## Same-Host A/B

Command:

```text
/data/projects/.rch-targets/frankenscipy-cod-a/release/perf_label_stats
```

The binary compares the old O(N*K) linear scan, the previous O(N+K) bucketed
route, the previous flat `HashMap` sum/count route, and the current dense route
in one optimized executable. `mism=0/0/0` means bit-identical output against all
three prior routes.

| N | K | old O(N*K) | bucketed O(N+K) | hashflat O(N+K) | dense O(N+K) | hash/dense | mismatches |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 65536 | 512 | 14.275 ms | 905.391 us | 625.594 us | 301.407 us | 2.08x | 0/0/0 |
| 262144 | 1024 | 99.769 ms | 3.688 ms | 2.652 ms | 1.263 ms | 2.10x | 0/0/0 |
| 262144 | 2048 | 209.063 ms | 4.015 ms | 2.798 ms | 1.383 ms | 2.02x | 0/0/0 |
| 589824 | 4096 | 931.944 ms | 10.575 ms | 6.772 ms | 3.351 ms | 2.02x | 0/0/0 |

Internal win/loss/neutral versus the prior flat `HashMap` route: `4/0/0`.

## rch Release A/B

Command:

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo run --release -p fsci-ndimage --bin perf_label_stats
```

Worker: `hz2`.

| N | K | hashflat O(N+K) | dense O(N+K) | hash/dense | mismatches |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 65536 | 512 | 690.229 us | 345.794 us | 2.00x | 0/0/0 |
| 262144 | 1024 | 2.922 ms | 1.421 ms | 2.06x | 0/0/0 |
| 262144 | 2048 | 3.495 ms | 1.410 ms | 2.48x | 0/0/0 |
| 589824 | 4096 | 7.207 ms | 3.249 ms | 2.22x | 0/0/0 |

## SciPy Oracle

Command:

```text
python3 docs/perf_oracle_label_stats.py
```

Local SciPy oracle output:

```text
scipy.ndimage.mean(labels, index) over K labels, reps=15
  N=  65536 K=  512  p50 =      0.158 ms
  N= 262144 K= 1024  p50 =      0.538 ms
  N= 262144 K= 2048  p50 =      0.550 ms
  N= 589824 K= 4096  p50 =      1.271 ms
```

| N | K | Rust dense | SciPy p50 | Rust vs SciPy | Verdict |
| ---: | ---: | ---: | ---: | ---: | --- |
| 65536 | 512 | 301.407 us | 158 us | 1.91x slower | loss |
| 262144 | 1024 | 1.263 ms | 538 us | 2.35x slower | loss |
| 262144 | 2048 | 1.383 ms | 550 us | 2.51x slower | loss |
| 589824 | 4096 | 3.351 ms | 1.271 ms | 2.64x slower | loss |

SciPy win/loss/neutral for final source: `0/4/0`.

## Guards

- PASS: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo test -p fsci-ndimage --lib -- --nocapture`
  - `241 passed; 0 failed`.
- PASS: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo check -p fsci-ndimage --all-targets`
  - existing warnings remain in `fsci-interpolate` and `diff_geom`; no new error.
- PASS: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo test -p fsci-conformance ndimage -- --nocapture`
  - conformance lib-side ndimage filter: `5 passed; 0 failed`; integration `diff_ndimage` failed on rch only because worker `hz2` lacked SciPy while `FSCI_REQUIRE_SCIPY_ORACLE=1`.
- PASS: `FSCI_REQUIRE_SCIPY_ORACLE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo test -p fsci-conformance --test diff_ndimage -- --nocapture`
  - local live SciPy-backed ndimage oracle: `5 passed; 0 failed`.
- PASS: `rustfmt --edition 2024 --check crates/fsci-ndimage/src/lib.rs crates/fsci-ndimage/src/bin/perf_label_stats.rs`.
- PASS: `git diff --check -- crates/fsci-ndimage/src/lib.rs crates/fsci-ndimage/src/bin/perf_label_stats.rs`.
- PASS: `ubs crates/fsci-ndimage/src/lib.rs crates/fsci-ndimage/src/bin/perf_label_stats.rs`
  - exit 0; no critical issues; broad existing warning inventory left untouched.
- BLOCKED: `cargo clippy -p fsci-ndimage --lib --bin perf_label_stats -- -D warnings`
  - stopped before this crate on existing `fsci-linalg` dependency lints.
- BLOCKED: `cargo clippy -p fsci-ndimage --lib --bin perf_label_stats --no-deps -- -D warnings`
  - reached `fsci-ndimage`, but failed on existing unrelated lib-file lints (`type_complexity`, `needless_range_loop`, `too_many_arguments`, `collapsible_if`) outside this patch.
- BLOCKED: full `cargo test -p fsci-conformance -- --nocapture`
  - unrelated existing failures: missing `P2C-007/contracts/contract_table.json`, Array API tolerance fallback mismatch, and missing `legacy_scipy_code/scipy` on rch worker image; run was interrupted after it stayed silent while still occupying the remote slot.

## Negative Evidence

The dense integer-label lookup halves the remaining Rust mean cost on compact
label workloads, reducing the SciPy gap from the prior 3.7-4.7x slower range to
1.9-2.6x slower. It is still a SciPy loss on every measured row.

Do not retry another per-element `HashMap` or grouped-bucket variant for this
workload. The next route needs a deeper constant-factor or algorithmic lever:
sorted-label remapping, fused integer-label generation from `label()`, SIMD
accumulation over contiguous label spans, or cache-tiled sum/count reductions.
