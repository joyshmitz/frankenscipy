# frankenscipy-fa62u - label mean cheap dense probe

Date: 2026-06-20
Agent: cod-b / MistyBirch
Decision: KEEP as an internal constant-factor win. The final same-host
SciPy score remains LOSS on all measured rows: `0/4/0`.

## Lever

`ndimage.mean(input, labels, index)` already had a compact dense integer-label
table from `frankenscipy-klb7o`, but every element still paid
`is_finite() + fract()` before table lookup. This pass changes the dense probe
to:

1. reject unless `0.0 <= label_value < dense_table_len as f64`;
2. cast once to `usize`;
3. require the exact round trip `label as f64 == label_value`;
4. load the dense table sentinel.

This keeps the observable label contract: NaN, fractional, negative, and
out-of-table labels are ignored; duplicate `index` entries keep first-position
semantics; `-0.0` remains accepted as zero.

## Rationale

The alien/profiling pass pointed at constant-factor work in the hot probe:
avoid expensive scalar classification in the per-element path, keep the dense
lookup cache-resident, and prove the older path inside the same binary before
trusting the result. No allocation/layout change was needed for this lever.

## Baseline

Pre-patch rch baseline on worker `hz2`:

```
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
  rch exec -- cargo run --release -p fsci-ndimage --bin perf_label_stats
```

| N | K | shipped dense route |
| ---: | ---: | ---: |
| 65536 | 512 | 367.695 us |
| 262144 | 1024 | 1.407 ms |
| 262144 | 2048 | 1.484 ms |
| 589824 | 4096 | 3.274 ms |

All rows had `mism=0/0/0`.

## Candidate A/B

Candidate rch run on worker `hz2` reconstructed the previous dense
`is_finite()+fract()` path in the benchmark binary and timed it against the new
public `mean` path:

```
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
  rch exec -- cargo run --release -p fsci-ndimage --bin perf_label_stats
```

| N | K | previous dense_fract | new dense_fast | Internal ratio | Mismatches |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 65536 | 512 | 353.495 us | 156.520 us | 2.26x faster | 0/0/0/0 |
| 262144 | 1024 | 1.444 ms | 640.903 us | 2.25x faster | 0/0/0/0 |
| 262144 | 2048 | 1.454 ms | 698.663 us | 2.08x faster | 0/0/0/0 |
| 589824 | 4096 | 3.593 ms | 1.696 ms | 2.12x faster | 0/0/0/0 |

Same-host replay from the transferred rch release binary, paired with the local
SciPy oracle, confirms the internal speedup under current machine load:

| N | K | previous dense_fract | new dense_fast | Internal ratio | Mismatches |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 65536 | 512 | 593.641 us | 278.230 us | 2.13x faster | 0/0/0/0 |
| 262144 | 1024 | 2.350 ms | 1.122 ms | 2.09x faster | 0/0/0/0 |
| 262144 | 2048 | 2.532 ms | 1.186 ms | 2.13x faster | 0/0/0/0 |
| 589824 | 4096 | 6.160 ms | 3.169 ms | 1.94x faster | 0/0/0/0 |

## SciPy Head-To-Head

SciPy oracle:

```
python3 docs/perf_oracle_label_stats.py
```

Rust oracle:

```
/data/projects/.rch-targets/frankenscipy-cod-b/release/perf_label_stats
```

Both were run on the local host because the rch worker image used for
conformance lacked Python SciPy.

| N | K | Rust dense_fast | SciPy ndimage.mean | Ratio | Verdict |
| ---: | ---: | ---: | ---: | ---: | --- |
| 65536 | 512 | 278.230 us | 0.210 ms | 1.33x slower | SciPy loss |
| 262144 | 1024 | 1.122 ms | 0.749 ms | 1.50x slower | SciPy loss |
| 262144 | 2048 | 1.186 ms | 0.751 ms | 1.58x slower | SciPy loss |
| 589824 | 4096 | 3.169 ms | 1.781 ms | 1.78x slower | SciPy loss |

Final SciPy win/loss/neutral: `0/4/0`.

## Gates

| Gate | Result | Notes |
| --- | --- | --- |
| rch release A/B | PASS | `cargo run --release -p fsci-ndimage --bin perf_label_stats` on `hz2`; new dense probe 2.08-2.26x faster, 0 mismatches |
| Local same-host head-to-head | PASS | transferred rch release binary vs local SciPy 1.17.1 oracle; final score `0/4/0` |
| Focused semantics test | PASS | `cargo test -p fsci-ndimage mean_dense_label_lookup_preserves_exact_label_semantics --lib -- --nocapture` via rch |
| Full ndimage lib tests | PASS | `cargo test -p fsci-ndimage --lib -- --nocapture` via rch: 241 passed / 0 failed |
| Per-crate compile | PASS | `cargo check -p fsci-ndimage --all-targets` via rch; unrelated warnings remain in dependencies |
| Local live SciPy conformance | PASS | `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test diff_ndimage -- --nocapture`: 5 passed / 0 failed |
| rch conformance filter | PARTIAL | conformance lib-side ndimage tests passed 5/0, then live `diff_ndimage` failed because worker `hz2` had no `scipy` Python module |
| Formatting | PASS | `rustfmt --edition 2024 --check crates/fsci-ndimage/src/lib.rs crates/fsci-ndimage/src/bin/perf_label_stats.rs` |
| Diff hygiene | PASS | `git diff --check -- crates/fsci-ndimage/src/lib.rs crates/fsci-ndimage/src/bin/perf_label_stats.rs` |
| UBS | PASS | `ubs crates/fsci-ndimage/src/lib.rs crates/fsci-ndimage/src/bin/perf_label_stats.rs`: exit 0, no critical issues; existing warning inventory left untouched |
| Clippy `-D warnings` | BLOCKED | no-deps clippy stops on pre-existing unrelated `fsci-ndimage` lints outside this patch, tracked by existing lint work |

## Negative Evidence

The cheap dense probe removes another large constant from the Rust route, but
SciPy's compiled C path still wins by 1.33-1.78x on the same host. The next
profitable lever is not another scalar probe variant. Route deeper into
parallel/cache-tiled sum/count reductions, vector-friendly integer-label
generation, or sorted/run-grouped label spans.

Retry stop rule: do not retry `fract()`, `is_finite()`, HashMap, or
`Vec<Vec<f64>>` grouping variants for this workload unless a new profile shows
that the hot path moved back there.
