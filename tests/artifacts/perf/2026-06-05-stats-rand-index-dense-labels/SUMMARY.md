# fsci-stats rand_index dense compact-label table

Bead: `frankenscipy-ltdc8`

## Target

Profile-backed hotspot from `cargo bench -p fsci-stats --bench stats_bench`:
`rand_index/k10/8000` was a top `fsci-stats` benchmark at 628.28 us median.

## Lever

Add a bounded dense contingency-table fast path for non-negative compact rounded
labels. The existing pair loop and sparse `HashMap` contingency path remain the
fallback for negative, non-finite, sparse, or too-large label domains.

This preserves:

- label rounding: exactly `round() as i64` before domain checks
- negative label behavior: falls back to the existing sparse path
- sparse large-label behavior: falls back to the existing sparse path
- pair-count formula: same `a`, `same_true`, `same_pred`, `b` inclusion-exclusion
- output ordering and RNG: not applicable, no randomization or output sequence

## Baseline

Command:

```bash
rch exec -- cargo bench -p fsci-stats --bench stats_bench -- --warm-up-time 1 --measurement-time 2 --sample-size 10
```

Worker: `ts2`

| benchmark | before median |
| --- | ---: |
| `rand_index/k10/2000` | 157.67 us |
| `rand_index/k10/8000` | 628.28 us |

## After

Command:

```bash
rch exec -- cargo bench -p fsci-stats --bench stats_bench -- rand_index --warm-up-time 1 --measurement-time 2 --sample-size 10
```

Worker: `ts1`

| benchmark | after median | ratio |
| --- | ---: | ---: |
| `rand_index/k10/2000` | 13.854 us | 11.38x |
| `rand_index/k10/8000` | 53.997 us | 11.64x |

RCH does not expose a worker pin for `exec`; worker split is recorded. The win is
large enough to clear the keep gate despite conservative cross-worker confidence.

## Behavior Proof

Golden command:

```bash
cargo run -p fsci-stats --bin perf_stats -- rand-index-golden
```

Golden payload SHA-256 before and after:

```text
1c0044b96fd00cd1cc40754e53d27920e9745c82fafdfae3f5897f1043b06192
```

Golden cases cover dense compact labels, negative labels, and sparse large
labels. Focused unit coverage compares the dense/fallback path against the
pair-loop oracle.

## Validation

```bash
rch exec -- cargo test -p fsci-stats rand_index --lib -- --nocapture
rch exec -- cargo check -p fsci-stats --all-targets
rch exec -- cargo clippy -p fsci-stats --all-targets -- -D warnings
ubs crates/fsci-stats/src/lib.rs crates/fsci-stats/src/bin/perf_stats.rs
```

Results:

- test: 8 passed, 0 failed
- check: exit 0
- clippy: exit 0
- UBS: no critical findings; warning inventory is pre-existing module-wide debt

`cargo fmt -p fsci-stats --check` was attempted separately and was blocked by
unrelated in-progress `fsci-stats` silhouette edits in the shared worktree. The
rand-index touched code is rustfmt-clean in the UBS shadow check.

## Score

Impact 5 x Confidence 4 / Effort 2 = 10.0. Keep.
