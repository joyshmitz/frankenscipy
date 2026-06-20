# frankenscipy-8l8r1.126 - one-based label mean specialization

Date: 2026-06-20
Agent: cod-b / MistyBirch

## Change

Specialize `ndimage.mean(input, labels, index)` when `index` is exactly
`[1, 2, ..., K]`. This skips dense label-table allocation and maps integer
labels directly to `label - 1` output slots. Arbitrary, duplicate, sparse, and
zero-containing indexes keep the existing dense-table/hash routes.

Alien route: cache/hardware-wall constant reduction. The old compact route was
already O(N + K), so the useful lever was removing per-element table
indirection and a short-lived allocation, not changing complexity.

## Isomorphism Proof

- Ordering preserved: yes. Inputs are scanned in the same element order.
- Tie-breaking unchanged: yes. The specialization only applies when every
  requested label is unique and one-based contiguous.
- Floating-point drift: identical. Each accepted value is added to the same
  output slot in the same order as the dense-table route.
- Label semantics preserved: negative labels, `-0.0`, `0.0`, NaN, out-of-range
  labels, and fractional labels are ignored; exact integer labels in `1..=K`
  are accepted.
- Golden guard: `mean_one_based_contiguous_lookup_preserves_exact_label_semantics`
  passed; `perf_label_stats` reported `mism=0/0/0/0/0`.

## Commands

```bash
AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz2 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo run --release -p fsci-ndimage --bin perf_label_stats
/data/projects/.rch-targets/frankenscipy-cod-b/release/perf_label_stats
python3 docs/perf_oracle_label_stats.py --reps 50
AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz2 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo test -p fsci-ndimage --lib -- --nocapture
FSCI_REQUIRE_SCIPY_ORACLE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo test -p fsci-conformance --test diff_ndimage -- --nocapture
AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo check -p fsci-ndimage --all-targets
```

## Same-Binary A/B

Remote rch `hz2`, release binary, current source reconstructing prior routes:

| N | K | dense table | one-based | Speedup | Mismatches |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 65536 | 512 | 167.797 us | 155.554 us | 1.08x | 0/0/0/0/0 |
| 262144 | 1024 | 667.403 us | 619.586 us | 1.08x | 0/0/0/0/0 |
| 262144 | 2048 | 833.283 us | 614.699 us | 1.36x | 0/0/0/0/0 |
| 589824 | 4096 | 2.005 ms | 1.504 ms | 1.33x | 0/0/0/0/0 |

Same-host transferred release binary:

| N | K | dense table | one-based | Speedup | Mismatches |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 65536 | 512 | 154.810 us | 153.257 us | 1.01x | 0/0/0/0/0 |
| 262144 | 1024 | 646.632 us | 634.996 us | 1.02x | 0/0/0/0/0 |
| 262144 | 2048 | 857.822 us | 687.054 us | 1.25x | 0/0/0/0/0 |
| 589824 | 4096 | 1.782 ms | 1.423 ms | 1.25x | 0/0/0/0/0 |

## SciPy Head-To-Head

Same host, transferred Rust release binary vs local SciPy 1.17.1:

| N | K | Rust one-based | SciPy `ndimage.mean` | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 65536 | 512 | 153.257 us | 0.189 ms | Rust 1.23x faster |
| 262144 | 1024 | 634.996 us | 0.585 ms | Rust 1.09x slower |
| 262144 | 2048 | 687.054 us | 0.576 ms | Rust 1.19x slower |
| 589824 | 4096 | 1.423 ms | 1.380 ms | Rust 1.03x slower |

Strict SciPy score: `1/3/0`.
Internal same-host score versus prior dense table: `4/0/0`.

## Gates

- PASS: focused one-based semantics test via rch `hz2`, 1 passed / 0 failed.
- PASS: full `fsci-ndimage --lib` via rch `hz2`, 242 passed / 0 failed.
- PASS: local live SciPy `diff_ndimage` conformance, 5 passed / 0 failed.
- PASS: `cargo check -p fsci-ndimage --all-targets` via rch `hz1`.
- PASS: touched-file rustfmt and `git diff --check`.
- PASS: changed-file UBS exited 0 with 0 critical issues; warning inventory
  remains non-blocking.
- BLOCKED: `cargo clippy -p fsci-ndimage --all-targets -- -D warnings`
  stops before this patch on existing `fsci-linalg` lints:
  `needless_range_loop` and `needless_borrow`.

## Decision

KEEP. The first measured row now beats SciPy and the largest-K rows cut the
remaining loss to near parity. Do not retry table-probe micro-variants for this
workload without a new profile; next work should target parallel/cache-tiled
label reductions or sorted/run-grouped spans.
