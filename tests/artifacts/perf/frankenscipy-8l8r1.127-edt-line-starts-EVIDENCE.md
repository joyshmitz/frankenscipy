# frankenscipy-8l8r1.127 EDT Line-Start Evidence

- Date: 2026-06-20
- Agent: cod-b / MistyBirch
- Crate: `fsci-ndimage`
- Decision: KEEP as a measured same-worker internal win; residual SciPy score is
  `1/3/0`, so route deeper rather than declaring the EDT cluster fully done.

## Lever

The prior `distance_transform_edt(return_indices=True)` feature-transform route
closed the algorithmic O(foreground * background) gap but still scanned every
flat index to discover separable line starts and allocated coordinates while
materializing output indices.

This patch:

- enumerates exact axis line starts by block/offset;
- uses flat row/column arithmetic for 2-D `return_indices`;
- reuses a coordinate scratch buffer for generic ndim.

The separable EDT math, feature-index propagation, tie behavior, and fallback
paths are unchanged.

## Commands

Baseline Rust:

```bash
AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo run --release -p fsci-ndimage --bin perf_edt
```

Final Rust:

```bash
AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1152480 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo run --release -p fsci-ndimage --bin perf_edt
```

SciPy oracle:

```bash
python3 docs/perf_oracle_edt_indices.py --reps 20
```

## Benchmark

All Rust rows are from rch worker `vmi1152480`. SciPy rows are local SciPy
1.17.1 because the rch worker image does not consistently provide SciPy.

| Image | Prior Rust `feat` | Final Rust `feat` | Internal speedup | SciPy oracle | Final vs SciPy |
| --- | ---: | ---: | ---: | ---: | --- |
| 64x64 | 325.742 us | 216.733 us | 1.50x | 173.434 us | 1.25x slower |
| 128x128 | 1.380 ms | 1.207 ms | 1.14x | 775.685 us | 1.56x slower |
| 192x192 | 3.814 ms | 2.107 ms | 1.81x | 2.280155 ms | 1.08x faster |
| 256x256 | 5.854 ms | 4.855 ms | 1.21x | 4.288605 ms | 1.13x slower |

Score:

- Same-worker internal keep/loss/neutral: `4/0/0`.
- Strict SciPy win/loss/neutral: `1/3/0`.

Secondary distance-only rows improved from `259.977 us -> 234.453 us`,
`1.094 ms -> 1.038 ms`, and `2.206 ms -> 2.119 ms` on the same final run
because the distance-only separable pass also uses exact line-start
enumeration.

## Correctness Gates

- PASS: `perf_edt` printed `0 mismatches / 10876 cells` on baseline and final
  runs, with matching digest rows.
- PASS: `FSCI_REQUIRE_SCIPY_ORACLE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo test -p fsci-conformance --test diff_ndimage_distance_transform_edt -- --nocapture`
  = `1 passed; 0 failed`.
- PASS: `AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo test -p fsci-ndimage distance_transform_edt --lib -- --nocapture`
  = `15 passed; 0 failed`.
- PASS: `AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo test -p fsci-ndimage --lib -- --nocapture`
  = `242 passed; 0 failed`.
- PASS: `AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo check -p fsci-ndimage --all-targets`.
- PASS: `rustfmt --edition 2024 --check crates/fsci-ndimage/src/lib.rs`.
- PASS: `git diff --check -- crates/fsci-ndimage/src/lib.rs`.
- BLOCKED: `cargo fmt -p fsci-ndimage --check` is blocked by pre-existing
  formatting drift in `crates/fsci-ndimage/benches/ndimage_bench.rs` and
  `crates/fsci-ndimage/src/bin/diff_fourier.rs`.
- BLOCKED: `cargo clippy -p fsci-ndimage --all-targets -- -D warnings` stops
  before this patch on existing `fsci-linalg` dependency lints.

## Negative Evidence

Do not retry flat-index line-start filtering or per-cell coordinate allocation
for EDT indices. This lever helped, but the final path still loses to SciPy on
three of four measured rows. Next work should target deeper feature-transform
constants: scratch layout, fused/tiled axis passes, SIMD-friendly
lower-envelope kernels, or a specialized 2-D feature-transform kernel with the
same nearest-background validity proof.
