# Rejected lever: row-slice DMatrix materialization

Bead: `frankenscipy-8l8r1.93`
Date: 2026-06-13
Worker: `vmi1149989`

## Target

Profile-backed determinant lane: `cargo bench -p fsci-linalg --bench linalg_bench -- det/256x256 --sample-size 20`.

## Lever

Changed `dmatrix_from_rows` from direct column-major `DMatrix::from_vec` construction to row-contiguous `extend_from_slice` plus `DMatrix::from_row_slice`.

This was a single materialization lever intended to improve row-read locality before determinant LU setup.

## Behavior proof

Commands:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1149989 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo test -j 1 -p fsci-linalg --lib --locked -- dmatrix_from_rows_preserves_entry_bits --nocapture
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1149989 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo test -j 1 -p fsci-linalg --lib --locked -- det_matches --nocapture
```

Results:

- `dmatrix_from_rows_preserves_entry_bits`: passed.
- `det_matches_scipy_reference_values`: passed.
- Entry ordering and bit identity were preserved by the proposed helper change.
- No floating-point operation ordering changed downstream; determinant parity still matched the existing SciPy reference-value test.
- No RNG-dependent path was touched.

## Benchmark

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1149989 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo bench -j 1 -p fsci-linalg --bench linalg_bench -- det/256x256 --sample-size 20
```

Same-worker Criterion intervals:

- Baseline: `[1.0959 ms 1.1294 ms 1.1721 ms]`
- After: `[1.5069 ms 1.5675 ms 1.6225 ms]`

Median delta: `+38.8%` slower.

## Decision

Rejected. Score is below the `2.0` keep threshold because impact is negative despite high confidence. The source change was reverted; only this evidence bundle remains.

