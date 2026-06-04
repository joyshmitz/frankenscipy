# Sparse Direct Transpose Unchecked Construction

Bead: `frankenscipy-ieq3m`

Verdict: kept.

## Profile Target

The post-CSR-metadata RCH sparse reprofile ranked direct compressed format conversion as the next sparse hotspot:

- `sparse_format_conversion/10000x10000_d0_csr_to_csc/10000`: `1.0744 ms` median on `ts2`.
- `sparse_format_conversion/10000x10000_d0_csc_to_csr/10000`: `1.0486 ms` median on `ts2`.

## Lever

For canonical compressed inputs, `can_direct_transpose_compressed` already validates the source index range, row/column ordering, and deduplication before the direct counting transpose is used. The transpose loop then constructs valid `indptr`, `indices`, and `data` arrays by counting each minor index and filling the output in source major-order.

This pass keeps the fallback COO path unchanged and changes only the direct path result construction:

- `csr_to_csc_direct` now constructs the CSC result without re-running `validate_compressed`.
- `csc_to_csr_direct` now constructs the CSR result without re-running `validate_compressed`.
- Both direct results set `CanonicalMeta { sorted_indices: true, deduplicated: true }` because the precondition and counting transpose establish that invariant.

## Isomorphism Contract

- Conversion eligibility and fallback behavior are unchanged.
- Sparse element order in direct output is unchanged.
- Duplicate handling is unchanged; only canonical inputs enter this direct path.
- Floating-point values are copied, not recomputed.
- No RNG, tie-breaking, approximation, or global state is involved.
- Hardened-mode rejection behavior for unsorted compressed inputs is unchanged.

## Golden Proof

Before and after RCH `conversion_golden_snapshot` payloads are byte-identical:

```text
f01e261f50d39eab13c364c8af2dee85d335ad78729e18d2014dfa17450d2efe
```

The focused conversion test run passed 10 tests, including the golden snapshot, dense-preservation roundtrips, shape preservation, and hardened conversion metadata/rejection tests.

## Rebench

Same-worker RCH comparison against the `ts2` post-pass-1 reprofile:

```text
sparse_format_conversion/10000x10000_d0_csr_to_csc/10000
baseline: 1.0744 ms
after:    808.71 us
delta:    1.33x, 24.7% lower median

sparse_format_conversion/10000x10000_d0_csc_to_csr/10000
baseline: 1.0486 ms
after:    841.34 us
delta:    1.25x, 19.8% lower median
```

Score: `6.0 = impact 3 * confidence 4 / effort 2`.

## Validation

- `cargo fmt -p fsci-sparse --check`: passed.
- `ubs crates/fsci-sparse/src/formats.rs crates/fsci-sparse/src/ops.rs`: no critical findings; warnings are existing sparse-code inventory.
- `rch exec -- cargo check -p fsci-sparse --all-targets --locked`: passed on `ts2`.
- `rch exec -- cargo clippy -p fsci-sparse --all-targets --locked -- -D warnings`: passed on `ts2`.
- `rch exec -- cargo test -p fsci-sparse --locked conversion -- --nocapture`: passed on `ts2`.
