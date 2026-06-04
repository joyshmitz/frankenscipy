# Sparse Diags Direct CSR Evidence

Bead: `frankenscipy-qaqai`

Target: `sparse_diags/tridiag/10000`, selected from the sparse conversion profile after no ready perf bead was available.

## One Lever

`fsci_sparse::construct::diags` now builds validated unique-offset diagonals directly into canonical CSR rows instead of materializing COO triplets and converting COO to CSR.

## Benchmarks

All benchmark commands were crate-scoped and run through RCH.

- Baseline: `cargo bench -p fsci-sparse --bench sparse_bench --locked -- sparse_diags/tridiag/10000 --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot`
  - Worker: `vmi1152480`
  - Median: `2.1526 ms`
- After: same command
  - Worker: `ts2`
  - Median: `234.24 us`
- After confirm: same command
  - Worker: `ts2`
  - Median: `235.15 us`

Conservative confirmed speedup: `2.1526 ms / 235.15 us = 9.15x`.

Score: `Impact 9.15 * Confidence 0.95 / Effort 1.5 = 5.80`, keep.

## Isomorphism Proof

- Duplicate offset rejection is still checked before construction.
- Shape inference still uses the existing `infer_shape` path.
- Explicit shape bounds still reject with `diagonal length exceeds matrix shape bounds`.
- CSR observable order is preserved: the old COO path canonicalized by row and column; the direct path iterates rows and sorts each row by column.
- Explicit zero values are preserved.
- Floating-point values are copied without arithmetic, so bit patterns are unchanged.
- There is no RNG, ordering tie-break, or nondeterministic surface in this function.

Golden output before and after:

- `golden_before.txt` SHA256: `4dd46b1521c932ceac96e8599d333c8565a9995ff1ee01c92ae27b191b0bee48`
- `golden_after.txt` SHA256: `4dd46b1521c932ceac96e8599d333c8565a9995ff1ee01c92ae27b191b0bee48`
- `golden_before_after.diff`: empty

## Validation

- `cargo fmt -p fsci-sparse --check`: pass
- `RCH_FORCE_REMOTE=1 rch exec -- cargo check -p fsci-sparse --all-targets --locked`: pass
- `RCH_FORCE_REMOTE=1 rch exec -- cargo test -p fsci-sparse --lib --locked`: pass, `307 passed`
- `RCH_FORCE_REMOTE=1 rch exec -- cargo clippy -p fsci-sparse --all-targets --locked -- -D warnings`: pass
- `ubs --ci --only=rust crates/fsci-sparse/src/construct.rs crates/fsci-sparse/src/bin/perf_sparse.rs crates/fsci-sparse/src/lib.rs`: exit 0
