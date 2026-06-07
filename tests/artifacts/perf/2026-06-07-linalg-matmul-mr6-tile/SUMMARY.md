# GEMM MR=6 Register Tile Rejection

Bead: `frankenscipy-8l8r1.41`

## Profile Target

The target remained the linalg no-gaps GEMM row after `frankenscipy-8l8r1.40`
rejected row-split granularity and restored source.

Fresh pre-edit RCH Criterion baseline:

```text
command: RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked -- --warm-up-time 1 --measurement-time 3 'matmul/1024x1024' --noplot
worker: vmi1149989
matmul/1024x1024 [145.60 ms 155.63 ms 165.84 ms]
```

## Lever Tested

Changed only the private flat-workspace GEMM full-tile kernel from `MR=4` to
`MR=6`, keeping `NR=8`, row ownership, output order, and each output cell's
monotonic `k` accumulation unchanged.

## Proof

Golden SHA proof:

```text
pre:  a9599c4ad5385340abda92b9467cad611efd52f6bdd93436d782a2a4e06e9e5e
post: a9599c4ad5385340abda92b9467cad611efd52f6bdd93436d782a2a4e06e9e5e
```

The SHA was computed from an ignored in-module flat-workspace full-tile payload
test, filtered to the deterministic bit payload and piped to `sha256sum`.
The proof helper was removed after rejecting the lever so source returned to the
pre-pass state.

Focused RCH proof:

```text
command: RCH_FORCE_REMOTE=1 rch exec -- cargo test -p fsci-linalg --release --lib matmul --locked -- --nocapture
worker: vmi1227854
result: ok
passed: matmul_ikj_is_bit_identical_to_naive_ijk
passed: matmul_flat_compute_rows_row_split_is_bit_identical
passed: matmul_flat_workspace_is_bit_identical_to_naive_ijk
passed: matmul_microkernel_is_bit_identical_to_flat_ikj
passed: matmul_microkernel_golden_digest
```

Isomorphism:

- Ordering: row order, column order, and row partitions unchanged.
- Tie-breaking: not applicable.
- Floating point: every cell kept the same monotonic `k` accumulation; tests compare `f64::to_bits`.
- RNG: none.
- Golden output: flat-workspace SHA unchanged.

## Rebench

Two after-run attempts on `vmi1227854` failed before Criterion due worker storage
exhaustion while unpacking crates:

```text
error: No space left on device (os error 28)
```

That worker was drained via `rch workers drain vmi1227854 --yes` so later runs
would avoid the failing worker.

Usable RCH after-run:

```text
command: RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked -- --warm-up-time 1 --measurement-time 3 'matmul/1024x1024' --noplot
worker: vmi1293453
matmul/1024x1024 [114.24 ms 118.26 ms 123.57 ms]
```

The usable after-run was only a cross-worker 1.32x median improvement against the
fresh `vmi1149989` baseline. RCH affinity is disabled locally, so the result did
not provide the same-worker confidence required to keep the lever.

## Decision

Rejected. Score `0.0`.

Source restored to the pre-pass state. Avoid more plain MR widening for this
target. Next GEMM attack should be a deeper exact-order packed-panel/cache
traversal primitive with a same-worker A/B plan.
