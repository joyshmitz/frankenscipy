# CSR Add Coarse Row-Morsel Thread Cap Trial

Bead: `frankenscipy-0gp7k`
Crate: `fsci-sparse`
Target: `sparse_arithmetic/10000x10000_d0_add/10000`

## Profile Context

Fresh RCH sparse reprofile after the epoch-SpMM rejection ranked
`sparse_arithmetic/10000x10000_d0_add/10000` as the next unclaimed sparse
hotspot at `2.1468 ms` median on `ts2`; the top SpMM row was already claimed by
`frankenscipy-2gmb9`.

## Lever Tried

Trialed one parallelization-model change in `parallel_chunk_count`: cap CSR add
row-block merging at 8 workers with 512 rows per worker instead of 16 workers
with 256 rows per worker.

## Baseline

Focused RCH Criterion baseline:

- `ts1`: `1.5402 ms` median, CI `[1.5135 ms, 1.5679 ms]`

## Golden Proof

- before SHA-256:
  `8391613d472fce793d2f8142226f247d710ff10b7085c3d2efb159c5f177a7e0`
- after SHA-256:
  `8391613d472fce793d2f8142226f247d710ff10b7085c3d2efb159c5f177a7e0`
- payload diff: empty

Ordering, tie behavior, per-row merge order, floating-point operation order,
zero elision, metadata tracking, and RNG absence were unchanged because the row
merge function was not modified.

## Performance Result

Rejected / source restored.

The after runs landed on `ts2`, not the baseline worker:

- first after: `1.6005 ms` median, CI `[1.5737 ms, 1.6279 ms]`
- confirmation: `1.8495 ms` median, CI `[1.8037 ms, 1.8902 ms]`

The evidence is cross-worker and noisy, and it does not prove a real win against
the focused `ts1` baseline. Score is `0.0`.

Restore proof:

- `git diff --quiet -- crates/fsci-sparse/src/ops.rs`: exit `0`
- `cargo fmt -p fsci-sparse --check`: passed after restore

Next CSR-add work should avoid thread-cap tuning and target a different
primitive, such as a behavior-preserving metadata proof that does not require
full structural scans.
