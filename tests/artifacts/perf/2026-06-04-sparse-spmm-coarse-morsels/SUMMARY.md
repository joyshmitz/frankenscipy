# SpMM Coarse Row-Morsel Thread Cap Trial

Bead: `frankenscipy-7oswx`
Crate: `fsci-sparse`
Target: `sparse_spmm/2000x2000_d1/2000`

## Profile Context

Fresh RCH sparse reprofile after the rejected row-band, capacity, and epoch
trials ranked `sparse_spmm/2000x2000_d1/2000` first at `13.018 ms` median on
`ts2`.

## Lever Tried

Trialed a parallelization-model change that coarsened SpMM row morsels and
lowered the OS-thread cap. The intended contract was to change only contiguous
row partitioning overhead while preserving row-local arithmetic and emission.

## Baseline

Focused RCH Criterion baseline on `ts2`:

- `12.635 ms` median, CI `[12.509 ms, 12.753 ms]`

## Golden Proof

Strict payload filtering removes cargo harness noise and keeps only deterministic
`n=`, `p...`, and `col:bits` records.

- before strict SHA-256:
  `0728e7d2e4072bf721c19f6b8d0a85a1e064bad0a69d4f10382efbc8ab4c5af2`
- after strict SHA-256:
  `0728e7d2e4072bf721c19f6b8d0a85a1e064bad0a69d4f10382efbc8ab4c5af2`
- strict diff: empty

## Result

Rejected / no source kept.

The after run measured `12.854 ms` median on `ts2`, but a concurrent
stack-workspace SpMM edit appeared in `crates/fsci-sparse/src/linalg.rs` during
the trial, so the timing no longer isolated this thread-cap lever. The
thread-cap line is not present in the working tree. Score is `0.0`.

Next sparse work should continue from a fresh profile after the active
stack-workspace lane resolves, then choose a different GraphBLAS-style SpGEMM
primitive if SpMM remains the dominant row.
