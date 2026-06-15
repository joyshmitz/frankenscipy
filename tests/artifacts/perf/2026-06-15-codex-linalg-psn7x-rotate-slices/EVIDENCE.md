# psn7x rotate-column slice helper rejection

Bead: `frankenscipy-psn7x`
Agent: `RubyWaterfall`
Base commit: `1feb36d2`
Date: 2026-06-15

## Profile-backed target

The stage profile in `../2026-06-15-codex-linalg-psn7x-tridiag-profile/` showed
`symmetric_tridiagonal_qr_eigen` as the largest remaining native-eigh stage:
`1089.798 ms` of `2317.048 ms` at `n=1200`.

## Candidate

Rewrote `rotate_eigenvector_columns` to borrow the two adjacent eigenvector columns
as mutable slices instead of using repeated `DMatrix` indexing.

## Behavior proof

RCH proof `rotate_eigenvector_columns_matches_indexed_update_bits` passed. It
compared the slice implementation against the old indexed update using
`f64::to_bits()` for every element.

Golden proof artifact SHA-256: `27c2c742b92969b9d0b7f4f40332d5eea1de3f89e73e5da0fc889a56d5067951`.

## Same-worker timing

RCH worker: `vmi1152480`

| size | baseline native (`1feb36d2`) | candidate native | result |
| --- | ---: | ---: | ---: |
| n=400 | 137.0 ms | 134.9 ms | 1.02x |
| n=800 | 662.8 ms | 1018.7 ms | 0.65x |
| n=1200 | 2191.3 ms | 3603.8 ms | 0.61x |

Score: 0.0. Reject.

## Source state

The candidate source hunk and its proof test were removed after the timing gate.
`git diff -- crates/fsci-linalg/src/lib.rs` was empty after restore.

## Route

Do not retry slice/index spelling for `rotate_eigenvector_columns`. The next
tridiagonal-eigensolver pass should use a structurally different primitive, such as
blocked/batched Givens accumulation, divide-and-conquer tridiagonal eigensolve, MRRR-style
representation work, or another algorithm that reduces the eigenvector-update cost
rather than micro-tuning the per-rotation loop.

## Artifacts

- `proof_rotate_slices_bits_rch.txt`
- `after_rotate_slices_timing_vmi1152480_rch.txt`
- `evidence_checksums.sha256`
