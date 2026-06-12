# `frankenscipy-8l8r1.91` rejection and next route

## Verdict

Rejected before release benchmarking.

The packed contiguous adjacent-Givens replay attacked the right storage problem
but not the right numerical primitive. It only approached correctness by
increasing the stored diagonal envelope, which turns the candidate back toward
dense storage by degrees. This needs a true implicit symmetric-band bulge
chase, not a wider window around a full-similarity replay.

## Baseline

The live dense-expanded `eig_banded(..., lower=true, eigvals_only=true)`
reference remains the `.90` RCH release probe:

- `256x256`, bandwidth `32`: `66.890671 ms`
- `512x512`, bandwidth `32`: `512.075240 ms`
- Baseline artifact:
  `tests/artifacts/perf/2026-06-11-linalg-8l8r1-90/after_band_to_tridiagonal_perf_probe_rch.txt`
- Baseline artifact sha256:
  `7477b565814151216220cd4dbd983c4a25454943b53c53d6133b5a43cb04ca3e`

The `.91` baseline contract was captured in
`tests/artifacts/perf/2026-06-12-linalg-8l8r1-91/baseline_contract.md`
with sha256
`f5b7ca3d6a9c266f2aff24fbd062dd0705ae7e22ba07334874814ef31975b11a`.

## Candidate Attempt

One lever was attempted: a values-only lower symmetric band reduction using
contiguous upper-band storage plus adjacent Givens rotations.

Rejected route families stayed excluded:

- no sparse map or hash map storage,
- no direct dense expansion on the attempted candidate path,
- no external BLAS/LAPACK/MKL/XLA,
- no unsafe code.

The current checkout retains no `fsci-linalg` source diff from this candidate.

## Proof Result

Focused RCH proof command:

```text
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 CARGO_BUILD_JOBS=1 rch exec -- cargo test -j 1 -p fsci-linalg --lib --locked -- eig_banded_packed_chase --nocapture
```

Worker: `ovh-a`

Result:

- `eig_banded_packed_chase_rejects_invalid_shapes`: passed
- `eig_banded_packed_chase_matches_dense_reference_values`: failed
- failure:
  `packed band chase exceeded workspace by 4.81685870336027183e-11`
- candidate/dense digests before failure:
  - `n=12`, bandwidth `3`: candidate `0x0fdd1b7319713d23`, dense `0xa20d0fb9eab56afb`
  - `n=33`, bandwidth `5`: candidate `0xda9c2057303b2fb4`, dense `0x25e568fa547219a2`

Proof artifact:
`tests/artifacts/perf/2026-06-12-linalg-8l8r1-91/proof_eig_banded_packed_chase_rch.txt`

Proof artifact sha256:
`53c4f7c8f414cdb23b43fcd889960374f1498db93a3ec855d9d64feb0fce94cf`

## Score

No release timing was run because proof failed.

Score:

```text
(Impact 0.0 * Confidence 4.0) / Effort 3.0 = 0.0
```

This is below the keep threshold `2.0`, so the lever is rejected.

## Next Primitive

Work `frankenscipy-8l8r1.92`: true implicit Francis-style symmetric-band
bulge chase for lower-band `eig_banded` values.

Required shift:

- carry the active bulge explicitly,
- apply rotations in the mathematically correct chase order,
- store only the band plus the fixed bulge envelope,
- do not widen a replay window until it becomes dense,
- keep eigenvalue parity against dense reference and public golden digest
  `0x287a5d3679a8bc6a`.
