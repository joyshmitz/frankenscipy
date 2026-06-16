# `frankenscipy-psn7x.4` row-wise lower-storage p-vector rejection

## Target

Profile-backed target: native symmetric `eigh` Householder reduction. The fresh
stage split in `../2026-06-15-rubywaterfall-linalg-psn7x4-stage-profile/`
showed reduction still dominates at large dense shapes after the accepted
rank-2/lower-storage keeps.

## Candidate

Temporary source lever: change only
`apply_symmetric_householder_trailing_rank2_lower_storage` so the
`p = tau * A_active * v` build walks each output row in the same column
accumulation order, reading the lower-storage active suffix row-wise instead of
the production column-wise traversal.

The candidate was behavior-clean but slower. Source was restored to zero diff.

## Baseline

RCH selected `vmi1152480`.

Public route baseline:

- 400x400 routed public `eigh`: `194.405749 ms`, digest `0x4b8334c92ce624eb`
- 800x800 routed public `eigh`: `594.388431 ms`, digest `0xad8a7e5fa1980bfb`
- 1200x1200 routed public `eigh`: `1928.816964 ms`, digest `0x181b3486089d0e4a`

## Proof

`proof_lower_storage_rowwise_p_bits_rch.txt` passed:

- `symmetric_rank2_lower_storage_matches_full_update_lower_bits`
- proves the candidate preserved `p`/`w` and stored lower-triangle bits against
  the full symmetric rank-2 update reference.
- Ordering, tie-breaking, RNG, public dispatch, validation, and trace behavior
  were not changed.

## Rebench

The public after probe was too noisy for a keep decision because the nalgebra
side-probe also moved materially. A same-binary A/B probe was added temporarily
and run through RCH on `vmi1152480`, comparing the production column-wise
lower-storage traversal against the row-wise candidate in one test process:

| shape | column-wise | row-wise | ratio |
|---:|---:|---:|---:|
| 400x400 | `19.452080 ms` | `50.530773 ms` | `0.384955x` |
| 800x800 | `139.423459 ms` | `211.790696 ms` | `0.658308x` |

Lower-triangle digests from the candidate proof probe:

- 400x400: `0xa62877810f595a68`
- 800x800: `0x400e567a3f1f6eb5`

## Verdict

Reject. Score: `Impact 0.0 * Confidence 4.0 / Effort 1.0 = 0.0`.

Do not retry row-wise lower-storage p traversal. The next admissible primitive
must change the reduction structure more deeply: blocked active-suffix tiles,
contiguous packed column panels with exact scalar p/w proof, or the two-stage
dense-to-band plus band-to-tridiagonal/bulge-chasing route.
