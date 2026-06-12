# Rejected lever: adjacent-Givens packed-band replay

Bead: `frankenscipy-8l8r1.91`

The attempted source lever was a contiguous packed-band values-only route for
`eig_banded(..., eigvals_only=true)`. It used adjacent Givens rotations over a
fixed-width band workspace. The source was restored to zero diff after the proof
failed.

## Baseline

RCH selected `vmi1152480`.

- Dense-expanded reference from `.90` release A/B:
  - `256x256`, bandwidth 32: `66.890671 ms`
  - `512x512`, bandwidth 32: `512.075240 ms`
- Existing full-to-band stage probe:
  - `256x256`: scalar `10.912319 ms`, compact-WY `7.576195 ms`, `1.440343x`
  - `512x512`: scalar `132.171825 ms`, compact-WY `82.742961 ms`, `1.597378x`
  - public `eigh` side-probe at `512x512`: `127.528251 ms`

## Proof Failure

Artifact: `proof_eig_banded_packed_chase_rch.txt`

The focused proof failed before a release perf gate:

- RCH worker: `ovh-a`
- Fixture digests before failure:
  - `n=12`, bandwidth 3
  - candidate digest `0x0fdd1b7319713d23`
  - dense digest `0xa20d0fb9eab56afb`
  - `n=33`, bandwidth 5
  - candidate digest `0xda9c2057303b2fb4`
  - dense digest `0x25e568fa547219a2`
- Retained proof failure:
  - `ConvergenceFailure { detail: "packed band chase exceeded workspace by 4.81685870336027183e-11" }`

This proves the adjacent-rotation formulation does not preserve the fixed band
envelope. Increasing the workspace width is not an acceptable fix because it
turns the route into dense storage by degrees.

## Verdict

Reject with score `(Impact 0.0 * Confidence 4.0) / Effort 3.0 = 0.0`.

Next bead: `frankenscipy-8l8r1.92`, a true implicit Francis-style
symmetric-band bulge chase that stores only the band plus the active bulge
envelope and applies rotations in chase order.
