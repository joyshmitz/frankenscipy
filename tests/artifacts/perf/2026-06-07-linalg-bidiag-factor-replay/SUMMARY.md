# Bidiag Factor Replay Workspace Reuse Rejection

Bead: `frankenscipy-8l8r1.44`

## Target

Fresh RCH baseline for the deeper public bidiagonal SVD route after the
acceptance-certificate rejection:

- `BIDIAG_LARGE_REDUCTION_PERF`: `282.843880 ms` on `vmi1149989`
- `THIN_BIDIAG_FACTOR_REPLAY`: dense reference `601.401509 ms`, reflector replay
  `300.625236 ms`, speedup `2.000502x` on `vmi1149989`
- Public SVD/lstsq/pinv golden payload passed on `ts1`

## Lever Tried

Reuse one right-Householder dot workspace while replaying right reflectors during
thin factor assembly. The old path allocated a fresh workspace in
`apply_householder_right` for every right reflector; the trial called
`apply_householder_right_with_workspace` with one persistent buffer.

This preserved the reflector order, dot-product order, update order, singular
values, sign canonicalization, rank threshold behavior, error behavior, and RNG
absence.

## Proof

RCH `vmi1149989`:

- `thin_bidiag_reflector_replay_matches_dense_product_reference`: passed

RCH `ts1` public golden:

- `public_svd_lstsq_pinv_golden_payload`: passed

## Rebench

The decisive same-binary A/B probe compared the allocating reference against the
reused-workspace replay path on `vmi1149989`:

- allocating reference: `296.836556 ms`
- reused workspace: `298.891629 ms`
- speedup: `0.993124x`
- digest: `0x8f521a39638fb520 == 0x8f521a39638fb520`

The probe proved bit identity but showed a small regression, so the change does
not clear the keep gate.

## Decision

Rejected. Source was restored; no production code from this trial remains.

Score: `0.0`.

Next primitive: stop allocator-only replay levers. Continue deeper with a true
two-stage communication-avoiding bidiagonal reducer or a packed panel path that
amortizes reflector formation and far-trailing updates across a full panel.
