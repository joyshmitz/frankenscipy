# frankenscipy-5wa2y Householder Reprofile Evidence

## Verdict

`frankenscipy-5wa2y` was closed as profile-backed routing evidence, not as a production source keep. The current native `eigh` route is no longer dominated by tridiagonal eigenvector construction after `psn7x`; the measured next frontier is the dense Householder reduction / banded integration path.

## Baseline

Fresh RCH public route baseline on `vmi1227854`:

| n | routed native | nalgebra | native speedup |
|---:|---:|---:|---:|
| 400 | 40.850222 ms | 42.489900 ms | 1.040139x |
| 800 | 298.144211 ms | 399.979928 ms | 1.341565x |
| 1200 | 817.526368 ms | 1583.132033 ms | 1.936490x |

The historical stage-breakdown probe name was absent in the current tip; the zero-test transcript is retained only as a failed probe command.

## Candidate Probe

The compact-WY full-to-band replay probe passed on `ovh-a`:

| n | scalar replay | compact-WY replay | speedup |
|---:|---:|---:|---:|
| 256 | 10.070524 ms | 7.278883 ms | 1.383526x |
| 512 | 103.929563 ms | 68.890608 ms | 1.508617x |

The probe showed a viable structural primitive, but the current `eig_banded` eigenvector route expanded band storage back to dense and entered the scalar dense Householder path. Wiring only full-to-band into public `eigh` would therefore have been an incomplete half-route.

## Route

Closed `frankenscipy-5wa2y` and created `frankenscipy-ql7n6` for the complete two-stage route: compact-WY full-to-band, band-to-tridiagonal, and eigenvector back-transform.
