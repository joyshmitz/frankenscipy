# Direct Public Bidiag Pseudo-Inverse Rejection

Bead: `frankenscipy-8l8r1.45`

## Target

Fresh public-route control after `frankenscipy-8l8r1.43` restored source:

- `lstsq/512x256`: `[96.219 ms 98.053 ms 99.988 ms]`
- `pinv/512x256`: `[97.281 ms 98.101 ms 99.033 ms]`

The trial targeted `DeterministicThinSvd::pseudo_inverse`, replacing the dense
`Vt.T * Sigma_pinv * U.T` construction with a direct diagonal-aware fill.

## Proof Gate

RCH worker: `vmi1156319`

Command:

```text
cargo test -p fsci-linalg --release --lib --locked thin_bidiag_direct_pseudo_inverse_matches_dense_sigma_reference -- --nocapture
```

Result:

```text
FAILED: pseudo-inverse drift at (5, 0)
left:  13802542400818801329
right: 13802542400818801328
```

The direct fill changed a public pseudo-inverse bit by one ULP against the dense
sigma reference before any benchmark keep gate was reached.

## Decision

Rejected. Source was restored to the dense sigma implementation; no production
code from this trial remains.

Score: `0.0 = proof failed`.

Next direction: do not retry direct pinv assembly unless the proof is reframed
around a bit-identical diagonal scaling plus the existing matrix multiply, or a
public-golden migration is explicitly approved. Continue `frankenscipy-8l8r1.44`
on the deeper bidiagonal reduction/replay primitive.
