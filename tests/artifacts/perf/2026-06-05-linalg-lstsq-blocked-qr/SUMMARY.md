# fsci-linalg lstsq/pinv QR target: profile gate and proof rejection

## Bead

`frankenscipy-275gu` proposed replacing the current `lstsq`/`pinv` SVD path with
blocked Householder QR.

## Baseline

RCH worker: `vmi1227854`

Command:

```text
RCH_FORCE_REMOTE=1 rch exec -- cargo run --release -p fsci-linalg --bin perf_lstsq_probe --locked
```

Captured in `baseline_perf_lstsq_probe_rch.txt`, SHA-256:

```text
3efd966dab504ccafc9783f5523f5a83d5d771850f7ff357925e75be699d0693
```

Rows:

| operation | baseline |
| --- | ---: |
| `lstsq m=2000 n=1000` | `3388.7 ms` |
| `pinv  m=2000 n=1000` | `3350.2 ms` |
| `lstsq m=3000 n=1500` | `20309.9 ms` |
| `pinv  m=3000 n=1500` | `20739.8 ms` |
| `spd-solve n=1024` | `67.0 ms` |
| `spd-solve n=2048` | `271.8 ms` |

The dense rectangular bottleneck is SVD-family work. SPD solve is already on the
in-house blocked Cholesky path and is no longer the residual target.

## Proof Gate

QR alone cannot preserve the current default observable contract:

- `pinv` returns the Moore-Penrose pseudoinverse, which is SVD-defined for the
  current implementation and test surface.
- default `lstsq` returns `singular_values`, and those values are part of
  `LstsqResult`.
- replacing the default rectangular SVD solve with QR would change at least
  floating-point solution bits, certificate action, and likely `singular_values`
  unless an SVD side path remains.
- keeping an SVD side path to preserve observables leaves the measured `20s`
  bottleneck in place, so the proposed QR-only lever does not clear the keep gate
  for this bead.

## Decision

Reject this bead as specified. The correct no-gaps primitive is not a QR-only
substitution for default `lstsq`/`pinv`; it is an in-house safe-Rust SVD-class
kernel: blocked Householder bidiagonalization plus bidiagonal QR/divide-and-
conquer singular-vector reconstruction, using the existing parallel GEMM for
blocked panel/trailing updates.

Target for the follow-up primitive: `>=8x` on the measured `m=3000 n=1500`
`lstsq`/`pinv` rows while preserving output tolerances, rank, singular value
ordering, error classes, certificate semantics, and golden payload hashes.
