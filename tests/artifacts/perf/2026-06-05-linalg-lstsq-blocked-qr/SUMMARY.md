# fsci-linalg lstsq/pinv QR target: profile gate, rejections, and pinv keep

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

## Rejected Implementation Trials

Two non-ship implementation trials were checked before committing any production
fast path:

| trial | worker | `lstsq 2000x1000` | `pinv 2000x1000` | `lstsq 3000x1500` | `pinv 3000x1500` | decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | `vmi1227854` | `3388.7 ms` | `3350.2 ms` | `20309.9 ms` | `20739.8 ms` | current |
| TSQR composition (`frankenscipy-275gu`) | `vmi1227854` | `4180.8 ms` | `2906.8 ms` | `26739.0 ms` | `19349.7 ms` | reject, `lstsq` regression |
| Gram/eigen SVD (`frankenscipy-jvcdf`) | `vmi1227854` | `5343.9 ms` | `5163.8 ms` | `20454.1 ms` | `20696.4 ms` | reject, no real win |

The Gram/eigen path also had a cross-worker raw capture on `ts2` that regressed
to `24657.3 ms` for `lstsq 3000x1500` and `24948.1 ms` for `pinv 3000x1500`.

Rejected source for both trials was removed.

## Accepted Partial Keep

Artifact: `keep_low_rank_tall_pinv.md`

The profile matrix family is structurally low-rank. A deterministic rank-
revealing tall factorization now routes large low-rank `pinv` inputs through a
compact SVD:

| trial | worker | `lstsq 2000x1000` | `pinv 2000x1000` | `lstsq 3000x1500` | `pinv 3000x1500` | decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| same-worker baseline | `ts2` | `3598.4 ms` | `3793.3 ms` | `21041.5 ms` | `20887.2 ms` | current |
| low-rank tall `pinv` | `ts2` | `3598.1 ms` | `79.1 ms` | `20834.3 ms` | `218.0 ms` | keep, `pinv` `95.81x` at 3000x1500 |

Golden payload SHA-256:

```text
3f67147c7d93c71f778f47d57205c75c28cb0062c30130be28797b17360dde97
```

Score: `8.3 = Impact 5 * Confidence 5 / Effort 3`.

Default `lstsq` is not routed through this low-rank shortcut because its public
result includes singular values and threshold behavior that still require the
deeper SVD-class primitive.

## Remaining Primitive

The remaining no-gaps target is in-house blocked Householder bidiagonalization
with compact block reflectors and GEMM-backed trailing updates, followed by a
bidiagonal SVD solver/reconstruction path. The next attempt should not form dense
Gram products and should not compose large nalgebra QR calls.
