# Reject: QR-compressed public tall-SVD route

Bead: `frankenscipy-8l8r1.52`

## Lever Tried

The live trial routed tall public `svd`, `svdvals`, `lstsq_with_casp`, and
`pinv_with_casp` through a QR-compressed candidate:

```text
A = Q R
svd(R) = Ur S Vt
U = Q Ur
```

This is algorithmically different, but it is not the selected Pass 2
packed-panel Stage 1 primitive and it changes the public route directly.

## Proof

RCH worker `vmi1167313`:

- `public_qr_compressed_svd_route_matches_safe_svd_reference` passed.
- Local fallback public golden payload SHA stayed
  `1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225`.

The public-golden run still fell back local because no worker was admissible,
so it is not a remote proof.

## Rebench

Same-worker RCH worker `vmi1167313`, public route perf probe:

- Baseline/reference `lstsq`: `117.711395 ms`
- Routed `lstsq`: `124.124343 ms`
- `lstsq` speedup: `0.948334x`
- Baseline/reference `pinv`: `119.219584 ms`
- Routed `pinv`: `117.475517 ms`
- `pinv` speedup: `1.014846x`
- `lstsq_max_abs_diff`: `6.04183370001010189e-13`
- `pinv_max_abs_diff`: `3.60544927247019587e-14`

## Decision

Reject. The route regressed `lstsq` and only gave a marginal `pinv` win, so it
does not clear the `Score >= 2.0` keep gate and does not satisfy the `.52`
packed-panel target. The public route was restored by patch before continuing.
