# frankenscipy-8l8r1.69 pass 7 route

## Fresh profile

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 rch exec -- cargo test -p fsci-linalg --release --lib --locked deterministic_thin_svd_stage_breakdown_probe -- --ignored --nocapture --test-threads=1
```

Worker: `vmi1227854`

```text
reduction_ms=171.279
bidiagonal_svd_ms=86.861
back_transform_u_ms=40.573
back_transform_v_ms=95.990
```

The reduction remains the largest stage. Dense-Gram backend replacement was
rejected and reproduced as slower, so the next route must change the reduction
primitive rather than reusing a Gram/eigen shortcut.

## Rejected families not to repeat

- scalar structural-zero elision
- dense-bidiagonal materialization-only removal
- fused-step thread fanout
- full-rectangle verified-delta packed panel replay
- QR-first / TSQR public replacement
- normal-equation / Gram public route
- one-sided Jacobi public route
- square dense-Gram backend
- backend column-slice micro-tuning

## Selected next primitive

Implement a true DLABRD-style two-sided compact-panel reducer for the square SVD
core:

```text
A22 := A22 - V Y^T - X U^T
```

The rejected packed-panel attempt encoded a full far-rectangle delta and replayed
it as a wide update. The next primitive must instead accumulate narrow panel
state while generating the panel:

- left reflector panel `V`
- right reflector panel `U`
- update partners `Y` and `X`
- one cache-blocked far update over the trailing matrix

## Proof and benchmark contract

- Same-process A/B ignored test comparing current fused-step reduction vs the
  compact-panel reducer on `512x512` and `1024x512`.
- Reconstruction proof: `Q^T A V == B` within the current reduction tolerance.
- Ordering/tie/sign proof: public route gates and SVD ordering remain unchanged.
- Floating point proof: private reduction may differ by rounding order, but
  public golden SHA must remain
  `1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225`.
- Fallback: keep the current fused-step reducer unless the same-worker score is
  at least `2.0`.
- No unsafe and no external BLAS/LAPACK/MKL/XLA.

## EV score

Impact `5`, confidence `3`, effort `5`, score `3.0`.

Target ratio: at least `1.35x` for the reduction stage and at least `2.0` on
campaign score before any production route is retained.
