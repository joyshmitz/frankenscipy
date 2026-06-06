# Rejected Stage 4b Scalar DLABRD Panel Trial

Bead: `frankenscipy-z65tz`

## Baseline

- Private reducer baseline before trial: RCH `cargo test -p fsci-linalg --release --lib bidiag_large_reduction_perf_probe --locked -- --ignored --nocapture`
- Worker: `ts1`
- Shape: `1024x512`
- Time: `217.359386 ms`
- Digest: `0x90cdd3f8f71ed2c1`

Public guard before trial:

- RCH Criterion worker: `vmi1293453`
- `lstsq/512x256`: `[116.05 ms 117.81 ms 119.56 ms]`
- `pinv/512x256`: `[120.89 ms 123.24 ms 125.67 ms]`

## Trial

The trial implemented a scalar DLABRD-style blocked panel for the private
Golub-Kahan bidiagonal reducer. It accumulated local left/right reflectors and
deferred far-trailing corrections, but the far-trailing update still ran as
scalar per-element correction loops instead of a packed tiled kernel.

## Rejection Gate

- Command: RCH `cargo test -p fsci-linalg --release --lib bidiag_blocked_panel_perf_probe --locked -- --ignored --nocapture`
- Worker: `vmi1293453`
- Shape: `1024x512`
- Stage 4a workspace time: `185.033564 ms`
- Scalar blocked-panel time: `3345.262081 ms`
- Speedup: `0.055312x`
- Workspace digest: `0x90cdd3f8f71ed2c1`
- Blocked digest: `0xf668fb53ab94b5b1`
- Blocked reconstruction error: `4.91944263103505364e-10`

## Decision

Rejected with score `0.15 = Impact 1 * Confidence 3 / Effort 20`.

The source was restored to the Stage 4a workspace reducer. The next attempt must
be algorithmically different: packed block reflector panels with fused tiled
far-trailing updates for `V*Y^T + X*U^T`, or an equivalent safe-Rust
communication-avoiding primitive. Do not retry the scalar recurrence.
