# frankenscipy-8l8r1.70 rejection: true compact-panel reducer

## Baseline

Inherited from the `.69` route artifact:

- RCH worker: `vmi1227854`
- Command: `cargo test -p fsci-linalg --release --lib --locked deterministic_thin_svd_stage_breakdown_probe -- --ignored --nocapture --test-threads=1`
- Reduction: `171.279 ms`
- Bidiagonal backend: `86.861 ms`
- U replay: `40.573 ms`
- V replay: `95.990 ms`

Reduction remained the largest stage.

## Lever

Attempted one safe-Rust DLABRD-style compact panel reducer:

- Generate a panel of left and right Householder reflectors.
- Accumulate narrow `V`, `U`, `Y`, and `X` panel state.
- Apply one far update over the trailing matrix with `A22 := A22 - V Y^T - X U^T`.
- Keep the existing fused-step reducer as the same-process reference.

## Proof

RCH proof on `vmi1227854` passed for the temporary compact-panel reducer:

- `bidiag_compact_panel_reconstructs_medium_matrices`: ok
- `512x512` `Q^T A V - B` max abs:
  - fused: `2.91038304567337036e-11`
  - compact: `2.91038304567337036e-11`
- `1024x512` `Q^T A V - B` max abs:
  - fused: `5.45053755322605322e-11`
  - compact: `5.45053755322605322e-11`
- Digests matched the fused-step reference for both measured shapes.

## Same-worker RCH result

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_TEST_SLOTS=1 rch exec -- cargo test -p fsci-linalg --release --lib --locked bidiag_compact_panel_vs_fused_step_perf_probe -- --ignored --nocapture --test-threads=1
```

Worker: `vmi1227854`

```text
512x512:
fused_step_ms=135.001973
compact_panel_ms=513.887213
speedup=0.262707

1024x512:
fused_step_ms=253.606576
compact_panel_ms=1229.373766
speedup=0.206289
```

## Decision

Rejected. The proof was clean, but the attempted compact-panel generation
materialized too many pending low-rank corrections while building the panel.
It failed the Score >= 2.0 keep gate and was not retained as a production route.

Next route: move to a fundamentally different primitive rather than another
compact-panel micro-iteration. The ready `frankenscipy-mifdz` bead targets
two-stage blocked symmetric tridiagonalization, a communication-avoiding
full-to-band then band-to-tridiagonal reducer for dense `eigh`.
