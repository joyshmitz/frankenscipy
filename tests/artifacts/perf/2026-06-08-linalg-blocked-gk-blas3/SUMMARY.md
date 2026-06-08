# Blocked GK BLAS-3 Follow-up: Fused-Step Keep

Bead: `frankenscipy-8l8r1.57`

## Target

Profile-backed residual target after the TSQR rejection: the 1024x512
Golub-Kahan bidiagonal reduction core on the accepted public bidiag SVD route.

## Baseline

- Public route probe, RCH `vmi1152480`:
  - `reference_lstsq_ms=153.659592`
  - `routed_lstsq_ms=70.208542`
  - `reference_pinv_ms=160.874309`
  - `routed_pinv_ms=72.890051`
- Reduction-only probe, RCH `vmi1149989`:
  - `elapsed_ms=234.703792`
  - digest `0x90cdd3f8f71ed2c1`
- Isolated far-update proof kernel, RCH `vmi1152480`:
  - scalar `50.281123 ms`
  - fused `5.382248 ms`
  - speedup `9.342030x`
  - digest `0xd60df77cdefac734`
- Criterion baseline, RCH `vmi1227854`:
  - `lstsq/512x256` `[60.806 ms 63.268 ms 65.412 ms]`

## Lever

Integrated one safe-Rust fused Golub-Kahan step into
`golub_kahan_bidiagonal_reduction`. For every non-final step, the helper:

- computes the left Householder column scales in the same column order;
- updates the current row before forming the right reflector;
- updates the remaining left-tail cells exactly once while accumulating the
  right-reflector dot workspace;
- applies the right reflector with the existing column-major workspace order.

The final left-only column remains on the existing `apply_householder_left`
path. No TSQR, QR-first, thread fanout, C BLAS/LAPACK/MKL/XLA, unsafe code, or
public rank/tolerance policy changed.

## Isomorphism Proof

- RCH `bidiag_fused_step_matches_workspace_reference_bits` passed on
  `vmi1227854`.
- The proof compares every diagonal, superdiagonal, stored bidiagonal cell, and
  every left/right reflector field by `f64::to_bits`.
- Ordering/tie-breaking: public singular ordering and rank-gap guards remain
  unchanged.
- Floating point: the reducer preserves the existing workspace reducer's
  per-cell subtraction order for retained outputs and reflectors.
- RNG: none.
- Public golden SHA stayed
  `1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225`.

## Rebench

Primary same-process A/B, RCH `vmi1227854`:

- old workspace reducer: `297.211246 ms`
- fused-step reducer: `219.828076 ms`
- digest: `0x90cdd3f8f71ed2c1 == 0x90cdd3f8f71ed2c1`
- speedup: `1.352017x`

Supporting post-change probes:

- public route, RCH `vmi1149989`:
  - `routed_lstsq_ms=62.287645`
  - `routed_pinv_ms=71.213729`
  - `lstsq_max_abs_diff=1.07647224467655178e-12`
  - `pinv_max_abs_diff=2.28428387316625958e-14`
- reduction-only, RCH `vmi1152480`:
  - `elapsed_ms=249.312750`
  - digest `0x90cdd3f8f71ed2c1`
- Criterion after, RCH `vmi1149989`:
  - `lstsq/512x256` `[49.608 ms 51.047 ms 52.755 ms]`

## Validation

- `cargo fmt --manifest-path crates/fsci-linalg/Cargo.toml --check`: passed.
- `ubs crates/fsci-linalg/src/lib.rs`: zero critical issues.
- RCH `cargo check -p fsci-linalg --all-targets --locked`: passed.
- RCH `cargo clippy -p fsci-linalg --all-targets --no-deps --locked -- -D warnings`: passed.

## Decision

Keep.

Score: `3.38 = Impact 1.5 * Confidence 4.5 / Effort 2.0`.

Next primitive after reprofile: a fundamentally different packed/two-stage
communication-avoiding bidiagonalization route with reusable panel buffers,
not another scalar DLABRD recurrence.
