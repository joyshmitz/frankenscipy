# Pass 1 Baseline Contract

Bead: `frankenscipy-8l8r1.52`

## Commands

- Remote reduction baseline:
  `rch exec -- cargo test -p fsci-linalg --release --lib --locked bidiag_large_reduction_perf_probe -- --ignored --nocapture`
- Public golden anchor:
  `rch exec -- cargo test -p fsci-linalg --release --lib --locked public_svd_lstsq_pinv_golden_payload -- --nocapture`

## Baseline

Remote reduction baseline succeeded on RCH worker `vmi1167313`.

- Probe: `bidiag_large_reduction_perf_probe`
- Shape: `1024x512`
- Time: `414.461569 ms`
- Digest: `0x90cdd3f8f71ed2c1`
- First diagonal: `-1.00455335940616146e3`
- Last diagonal: `-6.45492359226604862e1`

The public golden payload command first fell back local because RCH had no
admissible workers, then a remote-required retry correctly refused local
fallback. The local payload still matches the existing public golden anchor:

- Public golden SHA-256:
  `1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225`

Remote public-golden proof must be rerun before any keep decision.

## Hotspot Target

The target remains the dense `golub_kahan_bidiagonal_reduction` lane used by
the public SVD/lstsq/pinv route. Prior fused-step work preserved exact bits but
only reached `1.064541x`, below the `.51` target, so `.52` must use a
fundamentally different two-stage packed-panel primitive.

## Isomorphism Contract

- Ordering preserved: the current public route constructs left reflectors,
  applies left updates, constructs right reflectors, applies right updates, and
  materializes diagonal/superdiagonal in step order. A kept `.52` route must
  either preserve these externally observable outputs exactly or provide a new
  explicit migration contract with identical public golden payload.
- Tie-breaking unchanged: singular ordering, sign canonicalization, rank
  thresholds, and fallback gates must remain unchanged for public SVD/lstsq/pinv
  outputs.
- Floating-point: reduction digest must stay `0x90cdd3f8f71ed2c1` for an
  exact-order lever. If the two-stage route changes internal reduction order, it
  must still keep public golden SHA and must include a bounded tolerance proof
  for all affected public outputs.
- RNG: unchanged; the target route is deterministic and uses no RNG.
- Golden outputs: public payload SHA must remain
  `1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225`.

## Opportunity Score

Two-stage packed-panel bidiagonalization with safe-Rust GEMM-shaped trailing
updates:

- Impact: `5`
- Confidence: `3`
- Effort: `4`
- Score: `3.75`

This clears the `Score >= 2.0` gate for an implementation attempt. Rejected
families for this hotspot: single-step pass fusion, replay/indexing cleanup,
dense compact-WY composition, scalar DLABRD panel retries, and thread fanout
micro-levers.
