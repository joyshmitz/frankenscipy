# Rejected: Private Blocked DLABRD Reducer Stage 4e

Bead: `frankenscipy-z65tz`

## Lever

Private blocked Golub-Kahan/DLABRD-style reducer with local `X/Y` panels and the
kept fused rank-k far-update helper. The candidate was never wired to public
`svd`, `svdvals`, `lstsq`, or `pinv`, and the source was restored after the
same-worker rejection gate.

## Baseline

RCH `vmi1264463` current Stage 4a workspace probe:

- `elapsed_ms=937.695274`
- `digest=0x90cdd3f8f71ed2c1`
- `first_diagonal=-1.00455335940616146e3`
- `last_diagonal=-6.45492359226604862e1`

The decisive A/B was rerun on the same RCH worker (`ts1`) after source sync:

- Stage 4a workspace: `209.153644 ms`
- Stage 4a digest: `0x90cdd3f8f71ed2c1`

## Behavior Proof

RCH `ts1` focused tests passed for the blocked-private reducer:

- golden digest: `0xece0eac424b2bd86`
- reconstruction error: `6.21516726972970446e-14`
- `Q` orthogonality error: `1.33226762955018785e-15`
- `V` orthogonality error: `9.99200722162640886e-16`
- first diagonal: `-2.48764259408531849e1`
- last diagonal: `-1.60543931124212698e1`

Local focused debug tests also passed. Public route semantics were unchanged
because the candidate stayed private: singular-value ordering, tie-breaking,
rank/rcond thresholds, certificates, error classes, and RNG behavior remained
on the existing public SVD-family path.

## Same-Worker Rebench

RCH `ts1` decisive A/B:

- Stage 4a workspace baseline: `209.153644 ms`
- Blocked-private candidate: `547.385880 ms`
- Candidate digest: `0x2a780dc23cc2fcc3`
- Candidate first diagonal: `-1.00455335940616169e3`
- Candidate last diagonal: `6.45492359226548729e1`
- Speedup: `0.382125x`

## Decision

Reject. The candidate is correct, but it is `2.62x` slower than the current
Stage 4a reducer on the same worker. Per-panel scalar formation, temporary
`DMatrix` allocations, and panel assembly overhead dominate before the fused
far update can amortize work. The factorization digest and last-diagonal sign
also differ from Stage 4a, so public wiring would need a deliberate golden
migration; performance fails first.

Score: `0.3 = Impact 0.5 * Confidence 3 / Effort 5`.

Next primitive: stop scalar DLABRD recurrence attempts and move to a
fundamentally different GEMM-shaped packed/tiled far-update route with reusable
panel buffers, or a two-stage communication-avoiding bidiagonalization path.
Target ratio is at least `2.5x` over Stage 4a with reconstruction and
golden-output proof before public wiring.
