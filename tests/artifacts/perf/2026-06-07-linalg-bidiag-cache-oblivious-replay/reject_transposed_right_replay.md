# Transposed Right-Reflector Replay Rejection

Bead: `frankenscipy-8l8r1.47`

## Target

Continue the thin bidiagonal SVD factor-replay path after the left column-slice
trial failed. The narrow hypothesis was that applying right reflectors through a
transposed column-major `V` buffer could improve locality while preserving the
same right-reflector order.

## Baseline

Prior same-worker RCH baseline on `ts1` for the same current replay path:

- Probe: `thin_bidiag_factor_replay_perf_probe`
- Shape: `1024x512`
- Current reflector replay: `250.248466 ms`
- Replay digest: `0x8f521a39638fb520`
- RCH artifact: `tests/artifacts/perf/2026-06-07-linalg-public-bidiag-route/bidiag_stage_probes_baseline_rch.txt`

## Lever Tried

Replace direct right-Householder updates over `Vt` with:

1. transpose `Vt` into column-major `V`
2. replay the same right reflectors over columns with `apply_householder_left`
3. transpose back into `Vt`

Left-reflector replay, singular-value ordering, sign canonicalization,
rank-threshold behavior, public route selection, error behavior, and RNG absence
were unchanged.

## Proof

RCH `ts1` focused bit proof passed:

- `thin_bidiag_transposed_right_replay_matches_direct_right_bits`
- Singular values, `U`, and `Vt` entries compared by `f64::to_bits`

Public SVD/lstsq/pinv golden payload SHA-256 stayed unchanged:

- Before: `1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225`
- After: `1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225`

## Rebench

Same-worker RCH `ts1` after the trial:

- Dense-product reference: `529.934294 ms`
- Transposed right replay: `278.751183 ms`
- Replay digest: `0x8f521a39638fb520`

Against the current `ts1` baseline replay time, the candidate regressed:

- Baseline replay: `250.248466 ms`
- Candidate replay: `278.751183 ms`
- Speedup: `0.897028x`

## Decision

Rejected. The candidate preserved bits and public golden output, but it
regressed the measured target by about `16%`, so it does not clear the Score
`>= 2.0` keep gate. Source is restored; no production code from this trial
remains.

Score: `0.0`.

Next primitive: reusable scratch-resident block-reflector replay buffers for
thin `U`, or a two-stage communication-avoiding bidiagonal reducer if the
replay-only route fails.
