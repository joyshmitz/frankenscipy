# fsci-integrate RK45 error/scale fusion: rejected candidate

Bead: frankenscipy-y187y
Date: 2026-06-02
Agent: OliveSnow

## Profile-backed target

Fresh fsci-integrate RCH Criterion profiling ranked `solve_ivp_lorenz_rk45` as
the dominant integrate target in the current pass. The profiled code evidence
was in `crates/fsci-integrate/src/rk.rs`: each adaptive RK attempt builds a
`scale` vector, then builds an `err` vector, then immediately consumes both to
compute the RMS error norm.

The candidate one-lever optimization fused that work into a direct error-norm
calculation over the existing RK stages. It did not change step selection,
accept/reject policy, event handling, dense output, or any public API.

Opportunity score before measurement:

| Hotspot | Impact | Confidence | Effort | Score |
|---------|--------|------------|--------|-------|
| RK45 per-attempt `scale` + `err` allocations | 3 | 3 | 3 | 3.0 |

## Behavior proof

The fused calculation preserved:

- Ordering: per-component accumulation order stayed index order.
- Tie-breaking: step accept/reject comparisons were unchanged.
- Floating-point: the same scale formula, error coefficients, sum order, RMS
  division, and square root were used.
- RNG: N/A.

Golden output was emitted by `perf_integrate golden` and compared by sha256:

- `golden_after.txt`
- sha256: `327b936597b6df9a5eb5d181a7f545a7c946458febea97652c73343781aa1eff`

`sha256sum -c golden_after.sha256` passed, and the RK golden was byte-identical
to the prior solve_ivp no-output golden artifact.

## RCH measurements

The candidate did not produce a reliable same-worker win:

- initial focused baseline, worker vmi1264463: 71.797 us
- scratch HEAD baseline, worker vmi1293453: 29.826 us
- after run, worker vmi1149989: 33.590 us
- after repeat, worker vmi1227854: 36.922 us
- earlier no-production baseline on vmi1149989: 20.445 us

The initial baseline was an outlier relative to later scratch HEAD evidence.
The after-runs were slower than the closest HEAD baseline evidence, and RCH did
not provide a reproducible same-worker before/after pair supporting a keep.

## Decision

Rejected. The code lever was backed out. The profiling harness and artifacts are
kept so future integrate passes can re-baseline cleanly, but no production RK
optimization is committed from this attempt.
