# fsci-opt Powell line-search scratch reuse: rejected candidate

Bead: frankenscipy-okrh6
Date: 2026-06-02
Agent: OliveSnow

## Profile-backed target

Fresh RCH Criterion profiling for `fsci-opt` ranked `powell/rosenbrock/10` as
the highest-latency row in the broad optimize bench matrix:

- `powell/rosenbrock/10`: `[359.18 us, 375.91 us, 401.09 us]`, worker vmi1293453
- `cg/rosenbrock/10`: `[309.30 us, 318.93 us, 328.93 us]`
- `powell/rosenbrock/5`: `[82.172 us, 85.335 us, 89.392 us]`
- `bfgs/rosenbrock/10`: `[74.371 us, 75.792 us, 77.263 us]`

The focused baseline artifact later measured `powell/rosenbrock/10` at
`[874.58 us, 965.52 us, 1.1121 ms]` on worker vmi1156319, but that result was
treated as a cross-worker outlier relative to the broad-profile row.

A scratch HEAD control run also measured the target at
`[441.97 us, 507.44 us, 608.62 us]` on worker vmi1227854.

## Candidate

The one-lever candidate reused one scratch `candidate_x` vector inside
`golden_section_direction_search` instead of allocating a new vector through
`add_scaled` for each one-dimensional objective sample.

Opportunity score before measurement:

| Hotspot | Impact | Confidence | Effort | Score |
|---------|--------|------------|--------|-------|
| Powell line-search candidate vector allocation | 2 | 3 | 2 | 3.0 |

## Behavior proof

Golden output was captured by the temporary `powell_rosenbrock10_golden_snapshot`
test before and after the candidate:

- before sha256: `d527d69305d175a37261d73e404bcb25996dc7fdac1f1c58ccbc0c987b5abf5e`
- after sha256: `d527d69305d175a37261d73e404bcb25996dc7fdac1f1c58ccbc0c987b5abf5e`

Isomorphism:

- Ordering preserved: alpha samples and objective calls stayed in the same order.
- Tie-breaking unchanged: the same `fb > fx`, `f_next > fb`, `fc < fd`, and
  `candidate_f <= fx` comparisons were used.
- Floating-point preserved: each candidate component remained
  `left + scale * right`; no summation, coefficient, tolerance, or comparison
  formula changed.
- RNG: N/A.

## RCH measurements

Post-candidate focused benchmark:

- `powell/rosenbrock/10`: `[356.30 us, 369.59 us, 385.37 us]`, worker vmi1149989
- repeat after-run: `[767.98 us, 803.58 us, 836.45 us]`, worker vmi1153651

This overlaps the broad profile baseline interval and is only about 1.6% faster
than the broad-profile mean. The repeat after-run is slower than the scratch HEAD
control and broad baseline. RCH did not provide a same-worker before/after pair,
and the focused baseline on vmi1156319 was an outlier, so the candidate does not
meet the required keep threshold.

## Decision

Rejected. The production `fsci-opt` code lever was backed out. The bead and
artifacts remain as evidence for the shifted bottleneck; a future opt pass needs
a stronger profile-backed lever or a reproducible same-worker comparison.
