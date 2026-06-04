# QMC 2D symmetric-pair primitive selection

Bead: `frankenscipy-pgm85`

## Profile Target

Fresh RCH Criterion baseline on worker `ts1`:

| Row | Median |
| --- | ---: |
| `qmc_discrepancy/centered/512x2` | `319.46 us` |
| `qmc_discrepancy/mixture/512x2` | `303.91 us` |
| `qmc_discrepancy/l2_star/512x2` | `222.95 us` |
| `qmc_discrepancy/wraparound/512x2` | `212.39 us` |

Golden SHA-256 before changes:

`1fb5885cc35367f57b0e818e165a28f87cbb0b9a43fdc7ba4728a6778af44daf`

## Primitive

The QMC 2D discrepancy kernels are row-major ordered double sums. For an
off-diagonal pair `(i, j)`, the distance terms are shared by the two oriented
entries `(i, j)` and `(j, i)`. The selected lever is a row-preserving symmetric
pair walk: compute shared pair deltas once, add the current row's oriented term
at the same point in the row-major order, and store the reverse oriented term
for the later row.

This follows the graveyard segmented-array/reduction pattern: keep an explicit
flat/segmented work representation rather than relying on nested-loop
recomputation. It also follows the FrankenSuite profile-first contract gate:
profile first, one lever, explicit proof, fallback by rejection if the golden
hash or benchmark fails.

## Proof Obligations

- Ordering: final additions to `double` must occur in the original row-major
  order: all `j < i`, then `j == i`, then all `j > i`.
- Orientation: reverse stored terms must be computed with the same arithmetic
  expression order as the original `(j, i)` loop body.
- Floating point: do not replace `powi(2)` with `delta * delta`; previous
  QMC work found that direct square was not an acceptable lever.
- RNG: no sampler state or seed changes.
- Tie-breaking: none introduced.
- Golden: `perf_stats qmc-golden` SHA-256 must remain unchanged.

## Candidate Score

| Candidate | Impact | Confidence | Effort | Score |
| --- | ---: | ---: | ---: | ---: |
| Row-preserving symmetric pair walk for 2D mixture | 4 | 3 | 2 | 6.0 |

The kept scope is the 2D mixture kernel first because it is the bead's primary
profile target. Sibling kernels can be re-profiled after this commit because
bottlenecks shift.
