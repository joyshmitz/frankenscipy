# Primitive Selection

## Candidate

Bounded symmetric-pair product cache for the 2D L2-star discrepancy double sum.

The profiled hot rows are the 2D QMC discrepancy double sums. Directly changing
the double-sum loop to `double += 2.0 * pair` would change row-major
floating-point accumulation order, so that shape is rejected before source edit.

The selected lever computes each unordered 2D L2-star pair product once into an
upper-triangle table, then replays the final `double += product` in the exact
original row-major `(i, j)` order. This removes duplicate `max` and multiply
work while preserving the final addition sequence.

## Graveyard Grounding

- Canonical graveyard cache-friendly/data-locality guidance favors contiguous
  tables when the working set fits in cache and the access pattern is predictable.
- The certified-rewrite artifact family applies because this is an algebraic
  common-subexpression rewrite on a hot numeric kernel under a strict golden gate.

## Artifact-Coding Proof Obligations

- Inputs and validation are unchanged.
- Point construction and sample order are unchanged.
- For finite unit-cube 2D inputs, `max(x_i, x_j) == max(x_j, x_i)` and the
  product term is symmetric.
- The final accumulation loop must visit `(i, j)` in the same row-major order as
  the baseline implementation.
- Golden output SHA-256 must remain
  `1fb5885cc35367f57b0e818e165a28f87cbb0b9a43fdc7ba4728a6778af44daf`.
- RNG and tie-breaking surfaces are absent.

## Risk Gate

- The cache is bounded. Inputs outside the cap use the old direct row-major
  implementation to avoid changing memory-failure behavior on very large `n`.
- Revert the source lever if the golden payload differs or if focused RCH
  Criterion does not clear Score >= 2.0.

## Score Target

`3.0 = impact 2 * confidence 3 / effort 2`
