# Sobol 2D Incremental Primitive Selection

## Profile Linkage

- Bead: `frankenscipy-8l8r1.7`
- Hotspot: `qmc_sampling/sobol_2d/4096`
- Fresh RCH baseline: `[471.70 us, 486.30 us, 496.97 us]`
- Golden before SHA: `ff622ca915b745828cff4da1fa1954c628deb6eb76d49612a606a2034862a815`

The benchmark constructs `SobolSampler::new(2)` and calls `sample(4096)`. The
current path computes `sobol_bits(idx, dim)` for both coordinates of every row,
walking the gray-code set bits after the prior direction-table cache.

## Selected Lever

Specialize only the 2D Sobol sampler path:

1. Compute raw Sobol bits for the starting `next_index` once for dimensions 0
   and 1.
2. Emit row-major `(dim0, dim1)` values with the same digital-shift xor and
   `bits_to_unit` conversion.
3. Advance raw bits for the next row with
   `bits ^= SOBOL_DIRECTION_TABLES[dim][trailing_zeros(next_index_after_increment)]`.
4. Preserve `saturating_add`; if `next_index` is already `u64::MAX`, do not
   apply another xor transition.

This is distinct from the previous cached direction-table lever: it removes the
remaining per-coordinate gray scan inside the hot 2D sampling loop.

## Score

| Candidate | Impact | Confidence | Effort | Score |
| --- | --- | --- | --- | --- |
| 2D incremental gray-code state update | 4 | 4 | 2 | 8.0 |

Proceed because `8.0 >= 2.0`.

## Isomorphism Obligations

- Ordering preserved: rows are emitted from the current `next_index`, then
  advanced once per requested row.
- Dimension order unchanged: each row remains dimension 0 followed by dimension
  1.
- Gray-code transition unchanged: Sobol raw bits for `i + 1` are equal to raw
  bits for `i` xor the direction at `trailing_zeros(i + 1)`.
- Digital shift unchanged: the same per-dimension shift words are xor'ed before
  conversion.
- Floating-point unchanged: `bits_to_unit` remains the only conversion path.
- RNG unchanged: no new seed generation or random calls are introduced.
- Tie-breaking unchanged: no comparisons or branch-dependent ordering are
  introduced.
- Saturation unchanged: once `next_index == u64::MAX`, repeated sampling emits
  the saturated index exactly as the current `saturating_add` path does.

## Fallback Trigger

Reject the lever if the Sobol 2D golden SHA changes, focused Sobol tests fail,
or the after RCH Criterion interval does not show a real improvement over the
fresh baseline.
