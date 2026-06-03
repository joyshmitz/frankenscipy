# Sobol 2D Direction-Table Primitive Selection

Bead: `frankenscipy-seo6h`

## Profile Target

Focused RCH Criterion baseline:

`cargo bench -p fsci-stats --bench stats_bench --locked -- qmc_sampling/sobol_2d/4096 --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot`

Baseline on `vmi1264463`: `[648.26 us, 914.73 us, 1.3581 ms]`.

The hot path is `SobolSampler::sample`, which calls `sobol_bits(idx, dim)` for
each row and dimension. `sobol_bits` walks gray-code bits and calls
`sobol_direction(dimension, bit)` for each set bit; dimension `1` recomputes
the same direction recurrence for every sampled index.

## Selected Lever

Precompute the 64 direction numbers for supported Sobol dimensions `0` and `1`
as static safe-Rust tables, then have `sobol_bits` xor table entries in the
same gray-code bit order.

This is the same algorithmic recurrence with invariant work hoisted out of the
sampling loop. It follows the graveyard locality guidance of replacing repeated
kernel setup with contiguous cached lookup tables where the proof obligation is
translation equivalence against the old recurrence.

## Isomorphism Obligations

- Gray-code order: unchanged; bits are still consumed from least significant to
  most significant.
- Xor order: unchanged; each selected direction is xored into `value` in the
  same loop order.
- Digital shift: unchanged; `sample` still applies `^ self.digital_shift[dim]`
  after `sobol_bits`.
- Output layout: unchanged row-major point order and dimension order.
- State: `next_index` still advances once per emitted row with saturating add.
- Floating point: unchanged; `bits_to_unit` still performs the only conversion.
- RNG/tie-breaking: no RNG or comparison/tie surface exists in this path.

## Score

Initial score: `6.0 = impact 3 * confidence 4 / effort 2`.

Reject if the Sobol golden SHA changes, focused RCH Criterion does not show a
real win, validation fails, or final Score drops below `2.0`.
