# Mixture 2D Powi-Square Primitive Selection

Bead: `frankenscipy-8l8r1.5`

## Profile Target

Source profile:

`tests/artifacts/perf/2026-06-03-stats-sobol-directions/reprofile_stats_after_sobol_rch.txt`

After the Sobol cached-direction pass, the shifted stats profile included:

- `qmc_discrepancy/mixture/512x2`: `627.41us` median.

Focused RCH Criterion baseline:

`cargo bench -p fsci-stats --bench stats_bench --locked -- qmc_discrepancy/mixture/512x2 --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot`

Baseline on `vmi1227854`: `[255.36 us, 267.89 us, 274.25 us]`.

## Selected Lever

Replace `delta.powi(2)` with `delta * delta` in the 2D mixture discrepancy
pair loop only.

The 2D mixture loop is O(n^2); for 512 rows it evaluates the squared delta
term twice per pair. Direct multiplication is the same mathematical operation
for exponent 2 and stays in the same formula position:

`... - 0.75 * |delta| + 0.5 * delta_squared`

No validation, loop ordering, output ordering, RNG, tie-breaking, or global
state surface changes.

## Isomorphism Obligations

- Validation order remains dimension, shape, empty, range checks before fast-path dispatch.
- Row order remains ascending.
- Pair order remains `point_i` outer, `point_j` inner.
- Dimension order remains coordinate 0 then coordinate 1.
- Formula term order is unchanged; only the square primitive changes.
- Golden QMC discrepancy payload must remain byte-identical.

## Score

Initial score: `6.0 = impact 2 * confidence 3 / effort 1`.

Reject if QMC golden changes, focused RCH Criterion does not show a real win,
validation fails, or final Score drops below `2.0`.
