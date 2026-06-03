# frankenscipy-lt8kr primitive selection

Pass: 3 - Alien Primitive Selection

## Symptom

The latest linalg profile remains dominated by `matmul/1024x1024`, and the
fresh focused baseline for this pass measured median `188.06 ms` on
`vmi1149989`.

## Selected Primitive

Use SIMD-width calibration in the full 4x8 flat-workspace GEMM tile.

The current kept kernel stores each row accumulator as one `Simd<f64, 8>`.
This trial keeps the same tile shape and output layout, but represents the
eight columns as two `Simd<f64, 4>` lane groups. That tests whether smaller
portable SIMD vectors reduce register pressure or wide-vector frequency costs
while preserving the exact same per-cell `k` accumulation order.

## Exclusions

- No C BLAS, MKL, LAPACK, XLA, or unsafe code.
- Not a row-panel retry.
- Not a packed-panel retry.
- Not an output-materialization tweak.
- Not a scalar tile-shape retry.
- Not an edge-path change.

## Isomorphism Contract

- Public API and error behavior remain unchanged.
- Validation order remains unchanged.
- Output row order remains unchanged.
- Each output cell/lane still accumulates `k = 0..ka` monotonically.
- The source uses lane-wise multiply followed by add; no reduction tree or
  reordered summation is introduced.
- Edge tiles retain the existing scalar loop.
- No RNG, tie-breaking, or global-state surface exists.
- Stable golden test-line SHA-256 must remain
  `b756bde7f00b52f08f37f77e67b4f03abcb06b2c551fe04315be57004b40551e`.

## Keep Gate

Reject unless focused RCH Criterion shows a real `matmul/1024x1024` win against
the pass baseline `188.06 ms` and Score is at least `2.0`.
