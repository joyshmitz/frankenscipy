# frankenscipy-8l8r1.22 primitive selection

Pass: 3 - Alien Primitive Selection

## Symptom

`matmul/1024x1024` remains the profile-backed linalg target. Fresh RCH
Criterion for this pass measured median `817.34 ms` on `vmi1156319` at
`a633e89d7b1da8755ae580e5f1f64e6e80069116`.

## Selected Primitive

Use safe Rust portable SIMD lanes for the full 4x8 tile path in
`matmul_flat_workspace`.

The scalar flat-workspace micro-kernel keeps four rows and eight columns in
register-resident accumulators. The SIMD trial keeps the same tile shape but
stores each row accumulator as `Simd<f64, 8>`, loading the contiguous
`B[k][j0..j0+8]` row slice into one vector and applying four lane-wise
`acc += splat(a_i_k) * b_vec` updates per `k`.

This is a deeper compute primitive than the rejected row-owned C output layout:
it targets arithmetic throughput in the hot tile loop, not output
materialization.

## Exclusions

- No C BLAS, MKL, LAPACK, XLA, or unsafe code.
- Not a packed-panel retry.
- Not a wider scalar tile retry.
- Not an output-materialization tweak.

## Isomorphism Contract

- Public API and error behavior remain unchanged.
- Validation order remains unchanged.
- Output row order remains unchanged.
- Each output cell/lane still accumulates `k = 0..ka` monotonically.
- The source uses explicit lane-wise multiply followed by add; no reduction tree
  or reordered summation is introduced.
- No RNG, tie-breaking, or global-state surface exists.
- Golden sorted test-line SHA-256 must remain
  `61e12eb58f34ccba1dcedd29425ff3292fd7df5769f7411352cd2a617a58d6c7`.

## Keep Gate

Reject unless focused RCH Criterion shows a real `matmul/1024x1024` win against
the pass baseline `817.34 ms` and Score is at least `2.0`.
