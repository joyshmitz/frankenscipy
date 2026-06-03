# Primitive Selection: MR6 Register-Fit GEMM Microkernel

- Bead: `frankenscipy-8l8r1.24`
- Target: `fsci-linalg` flat GEMM workspace full-tile path
- Baseline worker: `vmi1153651`
- Baseline keep row: `matmul/1024x1024` median `817.83 ms`
- Golden-before stable SHA-256: `b756bde7f00b52f08f37f77e67b4f03abcb06b2c551fe04315be57004b40551e`

## Selected Lever

Implement one register-file-fit `6 x 8` SIMD full-tile microkernel ahead of the existing `4 x 8` full-tile path. The microkernel keeps one `Simd<f64, 8>` accumulator per output row and reuses each loaded `B[k, j..j+8]` vector across six rows before falling back to the existing `4 x 8` kernel and scalar edge path.

This is an alien-graveyard/autotuned-register-blocking primitive: change the register-blocked row count to better balance B-vector reuse against SIMD-register pressure, instead of tuning the already rejected loop shape.

## Explicit Exclusions

- Not an `8 x 8` row-panel retry.
- Not a dual-`f64x4` SIMD-width split.
- Not a packed-panel retry.
- Not output materialization or allocation tuning.
- Not a scalar edge-path change.
- Not a validation, shape, or API-contract change.

## Isomorphism Contract

- Validation order and error behavior remain unchanged.
- Output row-major ordering remains unchanged.
- Each output cell still accumulates over `k = 0..ka` in increasing order.
- Floating-point operation grouping remains one lane-wise multiply-add per row and `k`; no cross-row reductions are introduced.
- Full-tile stores write the same `(row, col)` coordinates as before.
- Existing `4 x 8` SIMD and scalar edge paths remain available for remainders.
- No RNG, tie-breaking, global state, concurrency, or tolerance policy is touched.

## Keep Gate

Keep only if the after run preserves the golden SHA and delivers a real `matmul/1024x1024` median win versus `817.83 ms` with Score `>= 2.0`.
