# Pass 2 - Block-TSQR Tree/Replay Primitive Spec

Bead: `frankenscipy-8l8r1.56`

## One Lever

Add one private candidate for tall full-rank public SVD:

1. Partition `A` into deterministic contiguous row blocks with at least `n` rows per block.
2. Factor each block with existing deterministic unpivoted Householder QR, storing leaf reflectors and leaf `R`.
3. Reduce the leaf `R` matrices through a fixed left-to-right binary tree:
   - stack `[R_left; R_right]`,
   - QR the `2n x n` stack,
   - store the internal reflectors and parent `R`.
4. Compute the existing deterministic thin SVD of root `R`.
5. Replay TSQR `Q` top-down:
   - start with `U_R`,
   - apply each internal node's stored `Q` to `[input; 0]`,
   - split into left/right child inputs,
   - apply leaf `Q` to `[leaf_input; 0]`,
   - scatter leaf rows into final `U`.
6. Keep `S` and `Vt` from root `R`, canonicalize signs, and let the existing public guard accept or fall back.

## Determinism

- Block size is a pure function of shape.
- Leaves are contiguous and ordered by row.
- Tree pairing is fixed left-to-right; odd singleton leaves are carried unchanged.
- Reflector application order is the mathematical reverse replay order already used by the existing reduction code.
- No randomization or worker-count-dependent behavior.

## Public Guard

The new candidate may be tried before `public_bidiag_thin_svd_candidate`, but public call sites must still require
`public_bidiag_svd_accepts`. Any malformed dimensions, nonfinite values, rank loss, clustered spectrum, or failed
reconstruction falls back to the existing route.

## Keep Gate

Use the Pass 1 baselines:

- Criterion target on `vmi1167313`: beat `lstsq 120.64 ms` and `pinv 116.98 ms` with Score >= 2.0.
- Public route target on `vmi1153651`: avoid regressing `lstsq 131.225285 ms` and `pinv 118.656896 ms`.
- Golden SHA must remain `1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225`.
