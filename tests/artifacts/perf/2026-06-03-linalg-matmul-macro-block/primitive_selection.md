# frankenscipy-8l8r1.20 primitive selection

## Profile-backed target

Fresh RCH Criterion baseline on `vmi1156319`:

| row | median |
| --- | ---: |
| `matmul/512x512` | `100.62 ms` |
| `matmul/768x768` | `713.35 ms` |
| `matmul/1024x1024` | `1.1256 s` |

The latest reprofile still ranks `matmul/1024x1024` first, so the target
remains the large flat-workspace GEMM path.

## Alien primitive

Selected: cache-blocked row macro tiles over the flat GEMM workspace.

The current B-panel sweep traverses each 8-column panel across the full matrix
height. That attacks B-panel reuse but streams C and A through the entire matrix
for every column panel. The selected lever introduces an outer row macro block
for the large flat-workspace path so a bounded A/C row slab is reused across
all column panels before moving to the next row slab.

This is a cache/communication-locality lever, not another register-width tweak
and not the previously rejected packed-B retry.

## Isomorphism contract

- Validation and error order unchanged.
- Public output order unchanged: final return is still row-major `Vec<Vec<f64>>`.
- Floating-point order unchanged per cell: each `c[i][j]` accumulates `k = 0..ka`
  monotonically using the same separate `acc += a * b` updates.
- RNG unchanged: no RNG surface exists.
- Tie-breaking unchanged: GEMM has no tie-breaking surface.
- Golden before sorted test-line SHA-256:
  `61e12eb58f34ccba1dcedd29425ff3292fd7df5769f7411352cd2a617a58d6c7`.

## Keep gate

Keep only if RCH Criterion confirms a real `1024x1024` win with Score >= 2.0
and the after golden SHA matches the before SHA.
