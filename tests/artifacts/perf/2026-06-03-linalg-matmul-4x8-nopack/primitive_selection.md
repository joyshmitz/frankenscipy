# Primitive Selection

Bead: frankenscipy-8l8r1.16

Profile-backed target: post-solve reprofile at commit `4bb11e83` showed dense
GEMM as the shifted linalg hotspot:

- `matmul/768x768`: median `690.64 ms`
- `matmul/1024x1024`: median `1.5452 s`
- `baseline_solve/1000x1000`: median `104.21 ms`

Alien/no-gaps match: cache-aware dense numeric kernels, especially tiled GEMM
and register micro-kernels, are the no-gaps directive's flagship safe-Rust
target. This bead uses the smallest safe primitive in that family: widen the
existing no-pack full-tile register micro-kernel from `MR x NR = 4 x 4` to
`4 x 8`.

Rejected adjacent levers not repeated here:

- B-flat buffer
- A row-ref hoist
- store unroll
- scalar accumulator
- simple packed B-panel in the current `Vec<Vec<f64>>` layout

Fallback if this regressed: restore `NR = 4` and the four-column accumulator
stores, close the bead as rejected, and move to a new profile-backed primitive.
