# frankenscipy-l566o: panel SIMD micro-lever rejected

## Lever

Tried vectorizing the within-panel rank-1 update in `lu_factor_blocked`:

```text
data[i][jj] -= data[i][j] * data[j][jj]   for jj in j+1..kb
```

The attempted f64 panel hunk used 8-wide `Simd<f64, 8>` over contiguous panel
columns and preserved pivot search, row-swap order, division order, and the
per-element update expression. No source from this lever is retained.

## RCH evidence

Current post-U-solve baseline:

| Artifact | Worker | Mixed median | f64 median |
| --- | --- | ---: | ---: |
| `baseline_solve_1000_current_head_rch.txt` | `vmi1149989` | 32.824 ms | 50.530 ms |
| `baseline_post_u12_solve_1000_vmi1149989_rch.txt` | `vmi1227854` | 34.073 ms | 61.482 ms |

Candidate after runs:

| Artifact | Worker | Mixed median | f64 median | Decision |
| --- | --- | ---: | ---: | --- |
| `after_solve_1000_panel_simd_rch.txt` | `vmi1227854` | 35.878 ms | 54.207 ms | mixed regressed; f64 movement not enough to trust |
| `after_solve_1000_f64_panel_only_rch.txt` | `vmi1227854` | 38.014 ms | 63.797 ms | regression |

The mixed route is the primary `solve` hot path after `1d7d450b`; this lever
does not move it robustly. The f64-only movement is contradictory across the
same worker and does not clear the keep bar.

## Proof / isomorphism

No source is retained. `git diff --exit-code crates/fsci-linalg/src/lib.rs`
passed before closeout, so public ordering, pivot tie-breaking, floating-point
behavior, RNG surface, unsafe surface, and golden digest remain exactly those
of `c520dd39`.

## Routing

Do not repeat scalar-panel SIMD or trailing-GEMM cache-block micro-levers for
`l566o`. The accepted wins in this lane are already on `main`:

- `1d7d450b`: U-block triangular solve SIMD.
- `440d9d71`: multi-RHS TRSM inverse.

Next work needs a fresh profile after those commits and a different primitive,
not another adjacent blocked-LU loop tweak.
