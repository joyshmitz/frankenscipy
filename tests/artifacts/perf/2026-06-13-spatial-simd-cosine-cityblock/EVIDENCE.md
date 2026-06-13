# SIMD cosine + cityblock distance metrics

## Lever
Follow-up to the sqeuclidean SIMD win (3210af77). `cosine` did THREE scalar
`.map().sum()` reductions (a·b, Σa², Σb²); `cityblock` did one (`Σ|aᵢ-bᵢ|`). Rust does
not auto-vectorize float-sum folds. Added `simd_dot` / `simd_sqsum` helpers (8-wide,
2 accumulators + scalar tail) and rewrote:
- `cosine` = `1 - simd_dot(a,b) / (simd_sqsum(a).sqrt()·simd_sqsum(b).sqrt())` (norms
  over the full respective vectors, dot over min — exactly the old behaviour, vectorized).
- `cityblock` as 8-wide `Σ|a-b|` (SIMD `.abs()`).

## Isomorphism
perf_cdist probe (metric=Cosine, self-checks parallel vs sequential byte-for-byte):
**bit_identical=true** on all 5 shapes, before AND after. `cargo test -p fsci-spatial
--lib` = **184 passed, 0 failed** (incl cosine NaN-on-zero-norm and reference-value tests).

## Benchmark (perf_cdist metric=Cosine, seq = single-thread distance compute)
| shape                  | before (scalar) | after (SIMD) | speedup |
|------------------------|-----------------|--------------|---------|
| cdist 2000×2000 dim=3  | 150.9 ms        | 69.2 ms      | 2.18x   |
| cdist 4000×1000 dim=8  | 165.0 ms        | 66.6 ms      | 2.48x   |
| cdist 3000×3000 dim=16 | 427.6 ms        | 216.4 ms     | 1.98x   |
| pdist 3000 dim=3       | 210.9 ms        | 84.0 ms      | 2.51x   |
| pdist 4000 dim=16      | 371.0 ms        | 203.3 ms     | 1.82x   |

~2x bit-identical on the cosine kernel (3 fused reductions vectorize better than the
single-reduction euclidean). cityblock is the same pattern for the L1 path. clippy + fmt clean.
