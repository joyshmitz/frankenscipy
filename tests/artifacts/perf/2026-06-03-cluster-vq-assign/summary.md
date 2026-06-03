# fsci-cluster — nearest-centroid assignment optimization (2026-06-03)

**Lever (single):** replace the naive `argmin over sq_dist` inner loop shared by
`vq`, `kmeans`, and `mini_batch_kmeans` assignment with a contiguous-buffer,
prefilter-seeded, partial-distance early-abandoning nearest-centroid search.

## What changed

`crates/fsci-cluster/src/lib.rs`:
- `flatten_centroids()` — packs the ragged `Vec<Vec<f64>>` centroid set into one
  contiguous `k × d` row-major buffer once per assignment pass (`O(k·d)`,
  amortized over `n` observations). Kills the per-centroid heap-pointer chase and
  cache miss that dominated the old scan.
- `nearest_centroid()` — exact search over the flat buffer:
  1. `PREFILTER_DIMS`-coordinate probe picks a likely-nearest centroid; its full
     distance seeds a tight incumbent bound from the first comparison.
  2. Index-order scan with **strict-`>`** partial-distance abandonment
     (`sq_dist_within`): once the running squared sum passes the incumbent the
     centroid is rejected without finishing. Ties are summed in full and broken by
     lowest index (`sd == min_sq && c < best_c`).

## Isomorphism / behavior parity

The result is **bit-identical** to the naive argmin:
- The selected minimum is always a fully-summed `sq_dist` (the winner is never
  abandoned: its partial sums stay `≤` the incumbent), so labels, distances, and
  inertia match bit-for-bit.
- Strict-`>` abandonment + the lowest-index tie clause reproduce the original
  "first index wins ties" semantics regardless of the prefilter seed.

**Proof:** `perf_cluster golden` emits `to_bits()` of vq labels+distances and
kmeans labels+inertia over a fixed shape sweep.

```
golden_baseline.txt sha256 = abdf711959f94f83daa272fe37814f7952924e6ea0ddcdb4cdaf77444b40da96
golden_after.txt    sha256 = abdf711959f94f83daa272fe37814f7952924e6ea0ddcdb4cdaf77444b40da96   (IDENTICAL)
```

Harness A/B (`vq` vs in-binary reference `vq-base`) emits identical checksums; the
crate's 54 unit + metamorphic tests (incl. `mr_vq_matches_kmeans_assignment`)
pass; clippy `-D warnings` clean.

## Before / after (release-perf, hyperfine --warmup 3 --runs 15 -N, remote worker)

| n    | d   | k  | baseline (vq-base) | optimized (vq) | speedup |
|------|-----|----|--------------------|----------------|---------|
| 2000 | 64  | 32 | 67.1 ms            | 20.1 ms        | 3.34×   |
| 2000 | 128 | 32 | 106.0 ms           | 21.2 ms        | 5.00×   |
| 2000 | 256 | 32 | 171.9 ms           | 26.1 ms        | 6.58×   |
| 4000 | 128 | 64 | 303.2 ms           | 39.2 ms        | 7.74×   |

Score ≥ 2.0 cleared at every measured shape (3.3×–7.7×). Win grows with `d` and
`k` — exactly the BLAS-class regime (high-dim embeddings, large codebooks) where
the no-gaps directive demands parity.

## Path investigated and rejected

- Abandonment-only (no contiguous buffer, no seed): 1.1×–1.4×. Structural ceiling
  ~1.9× because ~half the centroids are scanned before the winner with a loose
  bound, and the cost is dominated by per-centroid pointer-chasing, not
  arithmetic. The contiguous buffer is what unlocks the win; the prefilter seed
  then makes abandonment bite from the first comparison.
