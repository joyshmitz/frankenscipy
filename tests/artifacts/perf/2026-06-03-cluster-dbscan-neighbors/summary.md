# fsci-cluster — DBSCAN neighbor-scan optimization (2026-06-03)

**Lever (single):** apply the contiguous-buffer + partial-distance abandonment
recipe (proven on the assignment path, `0380e77e`) to DBSCAN's `O(n²)` neighbor
query.

## What changed

`crates/fsci-cluster/src/lib.rs`, `dbscan()`:
- `flatten_points()` — pack the ragged `Vec<Vec<f64>>` observations into one
  contiguous `n × d` row-major buffer once. The neighbor scan re-walks every
  point, so streaming a contiguous buffer kills the per-point heap-pointer chase
  + cache miss that dominated the old `data[idx]` / `data[j]` indexing.
- Neighbor predicate now uses `sq_dist_within(pi, pj, eps2) <= eps2` instead of
  `sq_dist(pi, pj) <= eps2`: the squared distance is summed with **early
  abandonment** the instant the running sum exceeds `eps2`. In density clustering
  most pairs are non-neighbors and bail out after a few dimensions.

## Isomorphism / behavior parity

**Bit-identical** labels, core-sample set, and cluster count:
- A genuine neighbor (full distance `≤ eps2`) never abandons (its running sum
  stays `≤ eps2`), so it is summed in full and the `≤ eps2` test is exactly the
  original boolean. A non-neighbor abandons with a partial sum `> eps2`, still
  `> eps2`, so it is excluded identically. The membership predicate — and hence
  the entire BFS expansion — is unchanged.

**Proof:** `perf_cluster golden` emits DBSCAN `n_clusters`, per-point labels, and
core indices over a fixed shape sweep.

```
golden_baseline.txt sha256 = faf98a61e44f348f14fcb4357038d35c399c6cd2f7830192f0ddb0cef829ed49
golden_after.txt    sha256 = faf98a61e44f348f14fcb4357038d35c399c6cd2f7830192f0ddb0cef829ed49   (IDENTICAL)
```

In-binary A/B (`dbscan` vs reference `dbscan-base`) emits identical checksums;
99 + 54 unit/metamorphic tests pass; clippy `-D warnings` clean.

## Before / after (release-perf, hyperfine --warmup 2 --runs 12 -N, remote worker)

In-binary A/B of the single lever (`dbscan-base` = pre-lever naive scan):

| n    | d  | eps=3, min_samples=4 | speedup |
|------|----|----------------------|---------|
| 2000 | 16 |                      | 1.99×   |
| 2000 | 32 |                      | 3.42×   |
| 3000 | 32 |                      | 3.73×   |
| 3000 | 64 |                      | 7.95×   |

Win grows with `d` (more dimensions to abandon, more streaming benefit). Clears
Score ≥ 2.0 across the meaningful regime (`d ≥ 32`: 3.4×–8×); the `d = 16` case
sits at the 2.0 boundary.

## Notes

- `eps2`-bounded abandonment is exact for a *threshold* test (unlike argmin it
  needs no tie clause — the predicate is a pure `≤`), making this an even cleaner
  application of the lever than the assignment path.
- Remaining same-pattern candidate: `kmedoids` assignment + the M×M intra-cluster
  `dmat` build still index scattered `data[med]` / `data[members[i]]`.
