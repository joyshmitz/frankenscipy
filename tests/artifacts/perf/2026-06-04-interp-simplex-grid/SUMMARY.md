# Delaunay find_simplex: O(simplices) linear scan → uniform-grid spatial index

Bench: `scattered_2d` (fsci-interpolate), rch ts2. Function:
`Delaunay2D::find_simplex` (LinearNDInterpolator / griddata Linear).

## Lever
Scattered 2D linear interpolation locates each query's containing triangle by
scanning every simplex (O(simplices) per query, ~O(n·m) total). Added a uniform
grid (~sqrt(simplices) cells per axis) over the margin-padded simplex bounding
boxes, built once in `Delaunay2D::new`. Each simplex is bucketed into every cell
its padded bbox overlaps; `find_simplex` visits only the query cell's candidate
list (~O(1) for well-distributed data). Out-of-grid queries fall back to the
linear scan.

## Isomorphism
A simplex that passes `may_contain(query)` has a padded bbox containing the query,
which therefore overlaps the query's cell — so the cell's candidate set is a
*superset* of every simplex the linear scan would test, and candidates are stored
in ascending simplex-index order. Visiting them with the unchanged
`may_contain` + barycentric test returns the exact same first match, with the
exact same barycentric coordinates. Proven by
`delaunay_grid_find_simplex_matches_linear_scan` (simplex index + barycentric
`.to_bits()` equality vs the linear scan over npts 8/30/120/400, on a dense query
grid plus all vertices, edge midpoints, outside-hull, and grid-boundary points).
Full interpolate suite (123 tests) passes; clippy + fmt clean.

## Benchmark (LinearNDInterpolator, 576 points x 1024 queries, rch ts2)
| case                         | linear scan | grid index | Score |
|------------------------------|-------------|------------|-------|
| scattered_2d/linear_nd_eval  | 1.467 ms    | 102.9 µs   | 14.3x |

Eval-only (the triangulation is reused); the ratio grows with the point count
since the scan is O(simplices) per query while the grid is ~O(1). (griddata's
one-shot path is dominated by the separate O(n^2) Bowyer-Watson construction,
which this change does not touch.)
