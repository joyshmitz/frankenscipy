# Recommendation Card: Delaunay Candidate Rejection

Target: `frankenscipy-4czwo`, `fsci-interpolate` scattered 2D point location.

Measured symptom:
- Rust baseline, RCH focused: `griddata_linear/576x1024` median `6.3520 ms`.
- Upstream SciPy local comparison: `griddata(method="linear")` median `6.334958 ms`.
- Shared point-location rows: `linear_nd_eval_many` median `4.2368 ms`; `clough_tocher_eval_many` median `4.3761 ms`.

Alien primitive:
- Deterministic candidate generation before exact rerank, from the graveyard ANN pattern: reduce exact checks by generating a smaller candidate set, then preserve correctness with the original exact predicate.
- Cache-conscious contiguous metadata, from cache-oblivious/data-layout guidance: keep per-simplex bounds in a dense vector parallel to `simplices`.

One lever:
- Add conservative per-simplex coordinate bounds to `Delaunay2D` and skip barycentric evaluation only when the expanded bounds prove the query cannot satisfy the old `l_i >= -1e-10` acceptance test.

EV score:
- Impact: 3
- Confidence: 4
- Effort: 2
- Score: `(3 * 4) / 2 = 6.0`

Fallback trigger:
- Drop this lever if golden `griddata_linear` output changes, if RCH focused median does not improve, or if any crate-scoped gate fails.

Proof obligations:
- Ordering: original simplex scan order is unchanged; the prefilter only continues early for impossible candidates.
- Tie-breaking: first accepted simplex remains first in original order because no possible accepted simplex is skipped.
- Floating point: accepted candidates run the original barycentric formula and interpolation arithmetic unchanged.
- RNG: no RNG surface.
- Golden: before/after deterministic `griddata_linear` `f64::to_bits()` payload SHA must match.
