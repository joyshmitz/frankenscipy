# frankenscipy-8l8r1.104 - CG Wolfe Scratch Reuse Probe

## Target

- Bead: `frankenscipy-8l8r1.104`
- Crate: `fsci-opt`
- Focused benchmark: `cg/rosenbrock/10`
- Profile basis: no ready `[perf]` bead was available after a short wait; the broad `fsci-opt` Criterion run reached `cg/rosenbrock/10` as the largest completed row before the broad run was killed during Powell, so the keep gate was narrowed to this focused CG row.

## Candidate Lever

Reuse the nonlinear CG Wolfe finite-difference scratch path:

- add a private directional-derivative strong-Wolfe helper,
- reuse a trial-point buffer inside the line search,
- reuse the finite-difference perturbation buffer for directional gradients,
- update the PR+ direction in place.

No unsafe Rust, no external BLAS/LAPACK/XLA linkage, no RNG, and no algorithm-selection changes were introduced by the candidate.

## Isomorphism Contract

- Wolfe ordering/tie-breaking: alpha growth, Armijo checks, curvature checks, zoom bisection, and `amin` break behavior are intentionally identical to `line_search_wolfe2`.
- Floating point: each finite-difference component computes the same `(fp - fm) / (2.0 * step)` value; directional accumulation visits dimensions in ascending index order, matching the old gradient-vector dot order.
- PR+ direction: the in-place update preserves the old per-component operation order `(-next_grad) - (-beta * direction)`.
- RNG: none.
- Golden payload SHA-256:
  - before: `f02b24201c2844e1cb1577159ebb29535e2d16a8ccd3676279670e8b6fffad27`
  - after:  `f02b24201c2844e1cb1577159ebb29535e2d16a8ccd3676279670e8b6fffad27`
- `diff -u golden_minimize_before_payload.txt golden_minimize_after_payload.txt` produced no differences.

## Benchmark Evidence

Primary keep gate, same scratch path and same RCH worker:

| artifact | worker | row | p50 |
| --- | --- | --- | --- |
| `baseline_cg_rosenbrock10_clean_c5e31c81_match_attempt_rch.txt` | `vmi1227854` | `[310.66 us, 320.32 us, 339.65 us]` | `320.32 us` |
| `after_cg_rosenbrock10_clean_path_candidate_match_rch.txt` | `vmi1227854` | `[317.03 us, 323.84 us, 336.62 us]` | `323.84 us` |

Delta: `320.32 / 323.84 = 0.989x`; the candidate is a slight regression by midpoint and does not clear the keep gate.

Non-scoring artifacts:

- `after_cg_rosenbrock10_final_rch.txt` and `after_cg_rosenbrock10_match_attempt_rch.txt` are RCH candidate rows on `vmi1293453` without a matching same-worker baseline.
- `matched_one_worker_cg_rosenbrock10_rch.txt` is diagnostic only because `rch exec` rejected the outer `sh -lc` wrapper as non-compilation and ran the comparison locally.
- Earlier `vmi1149989` transcript evidence is routing-only because the final candidate diff changed after that row was captured.

## Score And Verdict

- Impact: `0.0` because the matched same-worker row regressed by p50.
- Confidence: `4.0` because the same path and same worker comparison is clean.
- Effort: `1.5`.
- Score: `(0.0 * 4.0) / 1.5 = 0.0`.

Verdict: reject this scratch-reuse lever; do not ship the source change for this bead.

## Next Route

Move away from the micro workspace lever and attack a deeper primitive: derivative-aware nonlinear CG value/gradient batching so Wolfe can evaluate `f` and finite-difference directional information with a shared perturbation schedule and fewer objective calls. Target ratio: at least `1.35x` on `cg/rosenbrock/10` before broader opt portfolio profiling.
