# Primitive Selection

Target: `frankenscipy-8l8r1.14`, `fsci-linalg` `baseline_solve/1000x1000`.

Profile-backed target:
- Latest committed linalg reprofile after rejected GEMM levers kept `baseline_solve/1000x1000` as the next completed linalg hotspot at `214.69 ms` median.
- Fresh RCH Criterion baseline for this bead measured `[545.06 ms, 555.58 ms, 566.84 ms]` on `vmi1156319`.

Alien-graveyard primitive:
- Apply the profile-contract rule from `/data/projects/alien_cs_graveyard/alien_cs_graveyard.md`: profile probe paths before touching internals, change only a proven hotspot, then re-profile because bottlenecks shift.
- Selected primitive: remove a non-observable diagnostic probe from the solve path. `is_positive_definite` is a Cholesky-like O(n^3) SPD probe. `solve` and `solve_with_audit` already select the direct/general LU path from the same matrix diagnostics and do not expose `report.positive_definite` in `SolveResult`.

One lever:
- Keep public `condition_diagnostics` behavior unchanged by leaving its full SPD evaluation enabled.
- Add a solve-internal diagnostics mode that skips only the Cholesky SPD probe unless the caller explicitly asserted `MatrixAssumption::PositiveDefinite`.
- Use that mode from `solve` and `solve_with_audit`.

Isomorphism:
- Matrix validation, transpose handling, assumption normalization, metadata incompatibility scoring, LU factorization, pivoting, selected action, warning/certificate fields, backward-error computation, error variants, output row/column order, tie-breaking absence, and RNG absence are unchanged.
- Floating-point solve arithmetic is unchanged; the removed work was a separate diagnostic-only Cholesky probe not consumed by solve output.
- Public diagnostics still evaluates and reports positive-definite status exactly as before.
