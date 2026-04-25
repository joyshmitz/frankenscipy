# fsci-linalg DISCREPANCIES

Functions where fsci-linalg deliberately diverges from `scipy.linalg`,
rather than passively missing the implementation. Each entry is a
documented decision; revisit if the assumption changes.

## Declined: BLAS / LAPACK driver introspection

scipy.linalg exports three functions that exist purely to introspect
the underlying BLAS/LAPACK implementation:

- `scipy.linalg.find_best_blas_type(arrays, dtype)`
- `scipy.linalg.get_blas_funcs(names, arrays, dtype, ilp64)`
- `scipy.linalg.get_lapack_funcs(names, arrays, dtype, ilp64)`

These are not implemented in fsci-linalg because:

1. fsci-linalg builds on `nalgebra` (pure Rust), not on a BLAS/LAPACK
   binding. The concept of "the BLAS routine name to call for this
   dtype" has no analogue in our backend.
2. Consumer code that genuinely needs to dispatch on backend type
   should be doing so above the linalg layer, not inside it.
3. Rust's monomorphization makes per-dtype routing automatic — there
   is nothing for the user to introspect.

Decision: do NOT port these. If a downstream consumer asks for them,
file a bead and reconsider; until then, a `LinalgError::Unsupported`
on attempt is the right policy.

## Declined: full scipy.linalg.lapack / scipy.linalg.blas surface

scipy.linalg.lapack exposes 480 Fortran routines (147 distinct
kernels × {s, d, c, z} dtype prefixes), and scipy.linalg.blas
exposes 128 routines (46 kernels). For the same reasons as the
introspection trio (no Fortran shim, monomorphization removes
per-dtype routing), fsci-linalg does NOT expose the LAPACK / BLAS
namespace.

Per frankenscipy-uzht, a curated migration table is maintained in
[LAPACK_MAPPING.md](./LAPACK_MAPPING.md). 24 of the 147 LAPACK
kernels and 14 of the 46 BLAS kernels — the most-used in scipy
porting — have a documented fsci-linalg counterpart there. The
audit script `audit/audit_lapack.py` enumerates the full surface
so the table can be kept in sync as scipy releases.

Users porting code that imports from `scipy.linalg.lapack` or
`scipy.linalg.blas` should consult LAPACK_MAPPING.md to find the
high-level fsci-linalg fn that supplants their direct Fortran call.

## Declined: `orthogonal_procrustes`

scipy.linalg.orthogonal_procrustes is mathematically the same problem
as scipy.spatial.procrustes (find the orthogonal R minimizing
||R·A − B||). fsci has the implementation in fsci-spatial::procrustes
already; duplicating it under fsci-linalg would require maintaining
two copies in lock-step.

Decision: leave the entry point in fsci-spatial only. Add a doc-link
from fsci-linalg's lib.rs preamble pointing scipy users at the
correct module if they grep for the symbol.

Related: frankenscipy-l15l (procrustes Newton-Schulz divergence on
rank-deficient input).

## Soft-declined: `solve_lyapunov` (deprecated alias)

scipy.linalg.solve_lyapunov is a deprecated alias for
solve_continuous_lyapunov. fsci-linalg provides
`solve_continuous_lyapunov` directly, so the alias adds no surface
beyond a deprecation-warning line.

Decision: do NOT port the alias. Direct callers can switch to
`solve_continuous_lyapunov`.

## Pending decision: high-numeric-cost decomposition derivatives

`expm_cond` and `expm_frechet` (sensitivity / Fréchet derivative of
the matrix exponential) sit at the intersection of "rare in user code"
and "expensive to verify". They would require a parity oracle that
itself depends on scipy's reference implementation, which is currently
the only published implementation in the Rust + scipy intersection.

Decision: defer. File as P2 follow-up if a downstream consumer needs
either of these; otherwise let them age out.

## Aliasing notes

| scipy name | fsci name | rationale |
|------------|-----------|-----------|
| `dft` | `dft_matrix` | `dft` is a 3-letter symbol that collides with the `fsci-fft` crate's namespace and is a noun, not a verb; rename clarifies it produces the DFT matrix rather than performing a DFT. |

## How divergences are tracked going forward

When a user-visible divergence ships:

1. Update this file with the rationale.
2. If the divergence affects parity scoring, mention it in the
   relevant fixture's `category` field (use `category: declined`
   so the scoring runner can classify it correctly).
3. Cross-reference the bead that introduced the divergence so a
   future refactor can re-evaluate.
