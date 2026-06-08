# perf: parallelize opt::hessian independent finite-difference components

Bead: frankenscipy-oe5zu

## Lever (ONE)
`hessian()` (crates/fsci-opt/src/lib.rs) filled the symmetric Hessian by a serial
double loop over the upper triangle, each entry an expensive adaptive central
second-difference with Richardson extrapolation (`adaptive_hessian_component`).
Every component reads only `f`, `x`, `f0` — no shared mutable state — so the
components are mutually independent. Compute them across cores in a single
`std::thread::scope` batch (chunked over the `(row, column)` pair list), then fold
the results back in the original pair order.

## Isomorphism / byte-identity argument
- `ddf`/`error`: each `(row, column)` and its mirror `(column, row)` are disjoint
  cells; write order is irrelevant.
- `nfev`: exact integer sum — order-independent.
- `nit`: `max` — order-independent.
- `success`: boolean `AND` — order-independent.
- `status`: `merge_differentiate_status` is a precedence pick
  (ErrorIncreased > MaxIterations > Converged) — commutative & associative.
- No floating-point reduction is reassociated: each component's `df`/`error` is the
  same scalar computed by the same `adaptive_hessian_component`, just on a different
  thread. The fold is sequential in pair order.
- Error path: results are folded with `?` in pair order, so the first failing pair
  returned is exactly the one the serial loop would have hit first.

Therefore the parallel result is **bit-identical** to the serial double loop.

## Proof (golden — serial baseline vs parallel NEW, identical)
Harness: `cargo run --profile release-perf -p fsci-opt --bin perf_hessian`
Objective: O(n^2)-per-eval pairwise-coupled scalar function (deterministic).

```
n=4  ddf_xor_bits=db5d13c07d7531a4 nfev=161  nit=5 success=true status=Converged
n=12 ddf_xor_bits=15e1344f3b48a201 nfev=1445 nit=6 success=true status=Converged
n=30 ddf_xor_bits=377d7eee32fbefb2 nfev=9097 nit=6 success=true status=Converged
```
Identical bits/counts in BOTH the stashed serial build and the parallel build.
sha256(golden payload) = bd381f6884d39502d12cf72f132598d9f899d09e96e9439365be5e823f209e97

## Timing (rch remote, release-perf)
| n   | serial   | parallel | speedup |
|-----|----------|----------|---------|
| 40  | 179.3 ms | 11.39 ms | 15.7x   |
| 70  | 1.680 s  | 70.03 ms | 24.0x   |
| 100 | 8.129 s  | 256.2 ms | 31.7x   |

Speedup grows with n (pair count ~ n^2/2 -> more independent work to spread).
Score >> 2.0.

## Gate
Serial path kept for `pairs.len() < 16` (small Hessians) so tiny problems never pay
thread-spawn. Validated: 3 hessian unit tests pass, clippy clean.
