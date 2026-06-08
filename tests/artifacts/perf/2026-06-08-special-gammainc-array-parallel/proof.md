# perf: parallelize regularized incomplete gamma real-array arms (gammainc/gammaincc)

Bead: frankenscipy-nr8mi

## Lever
`gammainc_dispatch` (crates/fsci-special/src/gamma.rs) — behind `gammainc` and `gammaincc`,
the regularized lower/upper incomplete gamma functions (the backbone of the chi² / F /
Poisson / gamma distribution CDFs) — mapped the expensive per-element kernel
(`gammainc_scalar`/`gammaincc_scalar`, a series / Lentz continued fraction) serially over the
three real-real vector arms: `(a_vec, x_scalar)`, `(a_scalar, x_vec)`, and the matched-length
`(a_vec, x_vec)` zip. Routed all three through the `par_map_indices` helper already present in
this file (added with the gamma-family commit), which evaluates the index range in parallel
chunks. (The complex broadcast arms are left serial — niche.)

## Isomorphism / byte-identity argument
- Each output index `i` is `gammainc_scalar(a_i, x_i, mode)` (or the upper variant) written to
  slot `i`; chunks cover the index range contiguously and concatenate in index order ⇒
  identical output `Vec`. No reduction.
- Error path: chunk results folded with `?` in chunk (=index) order ⇒ first failing index's
  `SpecialError` returned, exactly as the serial `.iter().map(...).collect()`.
- The `a_vec`/`x_vec` length-mismatch check is preserved before the parallel map. Gate: serial
  for `< 256` elements.

⇒ The returned value (and first error) is bit-identical to the serial implementation.

## Proof (golden — serial vs parallel, identical)
Harness: `cargo run --profile release-perf -p fsci-special --bin perf_gammainc_array`
```
n=100   gammainc=72f18d438a3d2452 gammaincc=12c2eace396e7f88
n=5000  gammainc=2164e0978d95dbf5 gammaincc=ec61a01b52730f57
n=50000 gammainc=ce81ddfc59fa2a4b gammaincc=6cbab9a73e7c380c
timing acc (gammainc): 1M=ba066c07e49794b8  2M=1ffeabfe07288908
```
Identical in the stashed serial build and the parallel build.
sha256(golden payload file) = 5e4779699687624557bc983ae009b65225c1be3e8d54a6ee1377e4e2f15bdd60

## Timing (rch remote, release-perf, 3 back-to-back runs each) — gammainc (a=3.5, x∈[0,30))
| array | serial (3x)             | new (3x)             | speedup |
|-------|-------------------------|----------------------|---------|
| 1M    | 130.6/127.7/129.1 ms    | 12.91/16.08/13.07 ms | ~9.9x   |
| 2M    | 264.8/268.2/252.9 ms    | 19.81/18.56/21.13 ms | ~13.4x  |

## Validation
25 gammainc unit tests pass; clippy: no warning in gamma.rs (the parallel arms are clean).
