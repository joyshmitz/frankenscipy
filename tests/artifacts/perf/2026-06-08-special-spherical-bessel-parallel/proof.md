# perf: parallelize spherical Bessel real-array arms (spherical_jn/yn/in/kn)

Bead: frankenscipy-y63pi

## Lever
`spherical_bessel_dispatch` (crates/fsci-special/src/bessel.rs) — behind spherical_jn,
spherical_yn, spherical_in, spherical_kn (spherical Bessel functions j_n/y_n/i_n/k_n;
quantum scattering, multipole expansions, EM) — mapped its per-element recurrence kernel
serially over the three real-real broadcast arms (`order_vec × x_scalar`, `order_scalar ×
x_vec`, `order_vec × x_vec` zip). Factored the repeated 4-way `match kind` into one local
`eval(order, x)` closure, then routed all three array arms through the `par_map_indices`
helper in bessel.rs. Complex-z arms left serial (niche).

## Isomorphism / byte-identity argument
- `eval` is pure in `(order, x)` given the fixed `kind`/`mode`. Each output index `i` is
  `eval(...broadcast at i...)` written to slot `i`; chunks cover the index range contiguously
  and concatenate in index order ⇒ identical output `Vec`. No reduction.
- Error path: chunk results folded with `?` in chunk (=index) order ⇒ first failing index's
  `SpecialError`, exactly as the serial `.iter().map(...).collect()`.
- The `order_vec × x_vec` length check + scalar/complex arms untouched. Gate: serial for
  `< 256` elements.

⇒ The returned value (and first error) is bit-identical to the serial implementation.

## Proof (golden — serial vs parallel, identical)
Harness: `cargo run --profile release-perf -p fsci-special --bin perf_spherical_bessel_array`
(order n=5, z∈(0.1,40.1))
```
n=100   sph_jn=3e9d426915ab36a5 sph_yn=bd9ec95db90648d9
n=5000  sph_jn=8b4273b852650bcb sph_yn=ace185993056713f
n=50000 sph_jn=09a76eb9741d0763 sph_yn=09760480e0e05392
timing acc: sph_jn 1M=1345ce45fd631c4a  sph_yn 1M=5f34f7f016ff7d8d
```
Identical in the stashed serial build and the parallel build.
sha256(golden payload file) = d77fe7446e0ec476e83b6ef921b5c71c25c7b7af0c62269ebb25f0940e1f1976

## Timing (rch remote, release-perf, 3 back-to-back runs each)
| function / array | serial (3x)          | new (3x)            | speedup |
|------------------|----------------------|---------------------|---------|
| sph_jn, 1M       | 91.94/88.00/82.88 ms | 11.53/10.29/11.37 ms| ~7.7x   |
| sph_yn, 1M       | 46.04/44.03/49.92 ms | 7.53/6.75/(–) ms    | ~6.6x   |

## Validation
27 spherical-Bessel unit tests pass; clippy: no warning in bessel.rs.
