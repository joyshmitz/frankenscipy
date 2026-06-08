# perf: parallelize arbitrary-order Bessel dispatch (jv/yv/iv/kv + scaled)

Bead: frankenscipy-5dhy5

## Lever
`bessel_dispatch` (crates/fsci-special/src/bessel.rs) — behind the **8** real-order Bessel
functions jv, yv, jve, yve, iv, kv, ive, kve (J_v/Y_v/I_v/K_v of arbitrary real order;
EM/waveguide/physics) — mapped its per-element kernel serially over the three real-real
broadcast arms (`order_vec × x_scalar`, `order_scalar × x_vec`, `order_vec × x_vec` zip).
Factored the repeated 8-way `match kind` into one local `eval(order, x)` closure (also
dedups/clarifies), then routed all three array arms through the `par_map_indices` helper
already in bessel.rs. Complex arms left serial (niche).

## Isomorphism / byte-identity argument
- `eval` is a pure function of `(order, x)` given the fixed `kind`/`mode`. Each output index
  `i` is `eval(...broadcast at i...)` written to slot `i`; chunks cover the index range
  contiguously and concatenate in index order ⇒ identical output `Vec`. No reduction.
- Error path: chunk results folded with `?` in chunk (=index) order ⇒ first failing index's
  `SpecialError`, exactly as the serial `.iter().map(...).collect()`.
- The scalar arm uses the same `eval`; the `order_vec×x_vec` length check + complex arms are
  untouched. Gate: serial for `< 256` elements.

⇒ The returned value (and first error) is bit-identical to the serial implementation.

## Proof (golden — serial vs parallel, identical)
Harness: `cargo run --profile release-perf -p fsci-special --bin perf_jv_array`
(order=2.5, z∈(0.1,50.1))
```
n=100   jv=e047bd4356bc9aac kv=61bfbdfbe450e3cc
n=5000  jv=4e57121f5e0c4051 kv=1d1bb7af14f82212
n=50000 jv=01a9bcef59522bbe kv=9e28968498630f4a
timing acc: jv 500k=16129554d2fefa7d  kv 100k=1036f70118c61d43
```
Identical in the stashed serial build and the parallel build.
sha256(golden payload file) = aa4d4e5a37e1e4a0cf7ea7e9904c82b2c0936d946dc37948ec963f1f022a56e9

## Timing (rch remote, release-perf, 3 back-to-back runs each)
| function / array | serial (3x)          | new (3x)             | speedup |
|------------------|----------------------|----------------------|---------|
| jv,  500k        | 100.7/99.1/97.3 ms   | 8.81/9.23/10.71 ms   | ~10.8x  |
| kv,  100k        | 5.754/5.667/5.880 s  | 232.1/226.1/234.0 ms | ~25x    |

(kv arbitrary-order is ~57µs/element serial — hence the large ratio.)

## Validation
103 bessel unit tests pass; clippy: no warning in bessel.rs.
