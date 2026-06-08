# perf: parallelize Bessel binary/ternary real-array dispatch (jn/yn/wright_bessel)

Bead: frankenscipy-15xpg

## Lever
The Bessel binary dispatcher `map_real_binary` (jn, yn) and ternary `map_real_ternary`
(wright_bessel, log_wright_bessel) in crates/fsci-special/src/bessel.rs mapped their
per-element kernels (jn/yn integer-order recurrence; wright_bessel ~1.5µs/elt series)
serially over the real-vec broadcast arms. Added a generic index-based `par_map_indices`
helper and routed all real-vec arms (binary: vec×scalar, scalar×vec, vec×vec zip; ternary:
the three single-vec arms) through it. This completes the bessel real-array family (the
unary `map_real_input` for j0/j1/y0/y1/i0/i1/k0/k1 was parallelized earlier).

## Isomorphism / byte-identity argument
- Each output index `i` is `kernel(...values[i]...)` written to slot `i`; chunks cover the
  index range contiguously and concatenate in index order ⇒ identical output `Vec`. No
  reduction.
- Error path: chunk results folded with `?` in chunk (=index) order ⇒ first failing index's
  `SpecialError`, exactly as the serial `.iter().map(kernel).collect()`.
- vec×vec length-mismatch check + complex/empty fail-closed arms untouched. Gate: serial
  for `< 256` elements.

⇒ The returned value (and first error) is bit-identical to the serial implementation.

## Proof (golden — serial vs parallel, identical)
Harness: `cargo run --profile release-perf -p fsci-special --bin perf_bessel_bt_array`
```
n=100   jn=37c81b7a478212a1 wright_bessel=7ad6adc6b0cf3e6e
n=5000  jn=27f7244a713810ca wright_bessel=85774ed057c43e0c
n=50000 jn=929ccd9941826c2f wright_bessel=35649479dfe1ac04
timing acc: jn 2M=37b846bb88af0d72  wright_bessel 500k=850a43d543f8897d
```
Identical in the stashed serial build and the parallel build.
sha256(golden payload file) = f40920ddde00d105180d889f0f9a5a909cc0948e2fe0abcb10ff86efb1724805

## Timing (rch remote, release-perf, 3 back-to-back runs each)
| function / array        | serial (3x)             | new (3x)             | speedup |
|-------------------------|-------------------------|----------------------|---------|
| jn,           2M (ord 7)| 209.8/213.7/226.5 ms    | 18.93/20.04/19.29 ms | ~11.1x  |
| wright_bessel, 500k     | 715.8/737.9/748.8 ms    | 30.05/32.22/30.19 ms | ~24.4x  |

## Validation
103 bessel unit tests pass; clippy: no warning in bessel.rs (helper + arms clean).

## NOTE (separate reject, not committed)
Also evaluated parallelizing linalg `kron`: REJECTED at ~1.06–1.17x — it is one multiply +
write per output cell over a ~134MB result (plus serial zero-init), i.e. memory-bandwidth
bound, so parallelism does not help. The clean embarrassingly-parallel linalg ops
(kron/hadamard_product/outer) are all bandwidth-bound; GEMM (compute-bound) is already
parallel. The change was reverted.
