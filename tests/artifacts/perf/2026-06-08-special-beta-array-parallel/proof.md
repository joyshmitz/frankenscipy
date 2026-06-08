# perf: parallelize beta-family real-array evaluation (beta::map_real_binary / ternary)

Bead: frankenscipy-u7crl

## Lever
The beta-family array dispatchers `map_real_binary` (beta, betaln) and `map_real_ternary`
(betainc, betaincc, betainccinv) in crates/fsci-special/src/beta.rs mapped an expensive
per-element kernel — the regularized incomplete-beta continued fraction, the backbone of
the Student-t / F / beta distribution CDFs — serially over the array arms. Added one shared
index-based helper `par_map_indices(n, f)` that evaluates `f(0..n)` in parallel index
chunks, and routed all 6 real-vec arms (binary: vec×scalar, scalar×vec, vec×vec; ternary:
vec/scalar/scalar, scalar/vec/scalar, scalar/scalar/vec) through it.

## Isomorphism / byte-identity argument
- Each output index `i` is `f(i)` written to slot `i`; chunks cover `0..n` contiguously and
  concatenate in index order ⇒ identical output `Vec`. No reduction.
- Error path: chunk results folded with `?` in chunk (=index) order ⇒ the first failing
  index's `SpecialError` is returned, exactly as the serial `(0..n).map(f).collect()`.
  (Parallel may evaluate elements past the first error in other chunks; that only affects
  Hardened-mode diagnostic trace emission, never the returned value/error.)
- The vec×vec length-mismatch check and all complex/empty fail-closed arms are untouched.
- Gate: serial for `< 256` elements.

⇒ The returned value (and first error) is bit-identical to the serial implementation.

## Proof (golden — serial vs parallel, identical)
Harness: `cargo run --profile release-perf -p fsci-special --bin perf_beta_array`
```
n=100   betainc xor=e3af7f242be4f7ac  beta xor=7d22fa80924083c3
n=5000  betainc xor=1de0fae8f11d3a15  beta xor=62d2279237750c5f
n=50000 betainc xor=b6df97cb56286e63  beta xor=2b2d0521bb80fc47
timing acc (betainc): 300k=f1ef90f45b3767eb  600k=64462edf3621f099
```
Identical in the stashed serial build and the parallel build.
sha256(golden payload file) = aba3a4bcfca78ddd82681222c72ba5a7fb26e57133ee4472031ea433314f3648

## Timing (rch remote, release-perf, 3 back-to-back runs each) — betainc (a=2.5,b=3.5, x∈(0,1))
| array | serial (3x)            | new (3x)            | speedup |
|-------|------------------------|---------------------|---------|
| 300k  | 71.6/66.0/71.0 ms      | 8.96/9.72/10.41 ms  | ~7.3x   |
| 600k  | 133.9/140.2/131.8 ms   | 11.68/10.96/12.04 ms| ~11.5x  |

## Validation
74 beta unit tests pass; clippy: no new warning from the new helper/arms (the 2 pre-existing
beta.rs warnings are in gamma_sign at lines ~1211/1213).
