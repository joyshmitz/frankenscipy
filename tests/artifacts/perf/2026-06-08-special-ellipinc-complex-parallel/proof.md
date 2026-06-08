# perf: parallelize complex elliptic incomplete-integral array arms (ellipkinc/ellipeinc)

Bead: frankenscipy-oqyqa

## Lever
`map_real_or_complex_binary` (crates/fsci-special/src/elliptic.rs) — the binary tensor
dispatcher behind `ellipkinc` (F(φ,m)) and `ellipeinc` (E(φ,m)). Its **real** array arms were
already parallel via `par_map_indices`, but its **8 complex array arms** (Complex×Complex and
mixed Real/Complex; vec×scalar, scalar×vec, vec×vec for each) still mapped `complex_kernel`
serially with `.iter().map(...).collect()`. Added `+ Sync` to the `G` (complex kernel) bound
and routed all 8 complex array arms through the same `par_map_indices` helper. The complex
Carlson-RF/RD kernel is far heavier per element than the real one, so the win is large.
Scalar arms, length checks, and the empty/fail-closed arms are untouched.

## Isomorphism / byte-identity argument
- `complex_kernel` (`ellipkinc_complex_scalar` / `ellipeinc_complex_scalar`) is a pure free
  `fn(Complex64, Complex64) -> Result<Complex64, _>`. Each output index `i` is
  `complex_kernel(...broadcast at i...)` written to slot `i`; for mixed arms the real operand
  is lifted with `Complex64::from_real` at the same index. Chunks cover the index range
  contiguously and concatenate in index order ⇒ identical output `Vec<Complex64>`. No reduction.
- Error path: chunk results folded with `?` in chunk (=index) order ⇒ first failing index's
  `SpecialError`, exactly as the serial `.collect::<Result<Vec<_>,_>>()`.
- vec×vec length checks preserved; gate: serial for `< 256` elements (`par_map_indices`).

⇒ The returned value (and first error) is bit-identical to the serial implementation.

## Proof (golden — serial vs parallel, identical)
Harness: `cargo run --profile release-perf -p fsci-special --bin perf_ellipinc_complex`
(phi = 0.7+0.15i complex scalar, m complex vec, re∈(0,0.8) im∈(-0.2,0.2))
```
n=100 ellipkinc(len=100,xor=d66c7541234890d4) ellipeinc(len=100,xor=e6342e3f74adfa03)
n=5000 ellipkinc(len=5000,xor=9a5a75b89cde019c) ellipeinc(len=5000,xor=b05e872b9d58fbc2)
n=50000 ellipkinc(len=50000,xor=1372b1c0a03ea31e) ellipeinc(len=50000,xor=1023190921811841)
timing acc: ellipkinc 300k=322d0aecf65c8f17  ellipeinc 300k=24bda3e201bdb153
```
Identical in the stashed serial build and the parallel build (golden xor + timing acc match).
sha256(golden_payload.txt) = cc401047059d103f473944284a373fe2cd09e15abdd879cb839ec8cafe1b3dd8

## Timing (rch remote, release-perf, 3 back-to-back runs each)
| function / array | serial (3x)             | new (3x)               | speedup |
|------------------|-------------------------|------------------------|---------|
| ellipkinc, 300k  | 1.132/1.077/1.054 s     | 41.35/45.30/43.63 ms   | ~25x    |
| ellipeinc, 300k  | 1.038/1.004/0.992 s     | 38.86/40.52/38.26 ms   | ~26x    |

## Validation
102 fsci-special ellip-related unit tests pass; clippy: no warning in elliptic.rs.
