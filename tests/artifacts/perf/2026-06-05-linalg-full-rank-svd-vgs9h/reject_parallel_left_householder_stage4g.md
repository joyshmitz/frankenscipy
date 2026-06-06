# Reject: Parallel Left-Householder Column Trial Stage 4g

Bead: `frankenscipy-z65tz`

## Profile Target

Fresh RCH reprofile after removing the tiled-4 far-update trial kept the private
`1024x512` Golub-Kahan bidiagonal reducer as the active hotspot.

## Baseline

RCH `ts1`, current Stage 4d reducer:

- shape: `1024x512`
- elapsed: `211.363045 ms`
- digest: `0x90cdd3f8f71ed2c1`
- first diagonal: `-1.00455335940616146e3`
- last diagonal: `-6.45492359226604862e1`

## Lever

Trial parallelized large left-Householder trailing updates across disjoint
column-major chunks using scoped worker threads. Each column kept the same
ascending reflector-value dot-product order and the same ascending update order.

The helper stayed private inside the unwired bidiagonal reducer. Public `svd`,
`svdvals`, `lstsq`, `pinv`, ordering, tie-breaking, rcond/rank thresholds,
certificates, error classes, and RNG behavior were unchanged.

## Behavior Proof

RCH `ts1`:

```text
cargo test -p fsci-linalg --release --lib bidiag_left_parallel_columns_match_sequential_bits --locked -- --nocapture
```

Result: passed. The proof forced four worker chunks on a `384x192` panel and
compared every cell against the original sequential left-Householder column loop
by `f64::to_bits`.

## Rebench

RCH `ts1`, same full private reducer probe:

- baseline: `211.363045 ms`
- candidate: `3157.400256 ms`
- digest: `0x90cdd3f8f71ed2c1`
- speedup: `0.066943x`

## Decision

Reject. Score: `0.0`: per-step scoped thread spawning swamps the independent
column work even though the arithmetic is bit-preserving.

Source was restored before staging. Do not retry per-step thread spawning. The
next structural primitive should amortize across a whole panel: persistent
packed panel buffers feeding a BLAS-3-style trailing update, or a two-stage
communication-avoiding bidiagonalization path. Target remains at least `2.5x`
over Stage 4a with reconstruction proof and public golden-output guards before
public wiring.
