# Conclusion

Bead: frankenscipy-8l8r1.16

Decision: keep.

One lever: widen the existing no-pack dense `matmul` full-tile register block
from `MR x NR = 4 x 4` to `4 x 8`. There is no B packing, no allocation, no API
change, no error-policy change, and the ragged-edge scalar path is unchanged.

## RCH Criterion

Command:

```text
rch exec -- cargo bench -p fsci-linalg --bench linalg_bench --locked -- matmul --warm-up-time 1 --measurement-time 5 --sample-size 10 --noplot
```

| Case | Before median | After median | Speedup |
| --- | ---: | ---: | ---: |
| `matmul/256x256` | `9.6564 ms` | `5.0341 ms` | `1.92x` |
| `matmul/512x512` | `91.706 ms` | `39.532 ms` | `2.32x` |
| `matmul/768x768` | `690.64 ms` | `134.11 ms` | `5.15x` |
| `matmul/1024x1024` | `1.5452 s` | `410.31 ms` | `3.77x` |

Score: `Impact 4 x Confidence 3 / Effort 1 = 12.0`.

## Behavior Proof

- Ordering: each output cell still accumulates `k = 0..n` in the same monotonic
  order. The lever computes more adjacent columns per full tile, but does not
  reassociate a cell's floating-point sum.
- Tie-breaking: not applicable. `matmul` has no ordering comparisons or ties.
- Floating point: no fast-math, no fused multiply-add request, no parallelism,
  and no changed reduction order per cell.
- RNG: not applicable. The production path does not use randomness.
- Golden sha256: sorted normalized before/after output matches:
  `96f3c62cd9ae70c54af3deac9a4cb0a52725ae5969bcef5831b9d1c62d361803`.
- Golden tests: `matmul_microkernel_golden_digest` and
  `matmul_microkernel_is_bit_identical_to_flat_ikj` passed before and after via
  RCH.

## Validation

- `cargo fmt -p fsci-linalg --check`: passed.
- `rch exec -- cargo test -p fsci-linalg --release --locked matmul_microkernel -- --nocapture`: passed before and after.
- `rch exec -- cargo check -p fsci-linalg --all-targets --locked`: passed.
- `rch exec -- cargo clippy -p fsci-linalg --all-targets --locked -- -D warnings`: passed.
- `ubs crates/fsci-linalg/src/lib.rs`: exit 0, 0 critical findings. Existing
  broad-file warning inventory was not changed by this lever.
