# Conclusion

Bead: frankenscipy-8l8r1.17

Decision: reject and restore source.

One lever tested: widen the existing no-pack dense `matmul` full-tile register
block from `MR x NR = 4 x 8` to `8 x 8`. The candidate kept direct B row loads,
used no packing, made no allocations, and left the ragged scalar tail path
unchanged.

## Behavior Proof

- Ordering: each output cell still accumulated `k = 0..n` in monotonic order.
- Tie-breaking: not applicable.
- Floating point: no fast-math, no requested FMA contraction, no parallelism,
  and no changed reduction order per cell.
- RNG: not applicable.
- Golden tests: `matmul_microkernel_golden_digest` and
  `matmul_microkernel_is_bit_identical_to_flat_ikj` passed before and after via
  RCH.
- Sorted normalized golden sha256 matched before and after:
  `96f3c62cd9ae70c54af3deac9a4cb0a52725ae5969bcef5831b9d1c62d361803`.

## Benchmark Evidence

The formal focused pre-edit baseline landed on `vmi1156319`, which was much
slower than the prior clean reprofile worker and is kept as evidence but not
used as a keep signal:

| Case | Focused baseline median on `vmi1156319` |
| --- | ---: |
| `matmul/256x256` | `11.517 ms` |
| `matmul/512x512` | `93.269 ms` |
| `matmul/768x768` | `572.06 ms` |
| `matmul/1024x1024` | `2.6493 s` |

The usable comparator is the clean pre-edit reprofile at commit `f7189fbc` on
`vmi1149989`, compared to the candidate after-run on the same worker:

| Case | Pre-edit 4x8 median | Candidate 8x8 median | Ratio |
| --- | ---: | ---: | ---: |
| `matmul/256x256` | `4.0600 ms` | `3.4996 ms` | `1.16x` |
| `matmul/512x512` | `32.956 ms` | `45.104 ms` | `0.73x` |
| `matmul/768x768` | `123.64 ms` | `150.90 ms` | `0.82x` |
| `matmul/1024x1024` | `321.49 ms` | `426.11 ms` | `0.75x` |

Score: `0.0`. The top profile row regressed, so the source was restored to the
kept `MR x NR = 4 x 8` kernel. `git diff -- crates/fsci-linalg/src/lib.rs` was
empty after restore, and `cargo fmt -p fsci-linalg --check` passed.
