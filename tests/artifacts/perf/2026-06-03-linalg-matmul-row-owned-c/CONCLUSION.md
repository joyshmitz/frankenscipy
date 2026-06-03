# frankenscipy-8l8r1.21 conclusion

## Verdict

Rejected. The row-owned C output layout did not clear the Score gate, and the
source edit was manually restored.

## Baseline and after

Fresh RCH baseline on `vmi1227854`:

| row | baseline median |
| --- | ---: |
| `matmul/256x256` | `5.3319 ms` |
| `matmul/512x512` | `41.678 ms` |
| `matmul/768x768` | `130.94 ms` |
| `matmul/1024x1024` | `348.89 ms` |

RCH after run selected `vmi1153651`:

| row | after median |
| --- | ---: |
| `matmul/256x256` | `12.023 ms` |
| `matmul/512x512` | `116.66 ms` |
| `matmul/768x768` | `819.74 ms` |
| `matmul/1024x1024` | `889.14 ms` |

The after run does not provide a keepable win. Score: rejected below `2.0`.

## Behavior proof

- RCH release `matmul` tests passed before and after.
- Sorted test-line SHA-256 stayed
  `61e12eb58f34ccba1dcedd29425ff3292fd7df5769f7411352cd2a617a58d6c7`.
- `golden_before_after_tests.diff` is empty.

## Restore proof

The attempted source edit to `crates/fsci-linalg/src/lib.rs` was manually
reversed after the rejection. `git diff -- crates/fsci-linalg/src/lib.rs` is
empty.

## Next Primitive

The next target should be a deeper large-GEMM primitive, not another
output-materialization tweak. The profile-backed target remains
`matmul/1024x1024`.
