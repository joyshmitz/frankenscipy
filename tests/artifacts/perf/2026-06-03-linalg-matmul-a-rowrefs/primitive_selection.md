# Primitive selection: A row-reference hoist

Bead: `frankenscipy-8l8r1.11`
Target: `fsci-linalg::matmul`, full 4x4 register-tile path.

## Profile target

RCH Criterion continues to identify `matmul` as the linalg hotspot. Fresh current-HEAD baseline on `vmi1153651`:

| Size | Median |
| --- | ---: |
| 256x256 | 12.326 ms |
| 512x512 | 148.22 ms |
| 768x768 | 1.0100 s |
| 1024x1024 | 2.0473 s |

Prior committed profile context still shows a faster worker but the same hotspot shape: 1024x1024 median `394.26 ms`.

## Rejected nearby levers

- Naive row/cache blocking: regressed.
- NC column panel: closed/rejected.
- Packed B panel / 4x8 path: closed/rejected.
- B-flat allocation/copy: rejected by exact paired benchmark under `frankenscipy-8l8r1.10`.

## Selected lever

Hoist the four `A` row references once per full 4x4 tile:

```rust
let a_row0 = &a[i0];
let a_row1 = &a[i0 + 1];
let a_row2 = &a[i0 + 2];
let a_row3 = &a[i0 + 3];
```

Then load `a_rowN[k]` inside the existing `k` loop. Keep direct `B` row indexing, the same 4x4 tile size, the same ragged scalar path, and no allocation or packing.

## Isomorphism obligations

- Shape validation and error order unchanged.
- Output row/column order unchanged.
- For every full-tile output cell, `k` accumulation order remains monotonic `0..ka`.
- Floating-point operation sequence remains separate `a * b` then `+=`; only the source expression for `a` changes from `a[i][k]` to a pre-borrowed row slice.
- RNG, tie-breaking, and global state are absent.

## Score target

Target Score: `3.0 = impact 2 * confidence 3 / effort 2`.

Reject if golden sha changes or if RCH benchmark evidence lacks a real win, especially if any 256/512/768 row materially regresses.
