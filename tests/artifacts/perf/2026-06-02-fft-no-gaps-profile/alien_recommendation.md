# Alien Recommendation: fsci-fft fft2 Axis Plan Hoist

## Profile Symptom

- Rust `baseline_fft2/fft2/512x512` broad median: 9.3870 ms on `vmi1149989`.
- Focused Rust baseline median: 8.2364 ms on `vmi1149989`.
- Local SciPy reference median for the same 512x512 complex input: 2.496414 ms.
- Residual gap after the shipped lever remains: post-change broad median 7.8781 ms, still about 3.16x local SciPy.

## Graveyard Match

- Canonical primitive: cache-conscious loop-invariant plan metadata hoisting.
- Source families: cache-oblivious locality guidance plus vectorized/batch execution guidance.
- Local symptom match: `apply_axis_transform` repeats identical radix-2 planning work for every row/column fiber even though the axis length and direction are invariant for the batch.

## Recommendation Card

- Shipped lever: build a local per-axis radix-2 plan containing the cached twiddle table and ascending bit-reversal swap list, then replay it for each fiber.
- Fallback: keep the existing backend dispatch path for non-Cooley-Tukey and non-power-of-two axes.
- Expected value score: Impact 3.0 x Confidence 4.0 / Effort 2.0 = 6.0.
- Rejected candidate: direct contiguous row transforms (`stride == 1`) preserved output bits but regressed focused median from 8.2364 ms to 14.343 ms, so it was not shipped.
- Rejected candidate: global bit-reversal swap cache preserved output bits but regressed focused median on its RCH run, so it was not shipped.

## Artifact-Coding Proof Obligations

- Axis order must remain unchanged.
- Bit-reversal swap order must remain the old ascending-`i` order.
- Twiddle values must come from the same `get_or_compute_twiddles(axis_len, inverse)` table.
- Butterfly arithmetic, normalization, validation, audit logging, and error classes must remain unchanged.
- No RNG is involved.
- Golden output must be byte-identical before and after.
