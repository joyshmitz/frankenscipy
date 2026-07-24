# convolve_axes → tap_delta / nd_filter_apply conversion (byte-identical)

Date: 2026-07-23 · Agent: CopperFalcon (cc) · Host: rch remote worker · Release build

## Lever

`ndimage::convolve_axes` was the last axes-mapped N-D filter still on the
pre-`nd_filter_apply` scalar path: for every output pixel × every kernel tap it
allocated `weights.unravel(flat_k)` (a `Vec<usize>`) plus a fresh `in_idx`
`Vec<i64>` (via `.collect()`), then called `get_boundary` per tap. Its siblings
`convolve_with_origins` / `correlate_with_origins` already route through
`nd_filter_apply` (precompute each tap's full-ndim input-index delta ONCE, gather
interior pixels straight from the flat buffer, fall back to `get_boundary` only on
the border). This lever converts `convolve_axes` to the same path — folding the
kernel flip, center offset, and origin into the precomputed `tap_delta`.

## Behavior unchanged (byte-identical)

`convolve_axes_nd_filter_matches_scalar_reference_bitwise`: new path vs the
retained scalar reference (`convolve_axes_scalar_reference`) are compared with
`to_bits()` equality across 10 shape/kernel/axes cases × 5 boundary modes
(Reflect/Constant/Nearest/Wrap/Mirror), including interior-heavy, all-border
(kernel ≥ input), and kernel == input. 0 differing bits. All 11 existing
scipy-parity `convolve*` tests stay green.

## Delta (A/B, same binary, 256×256 image, 5×5 kernel over axes (-2,-1), Reflect)

| arm | time (criterion median) |
|---|---|
| `scalar_reference_5x5/256` (old per-pixel×tap path) | 18.665 ms |
| `tap_delta_5x5/256` (nd_filter_apply) | 1.4247 ms |
| **speedup** | **13.1×** |

Both arms are parallel (`fill_pixels_parallel` / `nd_filter_apply`), so the win is
algorithmic (per-pixel×tap heap alloc + per-tap boundary arithmetic eliminated for
interior pixels), not a threading artifact. Harness: `bench_convolve_axes_ab` in
`crates/fsci-ndimage/benches/ndimage_bench.rs`.
