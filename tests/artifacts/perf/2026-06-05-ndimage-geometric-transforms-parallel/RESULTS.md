# perf(fsci-ndimage): parallelize geometric transforms (byte-identical)

## Lever
affine_transform / rotate / zoom / map_coordinates each loop over independent output
pixels, computing sample_interpolated from the read-only prefiltered spline coefficients.
The loops were serial. Parallelize across the output index: reuse the existing
fill_pixels_parallel helper for the NdArray-output transforms (affine/rotate/zoom),
thread::scope over the result Vec for map_coordinates. The one-time prefilter stays serial.

## Byte-identity
Each output pixel is a pure interpolation written to its own flat index (unravel_with_shape
matches NdArray::unravel exactly), so the result is bit-identical to the serial loop.
perf_affine FNV digests OLD(serial)==NEW(parallel) for all four transforms x {400,700,1000}.

## Bench (perf_affine, release-perf, 64 cores, n=400 representative)
| transform     | serial    | parallel | speedup |
|---------------|----------:|---------:|--------:|
| affine        | 152.2 ms  | 19.1 ms  |  8.0x   |
| rotate        | 126.3 ms  | 25.8 ms  |  4.9x   |
| zoom          | 340.6 ms  | 45.9 ms  |  7.4x   |
| map_coordinates| 143.7 ms | 24.3 ms  |  5.9x   |
224 fsci-ndimage tests pass; clippy clean.
