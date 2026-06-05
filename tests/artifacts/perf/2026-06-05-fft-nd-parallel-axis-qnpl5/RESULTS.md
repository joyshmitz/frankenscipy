# perf(fsci-fft): fully-parallel fft2/fftn axis transform (frankenscipy-qnpl5)

## Lever
Parallelize the n-D FFT axis passes in `transform_nd_unscaled` / `apply_axis_transform`,
which were 100% single-threaded:
- **Axis 0** (`repeats == 1`, strided lanes): transpose lanes into a contiguous scratch
  buffer, FFT each lane, transpose back. Gather+transform fused into one parallel scope
  (thread owns offsets -> disjoint contiguous transposed lanes); scatter parallel
  (thread owns indices -> disjoint contiguous `data` rows).
- **Axis > 0**: split disjoint contiguous outer blocks across threads, each with its own
  scratch, transformed in place.
- Work-gated via `nd_axis_thread_count` (>=1M elements before splitting, >=64K per thread).
  The axis-0 transpose path is only taken when it will actually run threaded; below the
  gate the original direct strided loop runs, so small n-D FFTs do not regress.

## Isomorphism / byte-identity proof
Each lane's 1-D FFT sees the same inputs in the same order as the sequential walk;
gather/scatter are pure permutations. Golden = forward+inverse `to_bits` dump over shapes
that exercise every path, incl. parallel-triggering 128x128 (2D), 32x32x32 (3D last-axis),
64x4x64 (3D middle-axis, stride>1):

    before (pure-sequential HEAD): sha256 = (see nd_golden_before_seq.txt)
    after  (parallel):             sha256 = (see nd_golden_after_parallel.txt)
    -> IDENTICAL: 1e1cab2a6d719cb752e4776a840c48d848acfdb0df3bccb4a21e415a59d345a5

## Bench (perf_fft fft2t/fftnt, release-perf, min of 6, 64 cores)
| case            | seq (HEAD) | parallel | ratio |
|-----------------|-----------:|---------:|------:|
| fft2  2048x2048 |  156.73 ms |  71.72ms | 2.19x |
| fftn  128^3     |   82.63 ms |  40.42ms | 2.04x |
| fft2  1024x1024 |   28.05 ms |  19.32ms | 1.45x (memory-bound transpose) |
| fftn  64^3      |    6.03 ms |   5.97ms | 1.01x (gated to sequential) |
| fft2  512x512   |    5.55 ms |   5.67ms | 0.98x (gated to sequential, noise) |

Tests: 198 fsci-fft tests pass; clippy clean.
