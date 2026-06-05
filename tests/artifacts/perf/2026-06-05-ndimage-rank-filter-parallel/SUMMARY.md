# fsci-ndimage median/rank filters: single-threaded -> multithreaded output pixels

## Target

median_filter and rank_filter looped `for flat_out in 0..n` computing each output
pixel (gather the size^ndim neighbourhood + select the rank-th value) sequentially.
fsci-ndimage had ZERO threading. Each output pixel depends only on a read-only
neighbourhood of the input -> embarrassingly parallel, byte-identical. (Builds on the
earlier nth-select win 20bbe2b2; this parallelises that loop.)

## Lever (one)

Extract the per-pixel work into `rank_filter_pixel` and a `fill_rank_filter` driver
that splits `output.data` into disjoint `chunks_mut` across threads (std::thread::scope);
each thread computes its pixel range, reading the shared `input`. Gated by
`ndimage_filter_thread_count` (work = pixels * kernel_total): sequential below 2^18.
median = rank `kernel_total/2`; both median_filter_with_origins and
rank_filter_index_with_origins use it.

## Isomorphism / proof (BYTE-IDENTICAL)

output.data[flat_out] = rank_filter_pixel(...) is the identical gather+select
regardless of the owning thread; pixels are disjoint. The existing rank-filter golden
**sha256 f8e053d76d1ccb9c3740b4651faf94ca6d96c585be41888dbba88e92812ff009 is
UNCHANGED** — and its 160x160/kernel-7 case (work 1.25M) exercises the parallel path,
so parallel == sequential bit-for-bit. fsci-ndimage 223 passed / 0 failed; fmt clean.

## Rebench (ndimage_bench, 160x160 image, same worker via stash)

| row | before (seq nth-select) | after (parallel) | speedup |
| --- | ---: | ---: | ---: |
| median_160x160/7 | 46.08 ms | 4.55 ms | 10.1x |
| rank_q25_160x160/7 | 46.07 ms | 4.67 ms | 9.9x |
| median_160x160/15 | 205.48 ms | 12.30 ms | 16.7x |
| rank_q25_160x160/15 | 209.13 ms | 12.05 ms | 17.4x |

Win grows with kernel size (more gather+select work per pixel). Byte-identical,
Score >> 2.0. Same row-parallel vein as cdist (7decfe77) / matmul / inv.
