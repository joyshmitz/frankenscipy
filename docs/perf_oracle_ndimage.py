#!/usr/bin/env python3
"""Oracle: scipy.ndimage correlate (5x5) + gaussian_filter (sigma=2) on a 256x256
image, matching the fsci-ndimage bench (bench_correlate_gaussian). Mirrors the
`image(side)` helper in crates/fsci-ndimage/benches/ndimage_bench.rs.
"""
import time
import numpy as np
from scipy import ndimage


def image(side):
    out = np.empty((side, side))
    for i in range(side * side):
        x = float(i)
        out[i // side, i % side] = np.sin(x * 0.01) + np.cos(x * 0.003) * 0.5
    return out


def med(fn, reps=9):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


if __name__ == "__main__":
    img = image(256)
    w = np.ones((5, 5))
    tc = med(lambda: ndimage.correlate(img, w, mode="reflect"))
    tg = med(lambda: ndimage.gaussian_filter(img, sigma=2.0, mode="reflect"))
    print(f"scipy ndimage correlate 5x5 256x256: {tc*1e6:.2f} us")
    print(f"scipy ndimage gaussian sigma2 256x256: {tg*1e6:.2f} us")
