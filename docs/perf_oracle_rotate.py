#!/usr/bin/env python3
"""Oracle: scipy.ndimage.rotate 30deg on a 256x256 image (order 1 and 3), matching
fsci-ndimage bench_rotate. Mirrors image(side) in ndimage_bench.rs.
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


def med(fn, r=7):
    ts = []
    for _ in range(r):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


if __name__ == "__main__":
    img = image(256)
    for order in (1, 3):
        t = med(lambda: ndimage.rotate(img, 30.0, reshape=False, order=order, mode="reflect"))
        print(f"scipy ndimage rotate 30deg 256x256 order={order}: {t*1e6:.2f} us")
