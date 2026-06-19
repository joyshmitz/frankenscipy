#!/usr/bin/env python3
"""SciPy oracle timing for distance_transform_edt(return_indices=True) — the
path fsci optimized from O(foreground*background) brute force to the exact
separable feature transform O(N*ndim). Random 50%-foreground square images.

Usage: python3 docs/perf_oracle_edt_indices.py [--reps N]
"""
import argparse
import statistics
import time

import numpy as np
from scipy import ndimage


def bench(fn, reps, warmups=2):
    for _ in range(warmups):
        fn()
    s = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        s.append(time.perf_counter() - t0)
    s.sort()
    return statistics.median(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    rng = np.random.default_rng(0xC0DE)
    print(f"scipy.ndimage.distance_transform_edt(return_indices=True), reps={args.reps}")
    for side in (64, 128, 192, 256):
        # 50% foreground (nonzero); scipy measures distance to nearest zero.
        img = (rng.random((side, side)) < 0.5).astype(np.float64)
        p50 = bench(
            lambda: ndimage.distance_transform_edt(img, return_indices=True),
            args.reps,
        )
        print(f"  {side}x{side} (N={side*side:>6})  p50 = {p50*1e6:12.3f} us")


if __name__ == "__main__":
    main()
