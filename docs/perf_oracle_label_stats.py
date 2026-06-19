#!/usr/bin/env python3
"""SciPy oracle timing for label-indexed measurements (ndimage.mean) over many
labels — the workload where fsci's old O(N*K) `position` scan blew up and the
new O(N+K) label->position map fixes it.

Usage: python3 docs/perf_oracle_label_stats.py [--reps N]
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
    ap.add_argument("--reps", type=int, default=15)
    args = ap.parse_args()
    print(f"scipy.ndimage.mean(labels, index) over K labels, reps={args.reps}")
    for side, k in [(256, 512), (512, 1024), (512, 2048), (768, 4096)]:
        rng = np.random.default_rng(0xA11CE ^ side ^ (k << 4))
        n = side * side
        labels = rng.integers(1, k + 1, size=(side, side))
        values = rng.random((side, side))
        index = np.arange(1, k + 1)
        p50 = bench(lambda: ndimage.mean(values, labels, index), args.reps)
        print(f"  N={n:>7} K={k:>5}  p50 = {p50*1e3:10.3f} ms")


if __name__ == "__main__":
    main()
