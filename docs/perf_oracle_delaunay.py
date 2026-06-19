#!/usr/bin/env python3
"""Oracle: scipy.spatial.Delaunay (Qhull) on the SAME deterministic 2-D points the
fsci-spatial bench_delaunay uses (n=1000/2000). Mirrors the point generator in
spatial_bench.rs::bench_delaunay.
"""
import time
import numpy as np
from scipy.spatial import Delaunay


def med(fn, r=9):
    ts = []
    for _ in range(r):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


if __name__ == "__main__":
    for n in (1000, 2000):
        pts = np.array([[(t * 0.6180339887) % 1.0 * 100.0, (t * 0.4142135624) % 1.0 * 100.0]
                        for t in range(n)])
        print(f"scipy Delaunay (Qhull) n={n}: {med(lambda: Delaunay(pts))*1e6:.2f} us")
