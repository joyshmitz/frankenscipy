#!/usr/bin/env python3
"""Oracle: scipy.spatial.cKDTree build + nearest-neighbour query on the SAME 4096 3-D
points the fsci-spatial bench_kdtree uses. Mirrors spatial_bench.rs::bench_kdtree
(pt(t)=[sin(t*0.1),cos(t*0.2),sin(t*0.05)]; queries at t+0.5).
"""
import time
import numpy as np
from scipy.spatial import cKDTree


def med(fn, r=11):
    ts = []
    for _ in range(r):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


if __name__ == "__main__":
    n = 4096
    data = np.array([[np.sin(t * 0.1), np.cos(t * 0.2), np.sin(t * 0.05)] for t in range(n)])
    queries = np.array([[np.sin((t + 0.5) * 0.1), np.cos((t + 0.5) * 0.2), np.sin((t + 0.5) * 0.05)] for t in range(n)])
    print(f"scipy cKDTree build n={n}: {med(lambda: cKDTree(data))*1e6:.2f} us")
    tree = cKDTree(data)
    print(f"scipy cKDTree query {n} pts: {med(lambda: tree.query(queries))*1e6:.2f} us")
