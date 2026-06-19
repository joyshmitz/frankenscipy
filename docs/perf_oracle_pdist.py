#!/usr/bin/env python3
"""Oracle: scipy.spatial.distance.pdist on the SAME data the fsci-spatial bench
(bench_pdist) uses. Mirrors crates/fsci-spatial/benches/spatial_bench.rs (n=256/512,
4-D, euclidean + cosine).
"""
import time
import numpy as np
from scipy.spatial.distance import pdist


def data(n):
    out = np.empty((n, 4))
    for i in range(n):
        t = float(i)
        out[i] = [np.sin(t * 0.1), np.cos(t * 0.2), t * 0.001, np.sin(t * 0.05)]
    return out


def med(fn, reps=9):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


if __name__ == "__main__":
    for n in (256, 512):
        X = data(n)
        te = med(lambda: pdist(X, metric="euclidean"))
        tc = med(lambda: pdist(X, metric="cosine"))
        print(f"scipy pdist euclidean n={n}: {te*1e6:.2f} us")
        print(f"scipy pdist cosine    n={n}: {tc*1e6:.2f} us")
