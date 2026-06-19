#!/usr/bin/env python3
"""Oracle: scipy.stats vectorized pdf/pmf over arrays, matching the fsci-stats
bench_distribution_batch workloads (GammaDist/BetaDist/Hypergeometric, n=4096 /
full support). Mirrors crates/fsci-stats/benches/stats_bench.rs.
"""
import time
import numpy as np
from scipy import stats


def med(fn, reps=9):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


if __name__ == "__main__":
    n = 4096
    # GammaDist::new(2.7, 1.5) -> scipy gamma(a=2.7, scale=1/1.5) (fsci rate=1.5)
    gx = np.array([0.01 + i * 0.01 for i in range(n)])
    t = med(lambda: stats.gamma.pdf(gx, 2.7, scale=1.0 / 1.5))
    print(f"scipy gamma.pdf  n={n}: {t*1e6:.2f} us")

    # BetaDist::new(2.5, 3.5)
    bx = np.array([(i + 0.5) / n for i in range(n)])
    t = med(lambda: stats.beta.pdf(bx, 2.5, 3.5))
    print(f"scipy beta.pdf   n={n}: {t*1e6:.2f} us")

    # Hypergeometric::new(2000, 700, 1200) over full support k=0..700
    # scipy hypergeom(M=population, n=successes, N=draws)
    ks = np.arange(0, 701)
    t = med(lambda: stats.hypergeom.pmf(ks, 2000, 700, 1200))
    print(f"scipy hypergeom.pmf supp=701: {t*1e6:.2f} us")
