import time, numpy as np, scipy.linalg as sla, os
n_list = [1000, 2048]
print(f"threads_env OMP={os.environ.get('OMP_NUM_THREADS','default')}")
for n in n_list:
    rng = np.random.default_rng(42)
    m = rng.standard_normal((n, n))
    a = m @ m.T + n * np.eye(n)
    # warm
    for _ in range(3): sla.cho_factor(a, lower=True)
    reps = 40 if n == 1000 else 12
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        sla.cho_factor(a, lower=True)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p50 = times[len(times)//2]
    gf = n**3/3/(p50*1e6)
    print(f"scipy cho_factor n={n} reps={reps} p50_ms={p50:.3f} min_ms={times[0]:.3f} gflops_p50={gf:.2f}")
