import sys, numpy as np
from scipy.linalg import cho_factor
n = int(sys.argv[1]); iters = int(sys.argv[2])
rng = np.random.default_rng(7); m = rng.standard_normal((n, n))
a = (m @ m.T)/n + n*np.eye(n)
for _ in range(iters):
    cho_factor(a, lower=True, check_finite=False, overwrite_a=False)
