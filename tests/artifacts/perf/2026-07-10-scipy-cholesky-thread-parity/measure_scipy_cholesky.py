import time, os, platform, hashlib
import numpy as np
from scipy.linalg import cho_factor
def stats(v):
    m=sum(v)/len(v); sd=(sum((x-m)**2 for x in v)/len(v))**0.5
    return min(v), m, (sd/m*100 if m else 0)
def spd(n, seed=7):
    rng=np.random.default_rng(seed); m=rng.standard_normal((n,n))
    return (m@m.T)/n + n*np.eye(n)
def bench(a,reps):
    t=time.perf_counter()
    for _ in range(reps): cho_factor(a, lower=True, check_finite=False, overwrite_a=True)
    return (time.perf_counter()-t)/reps*1000
print(f"# host={platform.node()} threads={os.environ.get('OMP_NUM_THREADS','default(all)')} "
      f"sha256={hashlib.sha256(open(__file__,'rb').read()).hexdigest()[:16]}")
# banked fsci cho_factor, same box, same Criterion run (parallel fsci)
FSCI={1000:31.945, 2048:None}
for n in (1000,2048):
    a=spd(n); reps,iters=(3,9) if n<1500 else (2,5)
    for _ in range(2): bench(a,reps)
    A,A2=[],[]
    for _ in range(iters):
        A.append(bench(a,reps)); A2.append(bench(a,reps))
    ab,am,acv=stats(A); bb,bm,bcv=stats(A2)
    null=ab/bb; fit = abs(null-1)<=0.03 and bcv<=5.0
    gf = (n**3/3)/(am/1000)/1e9
    extra=""
    if FSCI.get(n): extra=f"  fsci {FSCI[n]} ms => gap {FSCI[n]/am:.2f}x"
    print(f"{'OK   ' if fit else 'NOISE'} n={n:4d}: best {ab:8.3f} ms mean {am:8.3f} cv {acv:4.1f}% "
          f"NULL {null:.3f}x (cv {bcv:4.1f}%)  {gf:6.1f} GF/s{extra}")
