import time, numpy as np
from scipy.interpolate import RectBivariateSpline
side=32
g=np.array([i/(side-1) for i in range(side)])
z=np.array([[np.sin(xi*6.0)*np.cos(yi*4.0) for yi in g] for xi in g])
rect=RectBivariateSpline(g,g,z,kx=3,ky=3)
q=np.array([0.001+0.998*(i/64) for i in range(64)])
def med(fn,r=9):
    ts=[]
    for _ in range(r):
        t0=time.perf_counter(); fn(); ts.append(time.perf_counter()-t0)
    return sorted(ts)[len(ts)//2]
print("scipy RectBivariateSpline eval_grid 32->64x64: %.1f us"%(med(lambda: rect(q,q,grid=True))*1e6))
