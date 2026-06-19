import time, numpy as np
from scipy.signal import lfilter, filtfilt, sosfilt, medfilt, firls, remez
def sig(n): return np.array([np.sin(37*i/n)+0.35*np.cos(91*i/n)+0.1*((i*17%29)-14) for i in range(n)])
def med(fn,r=9):
    ts=[]
    for _ in range(r):
        t0=time.perf_counter(); fn(); ts.append(time.perf_counter()-t0)
    return sorted(ts)[len(ts)//2]
x=sig(4096); b=[0.06745527,0.13491055,0.06745527]; a=[1.0,-1.1429805,0.4128016]
sos=np.array([[0.06745527,0.13491055,0.06745527,1.0,-1.1429805,0.4128016]]*2)
print("scipy lfilter 4096: %.1f us"%(med(lambda: lfilter(b,a,x))*1e6))
print("scipy filtfilt 4096: %.1f us"%(med(lambda: filtfilt(b,a,x))*1e6))
print("scipy sosfilt 4096x2: %.1f us"%(med(lambda: sosfilt(sos,x))*1e6))
x8=sig(8192)
for k in [5,15]:
    print("scipy medfilt 8192 k%d: %.1f us"%(k,med(lambda: medfilt(x8,k))*1e6))
print("scipy firls 257: %.1f us"%(med(lambda: firls(257,[0,0.3,0.4,1.0],[1,1,0,0]))*1e6))
