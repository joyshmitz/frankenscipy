import math
import scipy.special as sc

def gamma_inc_cf(a, x):
    # evaluate continued fraction using Lentz's method
    # f = b0 + a1/b1 + a2/b2 + ...
    # CF = 1 / (b0 + a1/b1 + a2/b2 + ...)
    
    b0 = x + 1.0 - a
    c = 1.0 / 1e-30 # effectively infinity, or 1e30
    d = 1.0 / b0
    if d == 0: d = 1e-30
    f = d
    
    for m in range(1, 100):
        an = m * (a - m)
        bn = x + 1.0 - a + 2.0 * m
        
        d = bn + an * d
        if d == 0: d = 1e-30
        d = 1.0 / d
        
        c = bn + an / c
        if c == 0:
            c = 1e-30
        
        delta = c * d
        f *= delta
        
        if abs(delta - 1.0) < 1e-15:
            break
            
    return f * math.exp(-x + a * math.log(x) - math.lgamma(a))

a = 10.0
x = 15.0
print("cf:", gamma_inc_cf(a, x))
print("scipy:", sc.gammaincc(a, x))
x))
