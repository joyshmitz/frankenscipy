import math
import scipy.special as sc

def gamma_inc_cf(a, x):
    # evaluate continued fraction using Lentz's method
    # f = b0 + a1/b1 + a2/b2 + ...
    # Here, Q(a,x) = exp(-x) x^a / Gamma(a) * CF
    # CF = 1 / (x + (1-a) / (1 + 1 / (x + (2-a) / (1 + 2 / (x + ...)))))
    
    c = 1e-30
    d = 1.0 / x # b1 is x
    if d == 0:
        d = 1e-30
    f = d
    
    for m in range(1, 100):
        # Even step (n = 2m)
        an = m
        bn = x
        d = bn + an * d
        if d == 0:
            d = 1e-30
        d = 1.0 / d
        c = bn + an / c
        if c == 0:
            c = 1e-30
        delta = c * d
        f *= delta
        
        # Odd step (n = 2m + 1)
        an = m + 1 - a
        bn = 1.0
        d = bn + an * d
        if d == 0:
            d = 1e-30
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
print("scipy Q(a, x):", sc.gammaincc(a, x))
