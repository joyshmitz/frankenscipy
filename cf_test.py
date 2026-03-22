import math
import scipy.special as sc

def gamma_inc_cf(a, x):
    b0 = x + 1.0 - a
    d = 1.0 / b0
    c = 1e-30
    f = d
    for n in range(1, 200):
        an = n * (a - n)
        bn = x + 1.0 - a + 2.0 * n
        d = bn + an * d
        if d == 0: d = 1e-30
        d = 1.0 / d
        c = bn + an / c
        if c == 0: c = 1e-30
        delta = d * c
        f *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    
    return f * math.exp(-x + a * math.log(x) - math.lgamma(a))

a = 10.0
x = 15.0
print("cf:", gamma_inc_cf(a, x))
print("scipy Q(a, x):", sc.gammaincc(a, x))
print("scipy sf(x) for chi2(20):", sc.chdtrc(20, x*2))
