import numpy as np
from scipy.interpolate import BSpline

t = np.array([0, 0, 0, 1, 1, 1], dtype=float)
c = np.array([1.0, 2.0, 3.0], dtype=float)
k = 2
spl = BSpline(t, c, k)
anti = spl.antiderivative()
print("Knots:", anti.t)
print("Coeffs:", anti.c)
print("Degree:", anti.k)
