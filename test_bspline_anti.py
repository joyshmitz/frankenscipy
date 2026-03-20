import numpy as np
from scipy.interpolate import BSpline

t = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0]
c = [1.0, 2.0, 3.0, 4.0, 0.0]
k = 2
b = BSpline(t, c, k)
b_anti = b.antiderivative(1)
print("scipy anti c:", b_anti.c)
print("scipy anti t:", b_anti.t)
