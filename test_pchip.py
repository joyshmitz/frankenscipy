import numpy as np
from scipy.interpolate import pchip

x = np.array([0.0, 1.0, 3.0])
y = np.array([0.0, 1.0, 4.0]) # delta1 = 1.0 (h1=1.0), delta2 = 1.5 (h2=2.0)
pc = pchip(x, y)
print("scipy derivatives:", pc(x, nu=1))

h = np.diff(x)
delta = np.diff(y)/h
w1 = 2*h[1] + h[0] # 2*2 + 1 = 5
w2 = h[1] + 2*h[0] # 2 + 2 = 4
d1 = (w1 + w2) / (w1 / delta[0] + w2 / delta[1])
print("fsci-interpolate deriv at x=1:", d1)
