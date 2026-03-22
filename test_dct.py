import numpy as np
import scipy.fft as fft

x = np.array([1.0, 2.0, 3.0, 4.0])
print("dct1:", fft.dct(x, type=1))
print("dct2:", fft.dct(x, type=2))
print("dct3:", fft.dct(x, type=3))
print("dct4:", fft.dct(x, type=4))
print("idct2:", fft.idct(x, type=2))
