#! /usr/bin/python3

import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt

lmin = 1.0
lmax = 100.0
N = 20
lengthScales = np.exp(np.log(lmax / lmin) * np.linspace(0.0, 1.0, N) + np.log(lmin))

T = 100000
a = 100.0
t = np.linspace(-a*lmax,a*lmax, T)

fs = (T - 1) / (2*a*lmax)
f = np.linspace(0,(T-1)*fs / T, T)

rbf = np.empty([T,N])
RBF = np.empty([T,N])
fc = []
th = -60
for n in range(N):
    rbf[:,n] = np.exp(-0.5*(t / lengthScales[n])**2)
    RBF[:,n] = np.abs(fft(rbf[:,n]))**2
    RBF[:,n] = RBF[:,n] / lengthScales[n]**2
    # RBF[RBF[:,n] < 1.0e-14*RBF[0,n],n] = 1.0e-14*RBF[0,n]
    RBF[:,n] = RBF[:,n] / RBF[0,n]
    RBF[:,n] = 10*np.log10(RBF[:,n])
    fc.append(np.where(RBF[:,n] < th)[0][0]*fs / T)
res = 0.5 / np.array(fc)
scales = np.array(lengthScales)
print("Ratio resolution/lengthScale :", res.dot(scales) / scales.dot(scales))


fig, axes = plt.subplots(3,1)
axes[0].plot(t,rbf,'-*')
axes[0].grid()
axes[1].plot(f, RBF,'-*')
axes[1].grid()
axes[2].plot(lengthScales, res,'-*')
axes[2].grid()
axes[2].set_xlabel("length scale")
axes[2].set_ylabel("resolution")

plt.show(block=False)

