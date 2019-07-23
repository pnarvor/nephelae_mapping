#! /usr/bin/python3

import sys
sys.path.append('../../')
import numpy as np
import numpy.fft as npfft
import matplotlib.pyplot as plt
from   matplotlib import animation
import time

from nephelae_mapping.gprmapping import GprKernel
from nephelae_mapping.gprmapping import NephKernel
from sklearn.gaussian_process import kernels as gpk

kernel0 = NephKernel([1.0,2.0,3.0], noiseVariance = 0.01)
path = 'output/kernel0.nker'
kernel0.save(path, force=True)
kernel1 = NephKernel.load(path)

# N = 512
# span = [-10, 10]
# samplingRate = N / (span[-1] - span[0])
# locations    = np.linspace(span[0], span[-1], N)
# 
# values = kernel.values(locations.reshape(-1,1))
# ft    = np.abs(npfft.fft(values))
# ftmin = min([a for a in ft if a > 0])
# ft[ft < ftmin] = ftmin
# ft = ft / np.max(ft)
# ft = 10*np.log10(ft)
# 
# 
# fig, axes = plt.subplots(2,1)
# axes[0].plot(locations, values, label="kernel values")
# axes[0].grid()
# axes[0].legend(loc="upper right")
# # for ft in kernel.resolutions:
#     # axes[1].plot(np.linspace(0,1,len(ft)), ft, '.')
#     # axes[1].plot(ft, '.')
# axes[1].plot(np.linspace(0, samplingRate * (N-1) / N, N), ft, label="kernel fourier")
# axes[1].grid()
# axes[1].legend(loc="upper right")
# 
# plt.show(block=False)



