#! /usr/bin/python3

import sys
sys.path.append('../../')
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import animation
import time

# from nephelae_mapping.gprmapping import GprPredictor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels as gpk


# noiseStd = 1.0
noiseStd = 0.1
# a        = 10.0
a        = 2.0
l0       = 0.5 / (5.0)
T        = 100
# N        = 10
N        = 20
# v0       = 0.1
v0       = 0.3
# v0       = 0.5
tmax     = 10*v0

def process(t,x):
    # return a*np.cos((x - v0*t) / l0)
    return a*(np.exp(-((x-v0*t) / l0)**2 / 2.0))

tProbe  = np.linspace(0.0, tmax / v0, T)
xProbe  = 3.0*l0*np.cos(2*np.pi*0.5*tProbe) + v0*tProbe
ground  = process(tProbe, xProbe)
noise   = noiseStd*np.random.randn(len(ground))
samples = ground + noise


# Kernel definition + fit
kernel0 = (a * gpk.RBF(length_scale = [l0 / v0, l0])) + gpk.WhiteKernel(noiseStd**2)
gprProcessor0 = GaussianProcessRegressor(kernel0,
                                         alpha=0.0,
                                         optimizer=None,
                                         copy_X_train=False)
gprProcessor0.fit(np.array([tProbe, xProbe]).T, np.array([samples]).T)


T0,X0 = np.meshgrid(np.linspace(min(tProbe), max(tProbe), 1024),
                    np.linspace(min(xProbe), max(xProbe), 1024))
map0, std0 = gprProcessor0.predict(np.array([T0.ravel(), X0.ravel()]).T,
                                   return_std=True)
map0 = map0.reshape(T0.shape)
std0 = std0.reshape(T0.shape)


# display only
# figure0
fig, axes = plt.subplots(2,1, sharex=True, sharey=False)
axes[0].imshow(process(T0,X0), origin='lower',
               extent=[T0[0,0], T0[0,-1], X0[0,0], X0[-1,0]], aspect='auto')
axes[0].plot(tProbe, xProbe, 'o', label='sampling locations')
axes[0].set_xlabel('Time (s?)')
axes[0].set_ylabel('Position (m?)')
axes[0].legend(loc='lower right')
axes[0].grid()

axes[1].plot(tProbe,  ground, label='ground truth')
axes[1].plot(tProbe, samples, label='samples')
axes[1].set_xlabel('Time (s?)')
axes[1].set_ylabel('process value (?)')
axes[1].legend(loc='upper right')
axes[1].grid()



# figure1
fig, axes = plt.subplots(2,1, sharex=True, sharey=False)
axes[0].imshow(map0, origin='lower',
               extent=[T0[0,0], T0[0,-1], X0[0,0], X0[-1,0]], aspect='auto')
axes[0].plot(tProbe, xProbe, 'o', label='sampling locations')
axes[0].set_xlabel('Time (s?)')
axes[0].set_ylabel('Position (m?)')
axes[0].legend(loc='lower right')
axes[0].grid()

axes[1].imshow(std0, origin='lower',
               extent=[T0[0,0], T0[0,-1], X0[0,0], X0[-1,0]], aspect='auto')
axes[1].plot(tProbe, xProbe, 'o', label='sampling locations')
axes[1].set_xlabel('Time (s?)')
axes[1].set_ylabel('Position (m?)')
axes[1].legend(loc='lower right')
axes[1].grid()


plt.show(block=False)






