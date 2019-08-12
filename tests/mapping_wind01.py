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

class WindKernel:

    """
    Kernel compatible with sklearn.gaussian_process.Kernel
    to be used in GaussianProcessRegressor

    /!\ Hyper parameters for this kernel CANNOT be optimized through sklearn.
    When using with GaussianProcessRegressor, set optimizer=None

    /!\ Only implemented for dimension (t,x) for now for testing purposes.

    """

    # For compatibility
    def clone_with_theta(self, theta):
        return WindKernel(self.tScale, self.xScale,
                          self.stddev, self.noiseStddev,
                          self.windSpeed)


    def get_params(self, deep=True):
        return {}


    def set_params(self, **params):
        return


    # Actually used (maybe)
    def __init__(self, tScale=1.0, xScale=1.0,
                       stddev=1.0, noiseStddev=0.1,
                       windSpeed=0.0):
        self.tScale      = tScale
        self.xScale      = xScale
        self.stddev      = stddev
        self.noiseStddev = noiseStddev
        self.windSpeed   = windSpeed

        # For compatibility
        self.bounds          = np.array([])
        self.hyperparameters = []
        self.n_dims          = 0
        self.theta           = np.array([])

    
    # def evaluate(self, x, y):
    #     return self.stddev*
    
    def __call__(self, X):
        # X assumed to be N*2
        # res = np.diag(self.diag(X))
        # for i,x in enumerate(X[:-1,:]):
        #     for j,y in enumerate(X[i+1:,:]):
        #         res[i,j] = self.evaluate(x,y)

        # Far from most efficient but efficiency requires C++ implementation
        t0,t1 = np.meshgrid(X[:,0], X[:,0])
        dt = t1 - t0

        x0,x1 = np.meshgrid(X[:,1], X[:,1])
        dx = x1 - (x0 + self.windSpeed * dt)

        distMat = (dt / self.tScale)**2 + (dx / self.xScale)**2

        return self.stddev*np.exp(-0.5*distMat) + np.diag([self.noiseStddev]*X.shape[0])


    def diag(self, X):
        return np.array([self.stddev + self.noiseStddev]*X.shape[0])


    def is_stationary(self):
        return True


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
# Kernel from sklearn
kernel0 = ((a**2) * gpk.RBF(length_scale = [l0 / v0, l0])) + gpk.WhiteKernel(noiseStd**2)
gprProcessor0 = GaussianProcessRegressor(kernel0,
                                         alpha=0.0,
                                         optimizer=None,
                                         copy_X_train=False)
gprProcessor0.fit(np.array([tProbe, xProbe]).T, np.array([samples]).T)

# # Custom kernel
# kernel1 = WindKernel(l0 / v0, l0, a, 

# Prediction
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






