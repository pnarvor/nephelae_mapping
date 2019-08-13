#! /usr/bin/python3

import sys
sys.path.append('../../')
import numpy as np
import numpy.fft as npfft
import matplotlib.pyplot as plt
from   matplotlib import animation
import time

from netCDF4 import MFDataset
from nephelae_simulation.mesonh_interface import MesoNHVariable
from nephelae_base.types import Position
from nephelae_base.types import Bounds

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels as gpk


class WindKernel(gpk.Kernel):

    """
    Kernel compatible with sklearn.gaussian_process.Kernel
    to be used in GaussianProcessRegressor

    /!\ Hyper parameters optimizatin HAS NOT BEEN TESTED
    When using with GaussianProcessRegressor, set optimizer=None

    /!\ Only implemented for dimension (t,x,y) for now for testing purposes.

    """


    # Actually used (maybe)
    def __init__(self, lengthScale=[1.0,1.0,1.0],
                       stddev=1.0, noiseStddev=0.1,
                       windSpeed=[0.0,0.0]):
        self.lengthScale = lengthScale
        self.stddev      = stddev
        self.noiseStddev = noiseStddev
        self.windSpeed   = windSpeed

    
    def __call__(self, X, Y=None):

        if Y is None:
            Y = X

        # print("X shape: ", X.shape)
        # print("Y shape: ", X.shape, end="\n\n")

        # Far from most efficient but efficiency requires C++ implementation
        t0,t1 = np.meshgrid(X[:,0], Y[:,0], indexing='ij')
        dt = t1 - t0
        distMat = (dt / self.lengthScale[0])**2

        x0,x1 = np.meshgrid(X[:,1], Y[:,1], indexing='ij')
        dx = x1 - (x0 + self.windSpeed[0] * dt)
        distMat = distMat + (dx / self.lengthScale[1])**2

        x0,x1 = np.meshgrid(X[:,2], Y[:,2], indexing='ij')
        dx = x1 - (x0 + self.windSpeed[1] * dt)
        distMat = distMat + (dx / self.lengthScale[2])**2

        if Y is X:
            return self.stddev*np.exp(-0.5*distMat) + np.diag([self.noiseStddev]*X.shape[0])
        else:
            return self.stddev*np.exp(-0.5*distMat)


    def diag(self, X):
        return np.array([self.stddev + self.noiseStddev]*X.shape[0])


    def is_stationary(self):
        return True


mesonhPath = '/home/pnarvor/work/nephelae/data/MesoNH-2019-02/REFHR.1.ARMCu.4D.nc'
rct = MesoNHVariable(MFDataset(mesonhPath), 'RCT')

# Estimating advective wind
ut = MesoNHVariable(MFDataset(mesonhPath), 'UT')[50.0, 1100.0,:,:].data.mean()
vt = MesoNHVariable(MFDataset(mesonhPath), 'VT')[50.0, 1100.0,:,:].data.mean()
print("Advective wind :", [ut, vt])

rctSlice = rct[240,1100,:,:].data
print("Variance : ", (rctSlice**2).mean())

t = np.linspace(0,300.0,300)
# a0 = 400.0
a0 = 250.0
f0 = - 1 / 120.0
# f0 = 1 / 150.0

a1 = 0.0
# f1 = 1.5*f0
f1 = 2.5*f0
# f1 = -1.3*f0
# f1 = -2.5*f0
# f1 = -4.5*f0

tStart = 50.0
tEnd   = 700.0
t = np.linspace(tStart, tEnd, int(tEnd - tStart))
# p0 = Position(240.0, 1700.0, 2000.0, 1100.0)
# p0 = Position(50.0, 0.0, 2000.0, 1100.0)
p0 = Position(50.0, 100.0, 1950.0, 1100.0)
p  = np.array([[p0.t, p0.x, p0.y, p0.z]]*len(t))
# v0 = np.array([[9.09, 0.68]])
v0 = np.array([8.5, 0.9])

p[:,0] = t
p[:,1] = p[:,1] + a0*(a1 + np.cos(2*np.pi*f1*(t-t[0])))*np.cos(2*np.pi*f0*(t-t[0]))
p[:,2] = p[:,2] + a0*(a1 + np.cos(2*np.pi*f1*(t-t[0])))*np.sin(2*np.pi*f0*(t-t[0]))
print("Max velocity relative to wind :",
    max(np.sqrt(np.sum((p[1:,1:3] - p[:-1,1:3])**2, axis=1)) / (p[1:,0] - p[:-1,0])))
p[:,1:3] = p[:,1:3] + (t - tStart).reshape([len(t), 1]) @ v0.reshape([1,2]) 

# building prediction locations
# X0,Y0 = np.meshgrid(
#     np.linspace(rct.bounds[3][0], rct.bounds[3][-1], rct.shape[3]),
#     np.linspace(rct.bounds[2][0], rct.bounds[2][-1], rct.shape[2]))
b = rct.bounds
yBounds = [min(p[:,2]), max(p[:,2])]
tmp = rct[p0.t,p0.z,yBounds[0]:yBounds[1],:]
X0,Y0 = np.meshgrid(
    np.linspace(tmp.bounds[1][0], tmp.bounds[1][-1], tmp.shape[1]),
    np.linspace(tmp.bounds[0][0], tmp.bounds[0][-1], tmp.shape[0]))
xyLocations = np.array([[0]*X0.shape[0]*X0.shape[1], X0.ravel(), Y0.ravel()]).T
b[2].min = yBounds[0]
b[2].max = yBounds[1]

# Kernel
processVariance    = 1.0e-8
noiseStddev = 0.1 * np.sqrt(processVariance)
# lengthScales = [100, 50, 50]
# lengthScales = [70, 50, 50]
lengthScales = [70, 60, 60]
# lengthScales = [140, 120, 120]
kernel0 = WindKernel(lengthScales, processVariance, noiseStddev**2, v0)

rctValues = []
print("Getting rct values... ", end='')
sys.stdout.flush()
for pos in p:
    rctValues.append(rct[pos[0],pos[3],pos[2],pos[1]])
rctValues = np.array(rctValues)
print("Done !")
sys.stdout.flush()
noise = noiseStddev*np.random.randn(rctValues.shape[0])
rctValues = rctValues + noise

# # plotting rct values
# fig, axes = plt.subplots(1,1)
# axes.plot(p[:,0], np.array(rctValues))

profiling = False
if not profiling:
    fig, axes = plt.subplots(3,1,sharex=True,sharey=True)
simTime = p0.t
lastTime = time.time()
simSpeed = 50.0

def do_update(t):

    print("Sim time :", t)
    # prediction
    gprProcessor0 = GaussianProcessRegressor(kernel0,
                                             alpha=0.0,
                                             optimizer=None,
                                             copy_X_train=False)
    # trainSet = np.array([list(pos) + [rctVal] \
    #                     for pos, rctVal in zip(p[:,0:3],rctValues)\
    #                     if pos[0] < t and pos[0] > t - 2*lengthScales[0]])
    trainSet = np.array([list(pos) + [rctVal] \
                        for pos, rctVal in zip(p[:,0:3],rctValues)\
                        if pos[0] < t and pos[0] > t - 3*lengthScales[0]])
    print("Number of used measures samples :", trainSet.shape[0])
    gprProcessor0.fit(trainSet[:,:-1], trainSet[:,-1])

    xyLocations[:,0] = t
    map0, std0 = gprProcessor0.predict(xyLocations, return_std=True)
    map0[map0 < 0.0] = 0.0
    map0 = map0.reshape(X0.shape)
    std0 = std0.reshape(X0.shape)
   
    # display
    if not profiling:
        global axes
        axes[0].cla()
        axes[0].imshow(rct[t,p0.z,yBounds[0]:yBounds[1],:].data, origin='lower',
                       extent=[b[3].min, b[3].max, b[2].min, b[2].max])
        axes[0].grid()
        axes[0].set_title("Ground truth")

        try:
            axes[0].plot(p[:int(t-tStart + 0.5),1], p[:int(t-tStart + 0.5),2], '.')
        finally:
            pass

        axes[1].cla()
        axes[1].imshow(map0, origin='lower',
                       extent=[b[3].min, b[3].max, b[2].min, b[2].max])
        axes[1].grid()
        axes[1].set_title("MAP")

        axes[2].cla()
        axes[2].imshow(std0**2, origin='lower',
                       extent=[b[3].min, b[3].max, b[2].min, b[2].max])
        axes[2].grid()
        axes[2].set_title("Variance AP")

def init():
    pass
def update(i):

    # global lastTime
    global simTime
    # currentTime = time.time()
    # simTime = simTime + simSpeed*(currentTime - lastTime)
    # lastTime = currentTime
    # simTime = simTime + 5.0
    simTime = simTime + 2.0

    do_update(simTime)

if not profiling:
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        interval = 1)
    
    plt.show(block=False)
else:
    while simTime < 600:
        update(0)

